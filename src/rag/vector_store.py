"""Indexing and dense search in Qdrant (Docker server or embedded mode).

- Deterministic point ids (UUIDv5 of the ``chunk_id``) -> idempotent re-indexing.
- Payload indexes on the filterable fields (source, risk, article, topics...).
- Strict metadata filtering through :class:`RetrievalFilter`.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.core.config import Settings
from src.core.llm import build_embeddings
from src.core.logging import get_logger
from src.core.types import NonEmptyStr, PositiveInt, validated
from src.models.schemas import LegalChunk, RetrievalFilter

logger = get_logger(__name__)

_KEYWORD_FIELDS = (
    "metadata.source",
    "metadata.risk_category",
    "metadata.unit_type",
    "metadata.annex_number",
    "metadata.chapter_number",
    "metadata.topics",
)
_INTEGER_FIELDS = ("metadata.article_number",)


class StoreHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    reachable: bool
    collection_exists: bool = False
    points: int = 0
    error: str | None = None


def build_qdrant_client(settings: Settings) -> QdrantClient:
    """Server mode when ``QDRANT_URL`` is set, otherwise embedded mode (file or memory)."""
    if settings.qdrant_url:
        api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        return QdrantClient(url=settings.qdrant_url, api_key=api_key, timeout=30)
    if settings.qdrant_path is not None:
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(settings.qdrant_path))
    return QdrantClient(location=":memory:")


def build_filter(flt: RetrievalFilter | None) -> qm.Filter | None:
    """Translate a :class:`RetrievalFilter` into a Qdrant filter (``None`` if empty)."""
    if flt is None:
        return None
    must: list[qm.Condition] = []
    if flt.sources:
        must.append(
            qm.FieldCondition(key="metadata.source", match=qm.MatchAny(any=[s.value for s in flt.sources]))
        )
    if flt.risk_categories:
        must.append(
            qm.FieldCondition(
                key="metadata.risk_category", match=qm.MatchAny(any=[r.value for r in flt.risk_categories])
            )
        )
    if flt.unit_types:
        must.append(
            qm.FieldCondition(
                key="metadata.unit_type", match=qm.MatchAny(any=[u.value for u in flt.unit_types])
            )
        )
    if flt.article_numbers:
        must.append(
            qm.FieldCondition(key="metadata.article_number", match=qm.MatchAny(any=flt.article_numbers))
        )
    if flt.topics:
        must.append(qm.FieldCondition(key="metadata.topics", match=qm.MatchAny(any=flt.topics)))
    return qm.Filter(must=must) if must else None


class LegalVectorStore:
    """Qdrant facade specialised for the legal corpus."""

    def __init__(self, client: QdrantClient, embeddings: Embeddings, collection: str) -> None:
        if not collection:
            raise ValueError("collection name must not be empty")
        self.client = client
        self.embeddings = embeddings
        self.collection = collection

    # ------------------------------------------------------------------ management
    def collection_exists(self) -> bool:
        return bool(self.client.collection_exists(self.collection))

    @validated
    def ensure_collection(self, dim: PositiveInt, *, recreate: bool = False) -> None:
        if recreate and self.collection_exists():
            self.client.delete_collection(self.collection)
        if self.collection_exists():
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        for field_name in _KEYWORD_FIELDS:
            self._create_index(field_name, qm.PayloadSchemaType.KEYWORD)
        for field_name in _INTEGER_FIELDS:
            self._create_index(field_name, qm.PayloadSchemaType.INTEGER)
        logger.info("Qdrant collection '%s' created (dim=%d)", self.collection, dim)

    def _create_index(self, field_name: str, schema: qm.PayloadSchemaType) -> None:
        try:
            with warnings.catch_warnings():  # embedded mode warns that payload indexes are no-ops
                warnings.simplefilter("ignore", UserWarning)
                self.client.create_payload_index(self.collection, field_name=field_name, field_schema=schema)
        except Exception as exc:  # noqa: BLE001 - optional optimisation, never fatal
            logger.debug("Payload index not created (%s): %s", field_name, exc)

    def count(self) -> int:
        if not self.collection_exists():
            return 0
        return int(self.client.count(self.collection, exact=True).count)

    def health(self) -> StoreHealth:
        try:
            exists = self.collection_exists()
            return StoreHealth(reachable=True, collection_exists=exists, points=self.count() if exists else 0)
        except Exception as exc:  # noqa: BLE001 - any network error means "unavailable"
            return StoreHealth(reachable=False, error=str(exc))

    # ------------------------------------------------------------------ indexing
    @validated
    def index_chunks(
        self, chunks: Sequence[LegalChunk], *, recreate: bool = False, batch_size: PositiveInt = 64
    ) -> int:
        """Embed and upsert chunks; returns the number of points written."""
        if not chunks:
            return 0
        probe = self.embeddings.embed_query("dimension probe")
        self.ensure_collection(len(probe), recreate=recreate)
        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.embeddings.embed_documents([c.text for c in batch])
            points = [
                qm.PointStruct(id=c.point_id, vector=v, payload=self._payload(c))
                for c, v in zip(batch, vectors, strict=True)
            ]
            self.client.upsert(self.collection, points=points, wait=True)
            total += len(points)
            logger.info("Indexing: %d/%d", total, len(chunks))
        return total

    @staticmethod
    def _payload(chunk: LegalChunk) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "metadata": chunk.metadata.model_dump(mode="json", exclude={"citation", "unit_key"}),
        }

    # ------------------------------------------------------------------ search
    @validated
    def search(
        self, query: NonEmptyStr, *, top_k: PositiveInt, flt: RetrievalFilter | None = None
    ) -> list[tuple[LegalChunk, float]]:
        """Dense search for ``query`` with optional metadata filter."""
        return self.search_by_vector(self.embeddings.embed_query(query), top_k=top_k, flt=flt)

    def search_by_vector(
        self, vector: list[float], *, top_k: int, flt: RetrievalFilter | None = None
    ) -> list[tuple[LegalChunk, float]]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            query_filter=build_filter(flt),
            with_payload=True,
        )
        results: list[tuple[LegalChunk, float]] = []
        for point in response.points:
            payload = point.payload or {}
            chunk = LegalChunk.model_validate(
                {"chunk_id": payload["chunk_id"], "text": payload["text"], "metadata": payload["metadata"]}
            )
            results.append((chunk, float(point.score)))
        return results


def build_vector_store(settings: Settings, embeddings: Embeddings | None = None) -> LegalVectorStore:
    return LegalVectorStore(
        client=build_qdrant_client(settings),
        embeddings=embeddings or build_embeddings(settings),
        collection=settings.qdrant_collection,
    )
