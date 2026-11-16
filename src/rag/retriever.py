"""Hybrid search (dense Qdrant + BM25) fused with RRF, then cross-encoder reranking.

Pipeline for one query:
    1. Dense : Qdrant top-k (multilingual bge-m3 embeddings) with metadata filter.
    2. Sparse: in-memory BM25 top-k (same filter applied in Python).
    3. Fusion: Reciprocal Rank Fusion (k=60), robust to heterogeneous score scales.
    4. Rerank: FlashRank (ONNX, CPU) or the ``BAAI/bge-reranker-v2-m3`` cross-encoder.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict
from rank_bm25 import BM25Okapi

from src.core.config import RerankerBackend, Settings
from src.core.logging import get_logger
from src.core.types import NonEmptyStr, PositiveInt, validated
from src.models.schemas import (
    LegalChunk,
    Regulation,
    RetrievalFilter,
    RetrievedChunk,
    SubQuery,
)
from src.rag.vector_store import LegalVectorStore

logger = get_logger(__name__)

# =============================================================================
# Bilingual BM25 (the AI Act corpus is in English, the GDPR corpus in French)
# =============================================================================

# fmt: off
_STOPWORDS_EN = (
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it",
    "its", "of", "on", "or", "that", "the", "this", "to", "was", "were", "which", "with", "shall",
    "such", "any", "other", "may", "their", "those", "these", "where", "whether", "under", "into", "not",
)
_STOPWORDS_FR = (
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "l", "et", "ou", "en", "au", "aux", "par",
    "pour", "sur", "dans", "est", "sont", "qui", "que", "quoi", "ce", "cette", "ces", "son", "sa", "ses",
    "leur", "leurs", "il", "elle", "ils", "elles", "ne", "pas", "plus", "se", "s", "y", "ainsi", "lorsque",
)
# fmt: on
_STOPWORDS = frozenset(_STOPWORDS_EN + _STOPWORDS_FR)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Accent-insensitive normalisation and removal of EN/FR stop words."""
    normalized = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(c for c in normalized if not unicodedata.combining(c))
    return [t for t in _TOKEN_RE.findall(ascii_text) if len(t) > 1 and t not in _STOPWORDS]


def matches_filter(chunk: LegalChunk, flt: RetrievalFilter | None) -> bool:
    """Python-side equivalent of the Qdrant filter (used for BM25)."""
    if flt is None:
        return True
    m = chunk.metadata
    if flt.sources and m.source not in flt.sources:
        return False
    if flt.risk_categories and m.risk_category not in flt.risk_categories:
        return False
    if flt.unit_types and m.unit_type not in flt.unit_types:
        return False
    if flt.article_numbers and m.article_number not in flt.article_numbers:
        return False
    return not (flt.topics and not set(flt.topics) & set(m.topics))


class BM25Index:
    """In-memory lexical index (the corpus is a few hundred chunks)."""

    def __init__(self, chunks: Sequence[LegalChunk]) -> None:
        self.chunks = list(chunks)
        corpus = [tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    @validated
    def search(
        self, query: NonEmptyStr, *, top_k: PositiveInt, flt: RetrievalFilter | None = None
    ) -> list[tuple[LegalChunk, float]]:
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            (
                (c, float(s))
                for c, s in zip(self.chunks, scores, strict=True)
                if s > 0 and matches_filter(c, flt)
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:top_k]


# =============================================================================
# Rerankers
# =============================================================================


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]: ...


class NoOpReranker:
    """Keeps the RRF order (useful in tests or on very constrained machines)."""

    name = "none"

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        return list(candidates)[:top_n]


class FlashRankReranker:
    """Lightweight ONNX cross-encoder (CPU). ``ms-marco-MultiBERT-L-12`` is multilingual."""

    def __init__(self, model_name: str, cache_dir: str, max_length: int = 512) -> None:
        from flashrank import Ranker  # lazy import: the model is downloaded on first use

        self.name = f"flashrank:{model_name}"
        self._ranker = Ranker(model_name=model_name, cache_dir=cache_dir, max_length=max_length)

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        from flashrank import RerankRequest

        if not candidates:
            return []
        by_id = {c.chunk.chunk_id: c for c in candidates}
        passages: list[dict[str, Any]] = [{"id": c.chunk.chunk_id, "text": c.chunk.text} for c in candidates]
        ranked = self._ranker.rerank(RerankRequest(query=query, passages=passages))
        return [
            by_id[str(item["id"])].model_copy(update={"rerank_score": float(item["score"])})
            for item in ranked[:top_n]
        ]


class CrossEncoderReranker:
    """``BAAI/bge-reranker-v2-m3`` through sentence-transformers (``bge-reranker`` extra)."""

    def __init__(self, model_name: str, max_length: int = 1024) -> None:
        from sentence_transformers import CrossEncoder

        self.name = f"cross-encoder:{model_name}"
        self._model = CrossEncoder(model_name, max_length=max_length)

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        scores = self._model.predict([(query, c.chunk.text) for c in candidates])
        scored = [
            c.model_copy(update={"rerank_score": float(s)}) for c, s in zip(candidates, scores, strict=True)
        ]
        scored.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
        return scored[:top_n]


def build_reranker(settings: Settings) -> Reranker:
    """Instantiate the configured reranker, degrading gracefully to :class:`NoOpReranker`."""
    backend = settings.reranker_backend
    try:
        if backend is RerankerBackend.FLASHRANK:
            settings.reranker_cache_dir.mkdir(parents=True, exist_ok=True)
            return FlashRankReranker(settings.reranker_model, str(settings.reranker_cache_dir))
        if backend is RerankerBackend.CROSS_ENCODER:
            return CrossEncoderReranker(settings.cross_encoder_model)
    except Exception as exc:  # noqa: BLE001 - controlled degradation, logged
        logger.warning("Reranker '%s' unavailable (%s): falling back to RRF order.", backend.value, exc)
    return NoOpReranker()


# =============================================================================
# Hybrid retriever
# =============================================================================


@validated
def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[tuple[LegalChunk, float]]], k: PositiveInt = 60
) -> list[tuple[LegalChunk, float, dict[int, float]]]:
    """RRF: score(d) = sum(1 / (k + rank)). Also returns the raw score from each list."""
    fused: dict[str, float] = {}
    chunks: dict[str, LegalChunk] = {}
    raw: dict[str, dict[int, float]] = {}
    for list_index, ranking in enumerate(rankings):
        for rank, (chunk, score) in enumerate(ranking, start=1):
            cid = chunk.chunk_id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
            chunks[cid] = chunk
            raw.setdefault(cid, {})[list_index] = score
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [(chunks[cid], score, raw[cid]) for cid, score in ordered]


class HybridRetriever(BaseModel):
    """Dense + sparse retrieval, RRF fusion and reranking."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    store: LegalVectorStore
    bm25: BM25Index
    reranker: Reranker
    settings: Settings

    @validated
    def retrieve(
        self, query: NonEmptyStr, *, flt: RetrievalFilter | None = None, top_n: PositiveInt | None = None
    ) -> list[RetrievedChunk]:
        s = self.settings
        dense = self.store.search(query, top_k=s.dense_top_k, flt=flt)
        sparse = self.bm25.search(query, top_k=s.sparse_top_k, flt=flt)
        fused = reciprocal_rank_fusion([dense, sparse], k=s.rrf_k)[: s.rerank_candidates]
        candidates = [
            RetrievedChunk(
                chunk=chunk, dense_score=raw.get(0), sparse_score=raw.get(1), rrf_score=score, query=query
            )
            for chunk, score, raw in fused
        ]
        return self.reranker.rerank(query, candidates, top_n or s.rerank_top_n)

    @validated
    def retrieve_for(self, sub_query: SubQuery, *, top_n: PositiveInt | None = None) -> list[RetrievedChunk]:
        """Search restricted to the regulation targeted by the sub-question."""
        return self.retrieve(sub_query.question, flt=RetrievalFilter(sources=[sub_query.target]), top_n=top_n)

    @validated
    def retrieve_many(
        self, sub_queries: Sequence[SubQuery], *, top_n: PositiveInt | None = None
    ) -> list[RetrievedChunk]:
        return [hit for sq in sub_queries for hit in self.retrieve_for(sq, top_n=top_n)]


def deduplicate(evidence: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep, for each chunk, the occurrence with the best final score."""
    best: dict[str, RetrievedChunk] = {}
    for item in evidence:
        current = best.get(item.chunk.chunk_id)
        if current is None or item.final_score > current.final_score:
            best[item.chunk.chunk_id] = item
    return sorted(best.values(), key=lambda r: r.final_score, reverse=True)


@validated
def select_evidence(
    evidence: Sequence[RetrievedChunk], source: Regulation, max_chars: PositiveInt
) -> list[RetrievedChunk]:
    """Select the best evidence of one regulation within a character budget
    (at least one item is always kept so that agents never work without context)."""
    selected: list[RetrievedChunk] = []
    used = 0
    for item in deduplicate([e for e in evidence if e.chunk.metadata.source is source]):
        size = len(item.chunk.text)
        if selected and used + size > max_chars:
            continue
        selected.append(item)
        used += size
    return selected


def format_evidence(evidence: Sequence[RetrievedChunk]) -> str:
    """Context block injected into prompts, with citable identifiers."""
    blocks = []
    for item in evidence:
        m = item.chunk.metadata
        blocks.append(
            f'<source id="{item.chunk.chunk_id}" ref="{m.citation}" '
            f'applies_from="{m.enforcement_date.isoformat()}">\n{item.chunk.text}\n</source>'
        )
    return "\n\n".join(blocks)


def build_retriever(
    settings: Settings,
    *,
    store: LegalVectorStore | None = None,
    chunks: Sequence[LegalChunk] | None = None,
    reranker: Reranker | None = None,
) -> HybridRetriever:
    from src.rag.preprocess_laws import load_or_build_corpus
    from src.rag.vector_store import build_vector_store

    corpus = list(chunks) if chunks is not None else load_or_build_corpus(settings)
    return HybridRetriever(
        store=store or build_vector_store(settings),
        bm25=BM25Index(corpus),
        reranker=reranker or build_reranker(settings),
        settings=settings,
    )
