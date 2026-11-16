from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient

from src.models.schemas import (
    LegalChunk,
    Regulation,
    RetrievalFilter,
    RetrievedChunk,
    RiskCategory,
    SubQuery,
    UnitType,
)
from src.rag.retriever import (
    BM25Index,
    HybridRetriever,
    NoOpReranker,
    deduplicate,
    format_evidence,
    matches_filter,
    reciprocal_rank_fusion,
    select_evidence,
    tokenize,
)
from src.rag.vector_store import LegalVectorStore, build_filter
from tests.conftest import HashingEmbeddings, find_chunk


def test_tokenize_is_accent_insensitive_and_drops_stopwords() -> None:
    assert tokenize("L'Analyse d'impact relative à la protection des données") == [
        "analyse",
        "impact",
        "relative",
        "protection",
        "donnees",
    ]


def test_bm25_finds_breach_notification_article(corpus: list[LegalChunk]) -> None:
    # French query: the GDPR corpus is the official French text.
    hits = BM25Index(corpus).search(
        "notification violation données autorité de contrôle 72 heures",
        top_k=3,
        flt=RetrievalFilter(sources=[Regulation.GDPR]),
    )
    assert hits[0][0].chunk_id == "gdpr:article:33:p1"


def test_bm25_validates_inputs(corpus: list[LegalChunk]) -> None:
    index = BM25Index(corpus[:5])
    with pytest.raises(ValidationError):
        index.search("   ", top_k=3)
    with pytest.raises(ValidationError):
        index.search("risk", top_k=0)


def test_matches_filter(corpus: list[LegalChunk]) -> None:
    chunk = find_chunk(corpus, "ai_act:article:5:p1")
    assert matches_filter(chunk, RetrievalFilter(risk_categories=[RiskCategory.PROHIBITED]))
    assert not matches_filter(chunk, RetrievalFilter(sources=[Regulation.GDPR]))
    assert matches_filter(
        chunk, RetrievalFilter(topics=["prohibited_practices"], unit_types=[UnitType.ARTICLE])
    )
    assert not matches_filter(chunk, RetrievalFilter(article_numbers=[6]))


def test_rrf_rewards_consensus(corpus: list[LegalChunk]) -> None:
    a, b, c = corpus[:3]
    fused = reciprocal_rank_fusion([[(a, 0.9), (b, 0.8)], [(b, 12.0), (c, 5.0)]], k=60)
    assert fused[0][0].chunk_id == b.chunk_id
    assert fused[0][2] == {0: 0.8, 1: 12.0}


class TestVectorStore:
    def test_index_is_idempotent_and_filterable(self, corpus: list[LegalChunk]) -> None:
        store = LegalVectorStore(QdrantClient(location=":memory:"), HashingEmbeddings(), "t")
        subset = [c for c in corpus if c.metadata.article_number in {5, 6, 33, 35}]
        assert store.index_chunks(subset) == len(subset)
        store.index_chunks(subset)  # re-indexing: same ids
        assert store.count() == len(subset)
        hits = store.search("analyse d'impact", top_k=5, flt=RetrievalFilter(sources=[Regulation.GDPR]))
        assert hits
        assert all(c.metadata.source is Regulation.GDPR for c, _ in hits)
        assert store.health().reachable

    def test_recreate_drops_collection(self, corpus: list[LegalChunk]) -> None:
        store = LegalVectorStore(QdrantClient(location=":memory:"), HashingEmbeddings(), "t")
        store.index_chunks(corpus[:4])
        store.index_chunks(corpus[:2], recreate=True)
        assert store.count() == 2

    def test_build_filter(self) -> None:
        assert build_filter(None) is None
        assert build_filter(RetrievalFilter()) is None
        flt = build_filter(RetrievalFilter(sources=[Regulation.AI_ACT], article_numbers=[5]))
        assert flt is not None
        assert isinstance(flt.must, list)
        assert len(flt.must) == 2

    def test_rejects_invalid_arguments(self) -> None:
        with pytest.raises(ValueError, match="collection"):
            LegalVectorStore(QdrantClient(location=":memory:"), HashingEmbeddings(), "")
        store = LegalVectorStore(QdrantClient(location=":memory:"), HashingEmbeddings(), "t")
        with pytest.raises(ValidationError):
            store.search("", top_k=3)


class _ReverseReranker:
    name = "reverse"

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        return [c.model_copy(update={"rerank_score": float(i)}) for i, c in enumerate(reversed(candidates))][
            :top_n
        ]


class TestHybridRetriever:
    def test_retrieve_for_targets_regulation(self, retriever: HybridRetriever) -> None:
        results = retriever.retrieve_for(
            SubQuery(
                question="Prohibited AI practices emotion recognition workplace", target=Regulation.AI_ACT
            )
        )
        assert results
        assert all(r.chunk.metadata.source is Regulation.AI_ACT for r in results)
        assert "ai_act:article:5:p1" in [r.chunk.chunk_id for r in results]
        assert all(r.query for r in results)

    def test_reranker_is_applied(self, retriever: HybridRetriever) -> None:
        retriever.reranker = _ReverseReranker()
        results = retriever.retrieve("special categories of data", top_n=3)
        assert [r.rerank_score for r in results] == [0.0, 1.0, 2.0]

    def test_invalid_reranker_assignment_is_rejected(self, retriever: HybridRetriever) -> None:
        with pytest.raises(ValidationError):
            retriever.reranker = object()  # type: ignore[assignment]

    def test_retrieve_validates_query(self, retriever: HybridRetriever) -> None:
        with pytest.raises(ValidationError):
            retriever.retrieve("  ")
        with pytest.raises(ValidationError):
            retriever.retrieve("risk", top_n=0)

    def test_evidence_helpers(self, corpus: list[LegalChunk]) -> None:
        a = RetrievedChunk(chunk=find_chunk(corpus, "gdpr:article:35:p1"), rrf_score=0.1)
        a_better = a.model_copy(update={"rrf_score": 0.5})
        b = RetrievedChunk(chunk=find_chunk(corpus, "ai_act:article:5:p1"), rrf_score=0.3)
        assert [e.rrf_score for e in deduplicate([a, b, a_better])] == [0.5, 0.3]
        selected = select_evidence([a, b], Regulation.GDPR, max_chars=10)
        assert [e.chunk.chunk_id for e in selected] == ["gdpr:article:35:p1"]  # at least one item kept
        block = format_evidence([b])
        assert 'id="ai_act:article:5:p1"' in block
        assert 'applies_from="2025-02-02"' in block
        assert NoOpReranker().rerank("q", [a, b], 1) == [a]
