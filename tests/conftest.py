"""Shared fixtures: real corpus, deterministic embeddings, in-memory Qdrant, fake LLM."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from qdrant_client import QdrantClient

from src.compliance.timeline import deadlines_for
from src.core.config import RerankerBackend, Settings
from src.models.schemas import (
    AIActAssessment,
    AuditReport,
    ChecklistItem,
    ComplianceStatus,
    EvidenceReference,
    GDPRAssessment,
    GDPRFinding,
    LegalBasis,
    LegalChunk,
    LegalCitation,
    Obligation,
    Priority,
    ProjectProfile,
    Recommendation,
    Regulation,
    RiskCategory,
    Severity,
)
from src.rag.preprocess_laws import read_jsonl
from src.rag.retriever import BM25Index, HybridRetriever, NoOpReranker, tokenize
from src.rag.vector_store import LegalVectorStore

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "processed" / "legal_chunks.jsonl"

T = TypeVar("T", bound=BaseModel)
Scripted = BaseModel | Exception | Callable[[str], BaseModel]


class HashingEmbeddings(Embeddings):
    """Deterministic embeddings (hashing trick over tokens): no service required."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeLLM:
    """Fake :class:`LLMService`: answers scripted per Pydantic schema.

    Items are consumed in order; the last item is "sticky" and reused for further calls.
    """

    model_name = "fake-llm"

    def __init__(self) -> None:
        self.responses: dict[type[BaseModel], list[Scripted]] = {}
        self.calls: list[tuple[str, str]] = []
        self.stream_text = "According to [AI Act, Art. 14], human oversight is required."

    def queue(self, schema: type[BaseModel], *items: Scripted) -> None:
        self.responses.setdefault(schema, []).extend(items)

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        self.calls.append((schema.__name__, user))
        queue = self.responses.get(schema)
        if not queue:
            raise AssertionError(f"No scripted answer for {schema.__name__}")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        result = item if isinstance(item, BaseModel) else item(user)
        assert isinstance(result, schema)
        return result

    def stream(self, messages: Sequence[BaseMessage]) -> Iterator[str]:
        self.calls.append(("stream", str(messages[-1].content)))
        for word in self.stream_text.split(" "):
            yield word + " "

    def called(self, schema_name: str) -> int:
        return sum(1 for name, _ in self.calls if name == schema_name)


@pytest.fixture(scope="session")
def corpus() -> list[LegalChunk]:
    assert CORPUS_PATH.is_file(), "Preprocessed corpus missing: python -m src.rag.preprocess_laws"
    return read_jsonl(CORPUS_PATH)


@pytest.fixture(scope="session")
def embeddings() -> HashingEmbeddings:
    return HashingEmbeddings()


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        qdrant_url=None,
        qdrant_path=None,
        qdrant_collection="test_corpus",
        reranker_backend=RerankerBackend.NONE,
        uploads_dir=tmp_path / "uploads",
        reports_dir=tmp_path / "reports",
        max_rethink_iterations=2,
    )


@pytest.fixture(scope="session")
def indexed_store(corpus: list[LegalChunk], embeddings: HashingEmbeddings) -> LegalVectorStore:
    store = LegalVectorStore(QdrantClient(location=":memory:"), embeddings, "test_corpus")
    store.index_chunks(corpus)
    return store


@pytest.fixture()
def retriever(
    indexed_store: LegalVectorStore, corpus: list[LegalChunk], settings: Settings
) -> HybridRetriever:
    return HybridRetriever(
        store=indexed_store, bm25=BM25Index(corpus), reranker=NoOpReranker(), settings=settings
    )


@pytest.fixture()
def fake_llm() -> FakeLLM:
    return FakeLLM()


def find_chunk(corpus: Sequence[LegalChunk], chunk_id: str) -> LegalChunk:
    return next(c for c in corpus if c.chunk_id == chunk_id)


def build_pptx(path: Path, slides: Sequence[dict[str, Any]]) -> Path:
    """Build a test deck: title, bullets (with levels), table, speaker notes."""
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    for spec in slides:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = spec["title"]
        body = slide.placeholders[1].text_frame
        bullets: list[tuple[int, str]] = spec.get("bullets", [])
        for i, (level, text) in enumerate(bullets):
            para = body.paragraphs[0] if i == 0 else body.add_paragraph()
            para.text = text
            para.level = level
        if table := spec.get("table"):
            shape = slide.shapes.add_table(
                len(table), len(table[0]), Inches(1), Inches(5), Inches(6), Inches(1.2)
            )
            for r, row in enumerate(table):
                for c, value in enumerate(row):
                    shape.table.cell(r, c).text = value
        if notes := spec.get("notes"):
            slide.notes_slide.notes_text_frame.text = notes
    prs.save(str(path))
    return path


def sample_report() -> AuditReport:
    """Realistic audit report used by the export and UI tests."""
    ai = AIActAssessment(
        is_ai_system=True,
        risk_category=RiskCategory.HIGH_RISK,
        annex_iii_areas=["Employment (point 4)"],
        justification="CV screening: Annex III, point 4(a) - 'recruitment'.",
        obligations=[Obligation(reference="Article 14", description="Human oversight", gap="Missing")],
        citations=[LegalCitation(regulation=Regulation.AI_ACT, reference="Annex III", verified=True)],
        confidence=0.9,
        applicable_deadlines=deadlines_for([RiskCategory.HIGH_RISK], include_gdpr=False),
    )
    gdpr = GDPRAssessment(
        processes_personal_data=True,
        identified_legal_basis=LegalBasis.LEGITIMATE_INTERESTS,
        legal_basis_assessment="Legitimate interest to be documented (balancing test).",
        legal_basis_status=ComplianceStatus.PARTIALLY_COMPLIANT,
        minimisation_status=ComplianceStatus.TO_VERIFY,
        minimisation_assessment="5-year retention period is excessive.",
        dpia_required=True,
        dpia_triggers=["Systematic evaluation (Art. 35(3)(a))"],
        dpia_status=ComplianceStatus.NON_COMPLIANT,
        rights_status=ComplianceStatus.TO_VERIFY,
        rights_assessment="Procedure not described.",
        findings=[
            GDPRFinding(
                topic="dpia",
                status=ComplianceStatus.NON_COMPLIANT,
                severity=Severity.CRITICAL,
                finding="No DPIA - “work in progress”…",
                recommendation="Carry out the DPIA before going live.",
                citations=[LegalCitation(regulation=Regulation.GDPR, reference="Article 35")],
            )
        ],
        confidence=0.8,
    )
    return AuditReport(
        file_name="hirematch.pptx",
        document_sha256="a" * 64,
        profile=ProjectProfile(
            project_name="HireMatch", purpose="CV screening", description="d", sector="HR"
        ),
        ai_act=ai,
        gdpr=gdpr,
        executive_summary="High-risk system ≥ heavy obligations.",
        key_risks=["No DPIA"],
        compliance_score=42,
        recommendations=[
            Recommendation(
                recommendation_id="REC-01",
                title="Carry out the DPIA",
                description="Before going live.",
                regulation=Regulation.GDPR,
                priority=Priority.P0,
                references=["Article 35"],
            )
        ],
        checklist=[
            ChecklistItem(
                item_id="gdpr-dpia",
                label="Carry out a DPIA",
                regulation=Regulation.GDPR,
                reference="Article 35",
                priority=Priority.P0,
            )
        ],
        deadlines=deadlines_for([RiskCategory.HIGH_RISK]),
        evidence=[
            EvidenceReference(
                chunk_id="gdpr:article:35:p1",
                citation="GDPR, Art. 35",
                source=Regulation.GDPR,
                title="DPIA",
                score=0.9,
                excerpt="Lorsqu'un type de traitement…",
            )
        ],
        warnings=["GDPR - Valid citation but missing from the retrieved evidence: GDPR - Article 35"],
    )
