"""Contextual conversational assistant (page 2 chatbot) and legal QA.

The same pipeline (hybrid retrieval -> reranking -> grounded generation) serves:
- the Streamlit chatbot (:meth:`ComplianceAssistant.stream_answer`), contextualised by the
  audit report;
- the Ragas evaluation (:meth:`ComplianceAssistant.answer`), which measures the faithfulness
  and relevance of this pipeline.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from src.agents.prompts import CHAT_SYSTEM
from src.compliance.citations import CitationValidator
from src.core.config import Settings
from src.core.llm import LLMService
from src.core.types import NonEmptyStr, validated
from src.models.schemas import AuditReport, ChatTurn, Regulation, RetrievalFilter, RetrievedChunk
from src.rag.retriever import HybridRetriever, deduplicate, format_evidence

MAX_HISTORY_TURNS = 6


def report_context(report: AuditReport) -> str:
    """Compact summary of the report injected into the chatbot prompt."""
    ai, gdpr, p = report.ai_act, report.gdpr, report.profile
    recs = "\n".join(
        f"  - [{r.priority.value}] {r.title} ({', '.join(r.references)})" for r in report.recommendations
    )
    findings = "\n".join(
        f"  - {f.topic}: {f.status.label} ({f.severity.value}) - {f.finding}" for f in gdpr.findings
    )
    return (
        f"PROJECT: {p.project_name} - {p.purpose}\nSector: {p.sector}; role: {p.operator_role.value}\n"
        f"AI ACT: {ai.risk_category.label} (confidence {ai.confidence:.2f}) - {ai.justification}\n"
        f"GDPR: lawful basis {gdpr.identified_legal_basis.value} ({gdpr.legal_basis_status.label}); "
        f"DPIA required: {'yes' if gdpr.dpia_required else 'no'} ({gdpr.dpia_status.label})\n"
        f"GDPR findings:\n{findings or '  (none)'}\n"
        f"Compliance score: {report.compliance_score}/100\nAction plan:\n{recs}"
    )


class QAResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    contexts: list[RetrievedChunk]
    unknown_references: list[str] = Field(default_factory=list)

    @property
    def context_texts(self) -> list[str]:
        return [c.chunk.text for c in self.contexts]


class ComplianceAssistant:
    """Grounded question answering over the legal corpus, optionally aware of a report."""

    def __init__(
        self,
        llm: LLMService,
        retriever: HybridRetriever,
        validator: CitationValidator,
        settings: Settings,
        *,
        report: AuditReport | None = None,
        per_regulation_top_n: int = 4,
    ) -> None:
        if per_regulation_top_n < 1:
            raise ValueError("per_regulation_top_n must be >= 1")
        self.llm = llm
        self.retriever = retriever
        self.validator = validator
        self.settings = settings
        self.report = report
        self.top_n = per_regulation_top_n

    @validated
    def gather_context(
        self, question: NonEmptyStr, regulations: Sequence[Regulation] | None = None
    ) -> list[RetrievedChunk]:
        """Retrieve the best excerpts from each regulation for ``question``."""
        results: list[RetrievedChunk] = []
        for reg in regulations or list(Regulation):
            results.extend(
                self.retriever.retrieve(question, flt=RetrievalFilter(sources=[reg]), top_n=self.top_n)
            )
        return deduplicate(results)

    def build_messages(
        self, question: str, history: Sequence[ChatTurn], evidence: Sequence[RetrievedChunk]
    ) -> list[BaseMessage]:
        system = CHAT_SYSTEM
        if self.report is not None:
            system += f"\n\nPROJECT AUDIT REPORT:\n{report_context(self.report)}"
        messages: list[BaseMessage] = [SystemMessage(system)]
        for turn in list(history)[-2 * MAX_HISTORY_TURNS :]:
            messages.append(HumanMessage(turn.content) if turn.role == "user" else AIMessage(turn.content))
        messages.append(
            HumanMessage(f"RELEVANT LEGAL EXCERPTS:\n{format_evidence(evidence)}\n\nQUESTION: {question}")
        )
        return messages

    @validated
    def stream_answer(
        self, question: NonEmptyStr, history: Sequence[ChatTurn] = ()
    ) -> tuple[Iterator[str], list[RetrievedChunk]]:
        """Return a token stream and the evidence used (for source display)."""
        evidence = self.gather_context(question)
        return self.llm.stream(self.build_messages(question, history, evidence)), evidence

    @validated
    def answer(
        self,
        question: NonEmptyStr,
        history: Sequence[ChatTurn] = (),
        regulations: Sequence[Regulation] | None = None,
    ) -> QAResult:
        """Non-streaming answer, with detection of non-existent cited provisions."""
        evidence = self.gather_context(question, regulations)
        text = "".join(self.llm.stream(self.build_messages(question, history, evidence)))
        return QAResult(answer=text, contexts=evidence, unknown_references=self.validator.check_text(text))
