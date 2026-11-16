"""Integration tests of the full LangGraph graph (fake LLM, in-memory Qdrant, real corpus)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.agents.chat import ComplianceAssistant
from src.agents.graph import AGENT_NODES, ComplianceAuditor, ProgressEvent, route_after_rethink
from src.agents.nodes import AuditDependencies, anchor_queries
from src.compliance.citations import CitationValidator
from src.core.config import Settings
from src.core.llm import StructuredOutputError
from src.models.schemas import (
    AIActClassificationDraft,
    AuditReport,
    AuditState,
    ChatTurn,
    ComplianceStatus,
    GDPRAuditDraft,
    GDPRFinding,
    LegalBasis,
    LegalCitation,
    Obligation,
    OperatorRole,
    Priority,
    ProjectProfile,
    QueryPlan,
    RecommendationDraft,
    Regulation,
    RethinkerVerdict,
    RiskCategory,
    Severity,
    SubQuery,
    SynthesisDraft,
)
from src.rag.retriever import HybridRetriever
from tests.conftest import FakeLLM, build_pptx

pytestmark = pytest.mark.integration

TODAY = date(2026, 11, 16)

PROFILE = ProjectProfile(
    project_name="HireMatch",
    purpose="Automatically rank job applications",
    description="LLM-based CV scoring, shortlist sent to recruiters.",
    sector="human resources",
    ai_techniques=["LLM", "scoring"],
    operator_role=OperatorRole.PROVIDER,
    affected_persons=["candidates"],
    personal_data_categories=["identity", "work history"],
    automated_decisions=True,
    profiling=True,
    interacts_with_natural_persons=True,
)

AI_DRAFT = AIActClassificationDraft(
    is_ai_system=True,
    risk_category=RiskCategory.HIGH_RISK,
    secondary_categories=[RiskCategory.LIMITED_RISK],
    operator_role=OperatorRole.PROVIDER,
    annex_iii_areas=["4(a) recruitment and selection"],
    justification="CV screening falls under Annex III, point 4(a).",
    obligations=[
        Obligation(
            reference="Article 14", description="Human oversight", status=ComplianceStatus.NON_COMPLIANT
        ),
        Obligation(reference="Article 10", description="Data governance", status=ComplianceStatus.TO_VERIFY),
    ],
    citations=[
        LegalCitation(regulation=Regulation.AI_ACT, reference="Annex III, point 4"),
        LegalCitation(regulation=Regulation.AI_ACT, reference="Article 6(2)"),
        LegalCitation(regulation=Regulation.AI_ACT, reference="Article 187"),  # deliberate hallucination
    ],
    confidence=0.85,
)

GDPR_DRAFT = GDPRAuditDraft(
    processes_personal_data=True,
    identified_legal_basis=LegalBasis.LEGITIMATE_INTERESTS,
    legal_basis_assessment="Lawful basis not documented.",
    legal_basis_status=ComplianceStatus.PARTIALLY_COMPLIANT,
    minimisation_status=ComplianceStatus.TO_VERIFY,
    minimisation_assessment="Retention period not specified.",
    dpia_required=True,
    dpia_triggers=["Profiling with significant effects"],
    dpia_status=ComplianceStatus.NON_COMPLIANT,
    rights_status=ComplianceStatus.TO_VERIFY,
    rights_assessment="No right to human intervention.",
    findings=[
        GDPRFinding(
            topic="automated_decision_making",
            status=ComplianceStatus.NON_COMPLIANT,
            severity=Severity.CRITICAL,
            finding="Automatic rejections without human intervention.",
            recommendation="Introduce human review of rejections.",
            citations=[LegalCitation(regulation=Regulation.GDPR, reference="Article 22")],
        )
    ],
    confidence=0.8,
)

SYNTHESIS = SynthesisDraft(
    executive_summary="HireMatch is a high-risk system (Annex III).",
    key_risks=["Automated decisions"],
    recommendations=[
        RecommendationDraft(
            title="Set up human review",
            description="No rejection without human validation.",
            regulation=Regulation.AI_ACT,
            priority=Priority.P1,
            references=["Article 14"],
        ),
        RecommendationDraft(
            title="Carry out the DPIA",
            description="Before going live.",
            regulation=Regulation.GDPR,
            priority=Priority.P0,
            references=["Article 35"],
        ),
    ],
)


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    return build_pptx(
        tmp_path / "hirematch.pptx",
        [
            {"title": "HireMatch", "bullets": [(0, "Automatic CV screening by LLM"), (1, "Shortlist")]},
            {
                "title": "Data",
                "table": [["Data", "Retention"], ["CV", "Undefined"]],
                "notes": "No DPIA",
            },
        ],
    )


def _script(llm: FakeLLM, *, first_verdict_sufficient: bool = False) -> None:
    llm.queue(ProjectProfile, PROFILE)
    llm.queue(
        QueryPlan,
        QueryPlan(
            sub_queries=[
                SubQuery(
                    question="Is CV screening high-risk under Annex III employment?", target=Regulation.AI_ACT
                ),
                SubQuery(
                    question="Droit à une intervention humaine décision automatisée",  # French: GDPR corpus
                    target=Regulation.GDPR,
                ),
            ]
        ),
    )
    insufficient = RethinkerVerdict(
        sufficient=False,
        confidence=0.4,
        coverage_gaps=["Provider obligations"],
        follow_up_queries=[
            SubQuery(question="Obligations of providers of high-risk AI systems", target=Regulation.AI_ACT)
        ],
    )
    sufficient = RethinkerVerdict(sufficient=True, confidence=0.9)
    llm.queue(RethinkerVerdict, *([sufficient] if first_verdict_sufficient else [insufficient, sufficient]))
    llm.queue(AIActClassificationDraft, AI_DRAFT)
    llm.queue(GDPRAuditDraft, GDPR_DRAFT)
    llm.queue(SynthesisDraft, SYNTHESIS)


def _auditor(llm: FakeLLM, retriever: HybridRetriever, settings: Settings) -> ComplianceAuditor:
    deps = AuditDependencies(
        llm=llm,
        retriever=retriever,
        validator=CitationValidator.from_corpus(retriever.bm25.chunks),
        settings=settings,
        today=TODAY,
    )
    return ComplianceAuditor(deps)


def test_full_audit_with_rethinker_loop(
    fake_llm: FakeLLM, retriever: HybridRetriever, settings: Settings, project: Path
) -> None:
    _script(fake_llm)
    report = _auditor(fake_llm, retriever, settings).run(project)

    assert isinstance(report, AuditReport)
    assert report.file_name == "hirematch.pptx"
    assert report.rethink_iterations == 1  # one corrective loop, then a sufficient verdict
    assert fake_llm.called("RethinkerVerdict") == 2
    assert fake_llm.called("AIActClassificationDraft") == 1
    assert fake_llm.called("GDPRAuditDraft") == 1

    # Deterministic (Omnibus) deadlines, never generated by the LLM
    dates = {d.date for d in report.ai_act.applicable_deadlines}
    assert {date(2027, 12, 2), date(2026, 8, 2)} <= dates
    assert date(2028, 8, 2) not in dates  # no Annex I product

    # Anti-hallucination guardrail
    by_ref = {c.reference: c for c in report.ai_act.citations}
    assert by_ref["Article 187"].verified is False
    assert any("Article 187" in w and "not found in the corpus" in w for w in report.warnings)

    # Action plan and checklist
    assert [r.priority for r in report.recommendations] == [Priority.P0, Priority.P1]
    assert report.recommendations[0].due_date == date(2026, 12, 16)  # GDPR P0: +30 days
    assert report.recommendations[1].due_date == date(2027, 2, 14)  # AI Act P1: +90 days < 02/12/2027
    ids = {i.item_id for i in report.checklist}
    assert {"aia-hr-human-oversight", "gdpr-dpia", "gdpr-art22", "aia-transp-interaction", "REC-01"} <= ids
    assert 0 <= report.compliance_score < 60
    assert report.evidence
    assert {e.source for e in report.evidence} == set(Regulation)
    assert report.models["llm"] == "fake-llm"


def test_query_plan_contains_anchor_queries(
    fake_llm: FakeLLM, retriever: HybridRetriever, settings: Settings, project: Path
) -> None:
    _script(fake_llm, first_verdict_sufficient=True)
    auditor = _auditor(fake_llm, retriever, settings)
    report = auditor.run(project)
    assert report.rethink_iterations == 0
    anchors = anchor_queries(PROFILE)
    assert any(a.rationale == "Art. 22" for a in anchors)
    assert any(a.rationale == "Art. 50" for a in anchors)


def test_stream_emits_progress_then_report(
    fake_llm: FakeLLM, retriever: HybridRetriever, settings: Settings, project: Path
) -> None:
    _script(fake_llm)
    events = list(_auditor(fake_llm, retriever, settings).stream(project))
    nodes = [e.node for e in events if isinstance(e, ProgressEvent)]
    assert nodes[:3] == ["parse_document", "profile_project", "plan_queries"]
    assert nodes.count("retrieve") == 2
    assert nodes.count("rethink") == 2
    assert set(AGENT_NODES) <= set(nodes)
    assert nodes[-1] == "synthesize"
    assert isinstance(events[-1], AuditReport)


def test_rethinker_loop_is_bounded(
    fake_llm: FakeLLM, retriever: HybridRetriever, settings: Settings, project: Path
) -> None:
    _script(fake_llm)
    never_satisfied = [
        RethinkerVerdict(
            sufficient=False,
            confidence=0.2,
            follow_up_queries=[
                SubQuery(question=f"Reformulation {i} high-risk obligations", target=Regulation.AI_ACT)
            ],
        )
        for i in range(10)
    ]
    fake_llm.responses[RethinkerVerdict] = list(never_satisfied)
    report = _auditor(fake_llm, retriever, settings).run(project)
    assert report.rethink_iterations == settings.max_rethink_iterations == 2
    assert fake_llm.called("RethinkerVerdict") == 3


def test_degraded_mode_when_agents_fail(
    fake_llm: FakeLLM, retriever: HybridRetriever, settings: Settings, project: Path
) -> None:
    _script(fake_llm, first_verdict_sufficient=True)
    fake_llm.responses[AIActClassificationDraft] = [StructuredOutputError("invalid JSON")]
    fake_llm.responses[SynthesisDraft] = [StructuredOutputError("timeout")]
    report = _auditor(fake_llm, retriever, settings).run(project)
    assert report.ai_act.confidence == 0.0
    assert any("AI Act agent" in w for w in report.warnings)
    assert any("Synthesis agent" in w for w in report.warnings)
    assert "degraded mode" in report.executive_summary
    assert report.recommendations  # deterministic fallback from the GDPR findings


def test_route_after_rethink() -> None:
    state = AuditState(project_path="x.pdf")
    assert route_after_rethink(state) == AGENT_NODES
    state.pending_queries = [SubQuery(question="Annex III employment", target=Regulation.AI_ACT)]
    assert route_after_rethink(state) == "retrieve"


def test_mermaid_export(fake_llm: FakeLLM, retriever: HybridRetriever, settings: Settings) -> None:
    mermaid = _auditor(fake_llm, retriever, settings).mermaid()
    assert "rethink" in mermaid
    assert "audit_gdpr" in mermaid


def test_chat_assistant_answers_with_sources(
    fake_llm: FakeLLM, retriever: HybridRetriever, settings: Settings
) -> None:
    validator = CitationValidator.from_corpus(retriever.bm25.chunks)
    assistant = ComplianceAssistant(fake_llm, retriever, validator, settings)
    result = assistant.answer(
        "Which human oversight measures?", history=[ChatTurn(role="user", content="Hello")]
    )
    assert "[AI Act, Art. 14]" in result.answer
    assert result.unknown_references == []
    assert {c.chunk.metadata.source for c in result.contexts} == set(Regulation)
    fake_llm.stream_text = "See [GDPR, Art. 142]."
    assert assistant.answer("Question?").unknown_references == ["[GDPR, Art. 142]"]
