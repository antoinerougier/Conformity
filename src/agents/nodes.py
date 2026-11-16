"""Nodes of the LangGraph audit graph.

Each node is a method of :class:`AuditNodes`, which receives its dependencies by injection
(LLM, retriever, citation validator). Nodes only return the state fields they change; the
cumulative lists (``evidence``, ``verdicts``, ``errors``, ``trace``) are merged by the
reducers declared on :class:`AuditState`.

Every LLM call has a **deterministic fallback**: a failure of the local model degrades the
report ("to verify" statuses, warnings) without aborting the audit.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict

from src.agents import prompts
from src.compliance.citations import CitationValidator
from src.compliance.obligations import baseline_checklist
from src.compliance.scoring import compliance_score, recommendation_due_date
from src.compliance.timeline import deadlines_for
from src.core.config import Settings
from src.core.llm import LLMService
from src.core.logging import get_logger
from src.ingestion.parser import condense_markdown, parse_document
from src.models.schemas import (
    AIActAssessment,
    AIActClassificationDraft,
    AuditReport,
    AuditState,
    ChecklistItem,
    ComplianceStatus,
    EvidenceReference,
    GDPRAssessment,
    GDPRAuditDraft,
    LegalBasis,
    Priority,
    ProjectProfile,
    QueryPlan,
    Recommendation,
    RecommendationDraft,
    Regulation,
    RethinkerVerdict,
    RetrievedChunk,
    RiskCategory,
    Severity,
    SubQuery,
    SynthesisDraft,
)
from src.rag.retriever import HybridRetriever, deduplicate, format_evidence, select_evidence

logger = get_logger(__name__)

Update = dict[str, Any]
_T = TypeVar("_T")

MAX_FOLLOW_UP_QUERIES = 4
MIN_PLANNED_QUERIES = 6
MAX_RECOMMENDATIONS = 12

NODE_LABELS: dict[str, str] = {
    "parse_document": "Reading the project document",
    "profile_project": "Building the project profile",
    "plan_queries": "Turning the project into legal questions",
    "retrieve": "Searching the official texts",
    "rethink": "Checking that the evidence is sufficient",
    "classify_ai_act": "Classifying the system under the AI Act",
    "audit_gdpr": "Auditing personal data under the GDPR",
    "synthesize": "Writing the action plan",
}


class AuditDependencies(BaseModel):
    """Services injected into the graph nodes (validated at construction)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    llm: LLMService
    retriever: HybridRetriever
    validator: CitationValidator
    settings: Settings
    today: date | None = None

    @property
    def current_date(self) -> date:
        return self.today or date.today()


def _dump(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=1)


def anchor_queries(profile: ProjectProfile) -> list[SubQuery]:
    """Deterministic anchor queries guaranteeing coverage of the key provisions.

    GDPR anchors are phrased in French because they are search strings run against the
    official French GDPR text (lexical BM25 recall).
    """
    sector = profile.sector or "general"
    anchors = [
        SubQuery(
            question="Definition of an AI system and scope of the regulation",
            target=Regulation.AI_ACT,
            rationale="System qualification (Art. 2-3)",
        ),
        SubQuery(
            question="Prohibited AI practices",
            target=Regulation.AI_ACT,
            rationale="Rule out prohibited practices (Art. 5)",
        ),
        SubQuery(
            question=f"Is an AI system used in the {sector} sector classified as high-risk under Annex III "
            "and Article 6 classification rules?",
            target=Regulation.AI_ACT,
            rationale="High-risk classification",
        ),
        SubQuery(
            question="Licéité du traitement et bases légales",
            target=Regulation.GDPR,
            rationale="Lawful basis (Art. 6)",
        ),
        SubQuery(
            question="Analyse d'impact relative à la protection des données obligatoire",
            target=Regulation.GDPR,
            rationale="DPIA (Art. 35)",
        ),
        SubQuery(
            question="Principes de minimisation des données et limitation de la conservation",
            target=Regulation.GDPR,
            rationale="Principles (Art. 5)",
        ),
    ]
    if profile.automated_decisions or profile.profiling:
        anchors.append(
            SubQuery(
                question="Décision individuelle automatisée y compris le profilage, "
                "droit à une intervention humaine",
                target=Regulation.GDPR,
                rationale="Art. 22",
            )
        )
    if profile.interacts_with_natural_persons or profile.generates_synthetic_content:
        anchors.append(
            SubQuery(
                question="Transparency obligations for AI systems interacting with natural persons or "
                "generating synthetic content",
                target=Regulation.AI_ACT,
                rationale="Art. 50",
            )
        )
    if profile.uses_general_purpose_model:
        anchors.append(
            SubQuery(
                question="Obligations for providers of general-purpose AI models",
                target=Regulation.AI_ACT,
                rationale="Chapter V",
            )
        )
    if profile.special_category_data or profile.biometric_processing:
        anchors.append(
            SubQuery(
                question="Traitement portant sur des catégories particulières de données "
                "à caractère personnel",
                target=Regulation.GDPR,
                rationale="Art. 9",
            )
        )
    return anchors


def merge_queries(primary: Sequence[SubQuery], extra: Sequence[SubQuery], limit: int) -> list[SubQuery]:
    """Concatenate query lists, removing case-insensitive duplicates, keeping order."""
    seen: set[tuple[Regulation, str]] = set()
    merged: list[SubQuery] = []
    for query in [*primary, *extra]:
        if query.key not in seen:
            seen.add(query.key)
            merged.append(query)
    return merged[:limit]


def project_block(state: AuditState) -> str:
    """Project profile plus, when available, the interview transcript."""
    profile = _require(state.profile, "profile")
    block = f"PROJECT PROFILE:\n{_dump(profile)}"
    if state.interview_transcript:
        block += f"\n\nINTERVIEW WITH THE PROJECT TEAM (authoritative facts):\n{state.interview_transcript}"
    return block


def _evidence_digest(evidence: Sequence[RetrievedChunk], limit: int = 40) -> str:
    lines = []
    for item in deduplicate(evidence)[:limit]:
        m = item.chunk.metadata
        body = item.chunk.text.split("\n\n", 1)[-1][:220].replace("\n", " ")
        lines.append(f"- [{m.citation}] {m.title} (score {item.final_score:.3f}): {body}...")
    return "\n".join(lines) or "(no evidence)"


def _fallback_ai_act(reason: str) -> AIActClassificationDraft:
    return AIActClassificationDraft(
        is_ai_system=True,
        risk_category=RiskCategory.MINIMAL_RISK,
        justification=f"Automatic classification unavailable ({reason}). "
        "Default category, to be confirmed manually.",
        confidence=0.0,
    )


def _fallback_gdpr(reason: str) -> GDPRAuditDraft:
    pending = ComplianceStatus.TO_VERIFY
    note = f"Automatic analysis unavailable ({reason}) - to be performed manually."
    return GDPRAuditDraft(
        processes_personal_data=True,
        identified_legal_basis=LegalBasis.NOT_IDENTIFIED,
        legal_basis_assessment=note,
        legal_basis_status=pending,
        minimisation_status=pending,
        minimisation_assessment=note,
        dpia_required=False,
        dpia_status=pending,
        rights_status=pending,
        rights_assessment=note,
        confidence=0.0,
    )


def _require(value: _T | None, name: str) -> _T:
    """Narrow an optional state field, failing loudly if a previous node did not run."""
    if value is None:
        raise RuntimeError(f"State field '{name}' must be set before this node runs")
    return value


class AuditNodes:
    """Implementation of the graph nodes."""

    def __init__(self, deps: AuditDependencies) -> None:
        self.deps = deps
        self.s = deps.settings

    # ------------------------------------------------------------------ ingestion
    def parse_document(self, state: AuditState) -> Update:
        if state.document is not None:  # document provided by the assessment session
            return {
                "file_name": state.file_name or state.document.file_name,
                "trace": [f"Using the assessment context: {len(state.document.sections)} sections"],
            }
        document = parse_document(Path(state.project_path), max_size_mb=self.s.max_upload_mb)
        return {
            "document": document,
            "file_name": document.file_name,
            "trace": [
                f"Document parsed: {len(document.sections)} sections, {document.char_count} characters"
            ],
        }

    def profile_project(self, state: AuditState) -> Update:
        document = _require(state.document, "document")
        if state.profile is not None:  # profile already built during the interview
            return {"trace": [f"Profile from the interview: {state.profile.project_name}"]}
        content = condense_markdown(document, self.s.project_context_chars)
        if state.interview_transcript:
            content += f"\n\nINTERVIEW WITH THE PROJECT TEAM:\n{state.interview_transcript}"
        try:
            profile = self.deps.llm.structured(
                prompts.PROFILE_SYSTEM, f"PROJECT DOCUMENT:\n\n{content}", ProjectProfile
            )
            return {"profile": profile, "trace": [f"Profile: {profile.project_name} - {profile.sector}"]}
        except Exception as exc:  # deterministic fallback, error reported in the report
            logger.exception("Profiling failed")
            profile = ProjectProfile(
                project_name=document.title or Path(document.file_name).stem,
                purpose="Undetermined (automatic extraction failed)",
                description=content[:500],
                sector="undetermined",
                unknowns=["Automatic profile extraction failed: manual review required"],
            )
            return {"profile": profile, "errors": [f"Profiling: {exc}"]}

    # ------------------------------------------------------------------ RAG
    def plan_queries(self, state: AuditState) -> Update:
        profile = _require(state.profile, "profile")
        anchors = anchor_queries(profile)
        system = prompts.PLANNER_SYSTEM.format(min_q=MIN_PLANNED_QUERIES, max_q=self.s.max_sub_queries)
        errors: list[str] = []
        try:
            generated = self.deps.llm.structured(system, project_block(state), QueryPlan).sub_queries
        except Exception as exc:  # anchors alone still give a usable plan
            logger.exception("Query planning failed")
            generated, errors = [], [f"Query rewriter: {exc}"]
        queries = merge_queries(generated, anchors, self.s.max_sub_queries + len(anchors))
        return {
            "query_plan": queries,
            "pending_queries": queries,
            "errors": errors,
            "trace": [f"{len(queries)} sub-questions ({len(generated)} generated, {len(anchors)} anchors)"],
        }

    def retrieve(self, state: AuditState) -> Update:
        results = self.deps.retriever.retrieve_many(state.pending_queries)
        unique = {r.chunk.metadata.unit_key for r in results}
        return {
            "evidence": results,
            "pending_queries": [],
            "trace": [
                f"Iteration {state.iteration}: {len(state.pending_queries)} queries -> "
                f"{len(results)} excerpts ({len(unique)} distinct articles/annexes)"
            ],
        }

    def rethink(self, state: AuditState) -> Update:
        coverage = {
            reg: sum(1 for e in state.evidence if e.chunk.metadata.source is reg) for reg in Regulation
        }
        asked = "\n".join(f"- ({q.target.value}) {q.question}" for q in state.query_plan)
        user = (
            f"{project_block(state)}\n\nSUB-QUESTIONS ALREADY ASKED:\n{asked}\n\n"
            f"RETRIEVED EVIDENCE:\n{_evidence_digest(state.evidence)}"
        )
        errors: list[str] = []
        try:
            verdict = self.deps.llm.structured(
                prompts.RETHINKER_SYSTEM.format(max_follow=MAX_FOLLOW_UP_QUERIES), user, RethinkerVerdict
            )
        except Exception as exc:  # on failure, proceed with the evidence gathered
            logger.exception("Rethinker failed")
            verdict = RethinkerVerdict(
                sufficient=True, confidence=0.0, reasoning=f"Critic unavailable: {exc}"
            )
            errors.append(f"Rethinker: {exc}")

        # Deterministic guardrail: a regulation without any evidence makes the file insufficient.
        missing = [reg for reg, n in coverage.items() if n == 0]
        if missing and verdict.sufficient:
            verdict = verdict.model_copy(
                update={
                    "sufficient": False,
                    "coverage_gaps": [*verdict.coverage_gaps, *(f"No {r.short} evidence" for r in missing)],
                }
            )

        follow_ups: list[SubQuery] = []
        if not verdict.sufficient and state.iteration < self.s.max_rethink_iterations:
            follow_ups = merge_queries(verdict.follow_up_queries, [], MAX_FOLLOW_UP_QUERIES)
            if not follow_ups:
                target = missing[0] if missing else Regulation.AI_ACT
                follow_ups = [
                    SubQuery(question=gap, target=target)
                    for gap in verdict.coverage_gaps[:MAX_FOLLOW_UP_QUERIES]
                    if len(gap.strip()) >= 5
                ]
            already = {q.key for q in state.query_plan}
            follow_ups = [q for q in follow_ups if q.key not in already]

        outcome = "sufficient" if verdict.sufficient else "insufficient"
        update: Update = {
            "verdicts": [verdict],
            "errors": errors,
            "trace": [
                f"Rethinker (iteration {state.iteration}): {outcome} (confidence {verdict.confidence:.2f})"
                + (f" -> {len(follow_ups)} reformulated queries" if follow_ups else "")
            ],
        }
        if follow_ups:
            update["pending_queries"] = follow_ups
            update["query_plan"] = [*state.query_plan, *follow_ups]
            update["iteration"] = state.iteration + 1
        return update

    # ------------------------------------------------------------------ agents
    def _context_block(self, state: AuditState, regulation: Regulation) -> tuple[str, list[RetrievedChunk]]:
        evidence = select_evidence(state.evidence, regulation, self.s.evidence_context_chars)
        return format_evidence(evidence), evidence

    def classify_ai_act(self, state: AuditState) -> Update:
        context, evidence = self._context_block(state, Regulation.AI_ACT)
        user = f"{project_block(state)}\n\nAI ACT EXCERPTS:\n{context}"
        errors: list[str] = []
        try:
            draft = self.deps.llm.structured(prompts.AI_ACT_SYSTEM, user, AIActClassificationDraft)
        except Exception as exc:  # deterministic fallback
            logger.exception("AI Act classification failed")
            draft, errors = _fallback_ai_act(str(exc)), [f"AI Act agent: {exc}"]

        citations, warnings = self.deps.validator.with_evidence(state.evidence).validate_citations(
            draft.citations
        )
        categories = [draft.risk_category, *draft.secondary_categories]
        assessment = AIActAssessment(
            **draft.model_dump(exclude={"citations"}),
            citations=citations,
            applicable_deadlines=deadlines_for(
                categories, annex_i_product=draft.annex_i_product, include_gdpr=False
            ),
        )
        return {
            "ai_act": assessment,
            "errors": errors + [f"AI Act - {w}" for w in warnings],
            "trace": [
                f"AI Act: {assessment.risk_category.label} (confidence {assessment.confidence:.2f}, "
                f"{len(evidence)} excerpts analysed)"
            ],
        }

    def audit_gdpr(self, state: AuditState) -> Update:
        context, _ = self._context_block(state, Regulation.GDPR)
        user = f"{project_block(state)}\n\nGDPR EXCERPTS (official French text):\n{context}"
        errors: list[str] = []
        try:
            draft = self.deps.llm.structured(prompts.GDPR_SYSTEM, user, GDPRAuditDraft)
        except Exception as exc:  # deterministic fallback
            logger.exception("GDPR audit failed")
            draft, errors = _fallback_gdpr(str(exc)), [f"GDPR agent: {exc}"]

        validator = self.deps.validator.with_evidence(state.evidence)
        findings = []
        warnings: list[str] = []
        for finding in draft.findings:
            citations, finding_warnings = validator.validate_citations(finding.citations)
            warnings.extend(finding_warnings)
            findings.append(finding.model_copy(update={"citations": citations}))
        assessment = GDPRAssessment(**draft.model_dump(exclude={"findings"}), findings=findings)
        dpia = "required" if assessment.dpia_required else "not required"
        return {
            "gdpr": assessment,
            "errors": errors + [f"GDPR - {w}" for w in warnings],
            "trace": [f"GDPR: {len(findings)} findings, DPIA {dpia}"],
        }

    # ------------------------------------------------------------------ synthesis
    def synthesize(self, state: AuditState) -> Update:
        profile = _require(state.profile, "profile")
        document = _require(state.document, "document")
        ai, gdpr = _require(state.ai_act, "ai_act"), _require(state.gdpr, "gdpr")
        deadlines = deadlines_for(
            [ai.risk_category, *ai.secondary_categories], annex_i_product=ai.annex_i_product
        )
        today = self.deps.current_date
        calendar = "\n".join(
            f"- {d.date.isoformat()}: {d.title} ({d.legal_basis})"
            + (" - ALREADY APPLICABLE" if d.is_past(today) else "")
            for d in deadlines
        )
        user = (
            f"TODAY: {today.isoformat()}\n\n{project_block(state)}\n\n"
            f"AI ACT CLASSIFICATION:\n{_dump(ai)}\n\nGDPR AUDIT:\n{_dump(gdpr)}\n\n"
            f"APPLICABLE REGULATORY CALENDAR:\n{calendar}"
        )
        errors: list[str] = []
        try:
            draft = self.deps.llm.structured(prompts.SYNTHESIS_SYSTEM, user, SynthesisDraft)
        except Exception as exc:  # deterministic fallback
            logger.exception("Synthesis failed")
            errors.append(f"Synthesis agent: {exc}")
            draft = fallback_synthesis(ai, gdpr)

        recommendations = finalize_recommendations(draft.recommendations, ai, today)
        report = AuditReport(
            file_name=state.file_name or document.file_name,
            document_sha256=document.sha256,
            profile=profile,
            ai_act=ai,
            gdpr=gdpr,
            executive_summary=draft.executive_summary,
            key_risks=draft.key_risks,
            compliance_score=compliance_score(ai, gdpr),
            recommendations=recommendations,
            checklist=build_checklist(ai, gdpr, profile, recommendations),
            deadlines=deadlines,
            evidence=[_evidence_reference(e) for e in deduplicate(state.evidence)],
            warnings=list(dict.fromkeys([*state.errors, *errors])),
            rethink_iterations=state.iteration,
            models={
                "llm": self.deps.llm.model_name,
                "embeddings": self.s.embedding_model,
                "reranker": self.deps.retriever.reranker.name,
            },
        )
        return {
            "report": report,
            "errors": errors,
            "trace": [f"Report generated - score {report.compliance_score}/100"],
        }


def _evidence_reference(item: RetrievedChunk) -> EvidenceReference:
    m = item.chunk.metadata
    return EvidenceReference(
        chunk_id=item.chunk.chunk_id,
        citation=m.citation,
        source=m.source,
        title=m.title,
        score=round(item.final_score, 4),
        excerpt=item.chunk.text.split("\n\n", 1)[-1][:400],
    )


def fallback_synthesis(ai: AIActAssessment, gdpr: GDPRAssessment) -> SynthesisDraft:
    """Rule-based synthesis used when the synthesis agent fails."""
    recs = [
        RecommendationDraft(
            title=f"Address: {f.finding[:80]}",
            description=f.recommendation,
            regulation=Regulation.GDPR,
            priority=Priority.P0 if f.severity is Severity.CRITICAL else Priority.P1,
            references=[c.reference for c in f.citations],
        )
        for f in gdpr.findings
        if f.status in {ComplianceStatus.NON_COMPLIANT, ComplianceStatus.PARTIALLY_COMPLIANT}
    ]
    recs += [
        RecommendationDraft(
            title=f"AI Act obligation: {o.reference}",
            description=o.gap or o.description,
            regulation=Regulation.AI_ACT,
            priority=Priority.P1,
            references=[o.reference],
        )
        for o in ai.obligations
        if o.status is not ComplianceStatus.COMPLIANT
    ]
    if not recs:
        recs = [
            RecommendationDraft(
                title="Have the audit reviewed by a lawyer / DPO",
                description="The automatic synthesis is unavailable: manual review of the report.",
                regulation=Regulation.GDPR,
                priority=Priority.P1,
            )
        ]
    return SynthesisDraft(
        executive_summary=(
            f"Estimated AI Act category: {ai.risk_category.label}. GDPR lawful basis: "
            f"{gdpr.identified_legal_basis.value} ({gdpr.legal_basis_status.label}). "
            f"DPIA required: {'yes' if gdpr.dpia_required else 'no'}. "
            "Summary generated without LLM (degraded mode)."
        ),
        recommendations=recs[:MAX_RECOMMENDATIONS],
    )


def finalize_recommendations(
    drafts: Sequence[RecommendationDraft], ai: AIActAssessment, today: date
) -> list[Recommendation]:
    """Sort by priority, assign stable ids and compute internal due dates."""
    ordered = sorted(drafts, key=lambda r: r.priority.rank)
    return [
        Recommendation(
            **d.model_dump(),
            recommendation_id=f"REC-{i:02d}",
            due_date=recommendation_due_date(
                d.regulation, d.priority, ai.risk_category, annex_i_product=ai.annex_i_product, today=today
            ),
        )
        for i, d in enumerate(ordered, start=1)
    ]


def build_checklist(
    ai: AIActAssessment, gdpr: GDPRAssessment, profile: ProjectProfile, recs: Sequence[Recommendation]
) -> list[ChecklistItem]:
    """Baseline catalogue items + one item per recommendation, sorted by priority and date."""
    items = baseline_checklist(ai, gdpr, profile)
    items += [
        ChecklistItem(
            item_id=r.recommendation_id,
            label=r.title,
            regulation=r.regulation,
            reference=", ".join(r.references) or "-",
            priority=r.priority,
            due_date=r.due_date,
            source="recommendation",
        )
        for r in recs
    ]
    return sorted(items, key=lambda it: (it.priority.rank, it.due_date or date.max))
