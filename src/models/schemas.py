"""Pydantic v2 domain models.

Three families of models live here:

1. **Legal corpus**: :class:`LegalChunk` and its metadata (source, article, risk category,
   date of application...).
2. **LLM structured outputs** (``*Draft``, :class:`QueryPlan`, :class:`RethinkerVerdict`...):
   deliberately flat and short schemas, which makes structured output from local models
   more reliable.
3. **Audit report** and **LangGraph state** (:class:`AuditState`): objects enriched
   deterministically by code (deadlines, score, citation validation).

All models forbid unknown fields (``extra="forbid"``) so typos and schema drift fail loudly.
"""

from __future__ import annotations

import datetime as _dt
import operator
import uuid
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    computed_field,
    field_validator,
    model_validator,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def normalise_confidence(value: Any) -> Any:
    """Map confidences given on a 0-10 or 0-100 scale, or as "85%", to [0, 1].

    Local models frequently ignore the requested scale; rejecting the whole answer for that
    would waste a full generation, so the value is normalised before validation.
    """
    if isinstance(value, str):
        try:
            value = float(value.strip().rstrip("%").strip())
        except ValueError:
            return value  # left to the regular validation error
    if isinstance(value, bool) or not isinstance(value, int | float):
        return value
    if value > 10:
        return min(value / 100, 1.0)
    if value > 1:
        return value / 10
    return max(float(value), 0.0)


Confidence = Annotated[
    float,
    BeforeValidator(normalise_confidence),
    Field(ge=0.0, le=1.0, description="Confidence as a decimal between 0 and 1, e.g. 0.75"),
]


class LenientStrEnum(StrEnum):
    """String enum accepting case/separator variants produced by LLMs ("High-risk" -> high_risk)."""

    @classmethod
    def _missing_(cls, value: object) -> LenientStrEnum | None:
        if isinstance(value, str):
            key = value.strip().lower().replace("-", "_").replace(" ", "_")
            for member in cls:
                if member.value == key:
                    return member
        return None


class _ComputedFieldsTolerant(BaseModel):
    """Ignore computed-field keys on input so that ``model_dump()`` output round-trips
    through validation even though unknown fields are forbidden."""

    @model_validator(mode="before")
    @classmethod
    def _drop_computed_fields(cls, data: Any) -> Any:
        if isinstance(data, dict) and cls.model_computed_fields:
            return {k: v for k, v in data.items() if k not in cls.model_computed_fields}
        return data


class StrictModel(_ComputedFieldsTolerant):
    """Base model: unknown fields are rejected, assignments are re-validated."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class FrozenModel(_ComputedFieldsTolerant):
    """Immutable, hashable value object."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# =============================================================================
# Enumerations
# =============================================================================


class Regulation(StrEnum):
    AI_ACT = "ai_act"
    GDPR = "gdpr"

    @property
    def label(self) -> str:
        return {"ai_act": "EU AI Act - Regulation (EU) 2024/1689", "gdpr": "GDPR - Regulation (EU) 2016/679"}[
            self.value
        ]

    @property
    def short(self) -> str:
        return {"ai_act": "AI Act", "gdpr": "GDPR"}[self.value]


class UnitType(StrEnum):
    ARTICLE = "article"
    ANNEX = "annex"


class RiskCategory(LenientStrEnum):
    """AI Act risk tiers (``not_applicable`` is used for GDPR provisions)."""

    PROHIBITED = "prohibited"
    HIGH_RISK = "high_risk"
    LIMITED_RISK = "limited_risk"
    MINIMAL_RISK = "minimal_risk"
    GPAI = "gpai"
    GPAI_SYSTEMIC_RISK = "gpai_systemic_risk"
    HORIZONTAL = "horizontal"
    NOT_APPLICABLE = "not_applicable"

    @property
    def label(self) -> str:
        return {
            "prohibited": "Prohibited practice",
            "high_risk": "High risk",
            "limited_risk": "Limited risk (transparency)",
            "minimal_risk": "Minimal risk",
            "gpai": "General-purpose AI model",
            "gpai_systemic_risk": "GPAI model with systemic risk",
            "horizontal": "Horizontal provision",
            "not_applicable": "Not applicable",
        }[self.value]


class OperatorRole(LenientStrEnum):
    """Role of the organisation within the meaning of Article 3 of the AI Act."""

    PROVIDER = "provider"
    DEPLOYER = "deployer"
    IMPORTER = "importer"
    DISTRIBUTOR = "distributor"
    AUTHORISED_REPRESENTATIVE = "authorised_representative"
    UNDETERMINED = "undetermined"


class ComplianceStatus(LenientStrEnum):
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    TO_VERIFY = "to_verify"
    NOT_APPLICABLE = "not_applicable"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").capitalize()


class Severity(LenientStrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Priority(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"

    @property
    def label(self) -> str:
        return {"P0": "P0 - Blocking", "P1": "P1 - High", "P2": "P2 - Medium", "P3": "P3 - Low"}[self.value]

    @property
    def rank(self) -> int:
        return int(self.value[1])


class Effort(LenientStrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LegalBasis(LenientStrEnum):
    """Lawful bases of GDPR Article 6(1)."""

    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"
    NOT_IDENTIFIED = "not_identified"


# =============================================================================
# Legal corpus
# =============================================================================


class LegalChunkMetadata(FrozenModel):
    """Metadata stored as Qdrant payload (strict filtering)."""

    source: Regulation
    regulation_title: NonEmptyStr
    celex: Annotated[str, StringConstraints(pattern=r"^\d{5}R\d{4}$")]
    language: Literal["en", "fr"]
    unit_type: UnitType
    article_number: Annotated[int, Field(ge=1)] | None = None
    annex_number: Annotated[str, StringConstraints(pattern=r"^[IVXLC]+$")] | None = None
    title: str
    chapter_number: str | None = None
    chapter_title: str | None = None
    section_number: str | None = None
    section_title: str | None = None
    risk_category: RiskCategory
    enforcement_date: _dt.date
    enforcement_note: str | None = None
    topics: tuple[str, ...] = ()
    part: int = Field(default=1, ge=1)
    total_parts: int = Field(default=1, ge=1)
    page_start: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_unit_numbering(self) -> LegalChunkMetadata:
        if self.unit_type is UnitType.ARTICLE and self.article_number is None:
            raise ValueError("article chunks require article_number")
        if self.unit_type is UnitType.ANNEX and self.annex_number is None:
            raise ValueError("annex chunks require annex_number")
        if self.part > self.total_parts:
            raise ValueError("part cannot exceed total_parts")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def citation(self) -> str:
        """Human-readable reference, e.g. ``AI Act, Art. 5`` or ``GDPR, Art. 35``."""
        prefix = self.source.short
        ref = (
            f"{prefix}, Annex {self.annex_number}"
            if self.unit_type is UnitType.ANNEX
            else f"{prefix}, Art. {self.article_number}"
        )
        if self.total_parts > 1:
            ref += f" (part {self.part}/{self.total_parts})"
        return ref

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unit_key(self) -> str:
        """Stable key of the legal unit, e.g. ``ai_act:article:5`` or ``ai_act:annex:III``."""
        number = self.annex_number if self.unit_type is UnitType.ANNEX else self.article_number
        return f"{self.source.value}:{self.unit_type.value}:{number}"


class LegalChunk(FrozenModel):
    """Fragment of legal text split along the official structure."""

    chunk_id: Annotated[str, StringConstraints(pattern=r"^(ai_act|gdpr):(article|annex):[0-9IVXLC]+:p\d+$")]
    text: NonEmptyStr
    metadata: LegalChunkMetadata

    @computed_field  # type: ignore[prop-decorator]
    @property
    def point_id(self) -> str:
        """Deterministic UUIDv5 used as the Qdrant point id (idempotent upserts)."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"eu-legal/{self.chunk_id}"))


class RetrievedChunk(StrictModel):
    """Search result, with traceability of the score at every stage."""

    chunk: LegalChunk
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    query: str = ""

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.rrf_score


class RetrievalFilter(StrictModel):
    """Metadata filter applied to both dense and sparse search."""

    sources: list[Regulation] | None = None
    risk_categories: list[RiskCategory] | None = None
    unit_types: list[UnitType] | None = None
    article_numbers: list[Annotated[int, Field(ge=1)]] | None = None
    topics: list[NonEmptyStr] | None = None


# =============================================================================
# Regulatory deadlines
# =============================================================================


class Deadline(FrozenModel):
    """Date of application of a set of provisions."""

    deadline_id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9\-]+$")]
    regulation: Regulation
    date: _dt.date
    title: NonEmptyStr
    description: NonEmptyStr
    legal_basis: NonEmptyStr
    applies_to: tuple[RiskCategory, ...] = ()
    amended_by: str | None = None

    def is_past(self, today: _dt.date | None = None) -> bool:
        return self.date <= (today or _dt.date.today())

    def days_remaining(self, today: _dt.date | None = None) -> int:
        return (self.date - (today or _dt.date.today())).days


# =============================================================================
# Ingestion of the project under audit
# =============================================================================


class ProjectSection(StrictModel):
    index: int = Field(ge=1)
    kind: Literal["page", "slide", "text"]
    title: str | None = None
    markdown: str
    notes: str | None = None
    table_count: int = Field(default=0, ge=0)


class ProjectDocument(StrictModel):
    file_name: NonEmptyStr
    file_type: Literal["pdf", "pptx", "text"]
    title: str | None = None
    sections: list[ProjectSection] = Field(min_length=1)
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def markdown(self) -> str:
        parts: list[str] = []
        for s in self.sections:
            header = f"## {s.kind.capitalize()} {s.index}" + (f" - {s.title}" if s.title else "")
            block = f"{header}\n\n{s.markdown.strip()}"
            if s.notes:
                block += f"\n\n> Notes: {s.notes.strip()}"
            parts.append(block)
        return "\n\n".join(parts)

    @property
    def char_count(self) -> int:
        return sum(len(s.markdown) + len(s.notes or "") for s in self.sections)


# =============================================================================
# LLM structured outputs (flat, short schemas)
# =============================================================================


class ProjectProfile(StrictModel):
    """Factual profile of the project extracted from the document (no legal qualification)."""

    project_name: NonEmptyStr = Field(description="Name of the project or AI system")
    purpose: NonEmptyStr = Field(description="Main purpose of the system, in one or two sentences")
    description: str = Field(description="Functional summary of the system (5 sentences max)")
    sector: str = Field(description="Business sector (HR, health, banking, education...)")
    ai_techniques: list[str] = Field(default_factory=list, description="AI techniques used")
    operator_role: OperatorRole = Field(
        default=OperatorRole.UNDETERMINED,
        description="Role of the organisation: provider (develops) or deployer (uses)...",
    )
    users: list[str] = Field(default_factory=list, description="Users of the system")
    affected_persons: list[str] = Field(
        default_factory=list, description="People affected by the system's outputs"
    )
    personal_data_categories: list[str] = Field(
        default_factory=list, description="Categories of personal data processed"
    )
    special_category_data: list[str] = Field(
        default_factory=list,
        description="Special categories (GDPR Art. 9): health, biometrics, opinions...",
    )
    data_sources: list[str] = Field(default_factory=list)
    automated_decisions: bool = Field(
        default=False, description="Automated decisions producing significant effects"
    )
    profiling: bool = False
    biometric_processing: bool = False
    interacts_with_natural_persons: bool = False
    generates_synthetic_content: bool = False
    uses_general_purpose_model: bool = Field(
        default=False, description="Relies on a general-purpose AI model (LLM...)"
    )
    international_transfers: bool = False
    retention_policy: str | None = None
    security_measures: list[str] = Field(default_factory=list)
    stated_legal_basis: str | None = Field(
        default=None, description="GDPR lawful basis stated in the document, if any"
    )
    dpia_mentioned: bool = False
    human_oversight_mentioned: bool = False
    deployment_regions: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(
        default_factory=list, description="Important information missing from the document"
    )


class SubQuery(StrictModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=5)] = Field(
        description="Targeted legal question"
    )
    target: Regulation = Field(description="Regulation to search")
    rationale: str = Field(default="", description="Why this question matters")

    @property
    def key(self) -> tuple[Regulation, str]:
        """Normalised identity used for de-duplication."""
        return self.target, self.question.casefold()


class QueryPlan(StrictModel):
    sub_queries: list[SubQuery] = Field(min_length=1)


class RethinkerVerdict(StrictModel):
    """Verdict of the critic node (Rethinker)."""

    sufficient: bool = Field(description="Is the evidence sufficient to reach a conclusion?")
    confidence: Confidence
    coverage_gaps: list[str] = Field(
        default_factory=list, description="Legal aspects not covered by the evidence"
    )
    follow_up_queries: list[SubQuery] = Field(
        default_factory=list, description="New, reformulated queries when insufficient"
    )
    reasoning: str = ""


class LegalCitation(StrictModel):
    regulation: Regulation
    reference: NonEmptyStr = Field(description="E.g. 'Article 6(2)', 'Annex III, point 4'")
    justification: str = Field(default="", description="Link between the provision and the project")
    verified: bool = Field(default=False, description="Set by code: provision present in the evidence")
    chunk_ids: list[str] = Field(default_factory=list)


class Obligation(StrictModel):
    reference: NonEmptyStr
    description: str
    status: ComplianceStatus = ComplianceStatus.TO_VERIFY
    gap: str | None = None


class AIActClassificationDraft(StrictModel):
    """Structured output of the AI Act classification agent."""

    is_ai_system: bool = Field(description="Does the system meet the Art. 3(1) definition?")
    risk_category: RiskCategory = Field(description="Main risk category")
    secondary_categories: list[RiskCategory] = Field(default_factory=list)
    operator_role: OperatorRole = OperatorRole.UNDETERMINED
    annex_iii_areas: list[str] = Field(default_factory=list, description="Annex III areas concerned, if any")
    annex_i_product: bool = Field(
        default=False, description="Safety component of a product covered by Annex I"
    )
    prohibited_practice_flags: list[str] = Field(default_factory=list)
    article_6_3_exception: bool = Field(
        default=False, description="Art. 6(3) derogation applicable (narrow procedural task...)"
    )
    justification: str
    obligations: list[Obligation] = Field(default_factory=list)
    citations: list[LegalCitation] = Field(default_factory=list)
    confidence: Confidence

    @field_validator("secondary_categories")
    @classmethod
    def _unique_categories(cls, value: list[RiskCategory]) -> list[RiskCategory]:
        return list(dict.fromkeys(value))


GDPRTopic = Literal[
    "legal_basis",
    "data_minimisation",
    "dpia",
    "data_subject_rights",
    "transparency",
    "automated_decision_making",
    "special_categories",
    "security",
    "storage_limitation",
    "international_transfers",
    "processors",
    "privacy_by_design",
    "other",
]


class GDPRFinding(StrictModel):
    topic: GDPRTopic
    status: ComplianceStatus
    severity: Severity
    finding: str = Field(description="Factual observation")
    recommendation: str
    citations: list[LegalCitation] = Field(default_factory=list)


class GDPRAuditDraft(StrictModel):
    """Structured output of the GDPR audit agent."""

    processes_personal_data: bool
    identified_legal_basis: LegalBasis
    legal_basis_assessment: str
    legal_basis_status: ComplianceStatus
    minimisation_status: ComplianceStatus
    minimisation_assessment: str
    dpia_required: bool
    dpia_triggers: list[str] = Field(default_factory=list)
    dpia_status: ComplianceStatus
    rights_status: ComplianceStatus
    rights_assessment: str
    findings: list[GDPRFinding] = Field(default_factory=list)
    confidence: Confidence


class RecommendationDraft(StrictModel):
    title: NonEmptyStr
    description: str
    regulation: Regulation
    priority: Priority
    effort: Effort = Effort.MEDIUM
    owner: str = Field(default="DPO / Compliance officer")
    references: list[str] = Field(default_factory=list, description="Provisions concerned")


class SynthesisDraft(StrictModel):
    """Structured output of the synthesis agent."""

    executive_summary: NonEmptyStr
    key_risks: list[str] = Field(default_factory=list)
    recommendations: list[RecommendationDraft] = Field(min_length=1)


# =============================================================================
# Assessment interview (the LLM asks, the project team answers)
# =============================================================================


class InterviewTopic(LenientStrEnum):
    PURPOSE = "purpose"
    ROLE = "role"
    USERS_AND_AFFECTED_PERSONS = "users_and_affected_persons"
    DECISIONS = "decisions"
    HUMAN_OVERSIGHT = "human_oversight"
    PERSONAL_DATA = "personal_data"
    SENSITIVE_DATA = "sensitive_data"
    RETENTION = "retention"
    TRANSFERS = "transfers"
    TRANSPARENCY = "transparency"
    MODEL = "model"
    SECURITY = "security"
    GOVERNANCE = "governance"


AnswerType = Literal["text", "yes_no", "choice"]


class QuestionDraft(StrictModel):
    """One question generated by the interviewer LLM."""

    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=8, max_length=400)]
    why: str = Field(default="", max_length=400, description="Why the answer matters legally")
    topic: InterviewTopic
    regulation: Regulation | None = None
    answer_type: AnswerType = "text"
    options: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]] = Field(
        default_factory=list, max_length=6, description="Choices when answer_type is 'choice'"
    )

    @model_validator(mode="after")
    def _choices_need_options(self) -> QuestionDraft:
        if self.answer_type == "choice" and len(self.options) < 2:
            raise ValueError("a 'choice' question needs at least two options")
        return self


class InterviewRoundDraft(StrictModel):
    """Structured output of the interviewer: the next questions, or 'ready'."""

    ready: bool = Field(description="True when the context is sufficient to run the audit")
    rationale: str = ""
    questions: list[QuestionDraft] = Field(default_factory=list, max_length=8)


class InterviewQuestion(QuestionDraft):
    question_id: Annotated[str, StringConstraints(pattern=r"^r\d+-q\d+$")]


class InterviewAnswer(StrictModel):
    question_id: Annotated[str, StringConstraints(pattern=r"^r\d+-q\d+$")]
    answer: Annotated[str, StringConstraints(strip_whitespace=True, max_length=4000)] = ""
    skipped: bool = False


class InterviewRound(StrictModel):
    number: int = Field(ge=1)
    questions: list[InterviewQuestion]
    answers: list[InterviewAnswer] = Field(default_factory=list)

    @property
    def answered(self) -> bool:
        return bool(self.answers)

    def transcript(self) -> str:
        by_id = {a.question_id: a for a in self.answers}
        lines = []
        for q in self.questions:
            a = by_id.get(q.question_id)
            reply = "(skipped)" if a is None or a.skipped or not a.answer else a.answer
            lines.append(f"Q: {q.question}\nA: {reply}")
        return "\n".join(lines)


# =============================================================================
# Audit report (objects enriched by code)
# =============================================================================


class AIActAssessment(AIActClassificationDraft):
    applicable_deadlines: list[Deadline] = Field(default_factory=list)


class GDPRAssessment(GDPRAuditDraft):
    pass


class Recommendation(RecommendationDraft):
    recommendation_id: Annotated[str, StringConstraints(pattern=r"^REC-\d{2,}$")]
    due_date: _dt.date | None = None


class ChecklistItem(StrictModel):
    item_id: NonEmptyStr
    label: NonEmptyStr
    regulation: Regulation
    reference: str
    priority: Priority
    due_date: _dt.date | None = None
    done: bool = False
    source: Literal["catalog", "recommendation"] = "catalog"


class EvidenceReference(StrictModel):
    chunk_id: str
    citation: str
    source: Regulation
    title: str
    score: float
    excerpt: str


class AuditReport(StrictModel):
    report_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    generated_at: _dt.datetime = Field(default_factory=lambda: _dt.datetime.now(_dt.UTC))
    file_name: NonEmptyStr
    document_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    profile: ProjectProfile
    ai_act: AIActAssessment
    gdpr: GDPRAssessment
    executive_summary: str
    key_risks: list[str] = Field(default_factory=list)
    compliance_score: int = Field(ge=0, le=100)
    recommendations: list[Recommendation]
    checklist: list[ChecklistItem]
    deadlines: list[Deadline]
    evidence: list[EvidenceReference]
    warnings: list[str] = Field(default_factory=list)
    rethink_iterations: int = Field(default=0, ge=0)
    models: dict[str, str] = Field(default_factory=dict)
    disclaimer: str = (
        "This report was generated automatically by a local AI system as a decision-support "
        "tool. It does not constitute legal advice and must be reviewed by a qualified lawyer "
        "or Data Protection Officer."
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def compliance_level(self) -> str:
        if self.compliance_score >= 80:
            return "High"
        if self.compliance_score >= 55:
            return "Intermediate"
        return "Low"


# =============================================================================
# LangGraph state
# =============================================================================


class AuditState(BaseModel):
    """Shared state of the audit graph.

    Fields annotated with a reducer (``operator.add``) are *cumulative*: each node returns
    only the new items, which enables the Rethinker's corrective loop and the parallel
    execution of the agents without overwriting.
    """

    model_config = ConfigDict(extra="forbid")

    # Input
    project_path: NonEmptyStr
    file_name: str = ""

    # Ingestion & analysis
    document: ProjectDocument | None = None
    profile: ProjectProfile | None = None
    interview_transcript: str = ""

    # RAG
    query_plan: list[SubQuery] = Field(default_factory=list)
    pending_queries: list[SubQuery] = Field(default_factory=list)
    evidence: Annotated[list[RetrievedChunk], operator.add] = Field(default_factory=list)
    verdicts: Annotated[list[RethinkerVerdict], operator.add] = Field(default_factory=list)
    iteration: int = Field(default=0, ge=0)

    # Agents
    ai_act: AIActAssessment | None = None
    gdpr: GDPRAssessment | None = None
    report: AuditReport | None = None

    # Observability
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)
    trace: Annotated[list[str], operator.add] = Field(default_factory=list)


class ChatTurn(StrictModel):
    role: Literal["user", "assistant"]
    content: NonEmptyStr
    citations: list[str] = Field(default_factory=list)
