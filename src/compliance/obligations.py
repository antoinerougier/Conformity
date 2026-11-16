"""Deterministic catalogue of baseline obligations (compliance checklist).

The LLM produces context-specific recommendations; this catalogue additionally guarantees
that the *structural* obligations of each category always appear in the checklist, with the
correct article reference and the correct deadline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from src.compliance.timeline import (
    AI_ACT_ART50_2_GRACE_END,
    AI_ACT_GENERAL_APPLICATION,
    AI_ACT_GPAI_GOVERNANCE,
    AI_ACT_PROHIBITIONS,
    GDPR_APPLICATION,
    compliance_due_date,
)
from src.core.types import validated
from src.models.schemas import (
    AIActClassificationDraft,
    ChecklistItem,
    GDPRAuditDraft,
    OperatorRole,
    Priority,
    ProjectProfile,
    Regulation,
    RiskCategory,
)


@dataclass(frozen=True, slots=True)
class _Context:
    """Read-only view over the validated analysis results used by catalogue conditions."""

    ai: AIActClassificationDraft
    gdpr: GDPRAuditDraft
    profile: ProjectProfile

    @property
    def categories(self) -> set[RiskCategory]:
        return {self.ai.risk_category, *self.ai.secondary_categories}

    @property
    def role(self) -> OperatorRole:
        role = self.ai.operator_role
        return self.profile.operator_role if role is OperatorRole.UNDETERMINED else role

    @property
    def is_provider(self) -> bool:
        return self.role in {OperatorRole.PROVIDER, OperatorRole.UNDETERMINED}

    @property
    def is_deployer(self) -> bool:
        return self.role in {OperatorRole.DEPLOYER, OperatorRole.UNDETERMINED}


Condition = Callable[[_Context], bool]
DueDate = Callable[[_Context], date]


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    item_id: str
    label: str
    regulation: Regulation
    reference: str
    priority: Priority
    condition: Condition
    due: DueDate


def _high_risk(c: _Context) -> bool:
    return RiskCategory.HIGH_RISK in c.categories


def _high_risk_due(c: _Context) -> date:
    return compliance_due_date(RiskCategory.HIGH_RISK, annex_i_product=c.ai.annex_i_product)


def _gpai(c: _Context) -> bool:
    return bool({RiskCategory.GPAI, RiskCategory.GPAI_SYSTEMIC_RISK} & c.categories)


def _processes_data(c: _Context) -> bool:
    return c.gdpr.processes_personal_data


def _const(d: date) -> DueDate:
    return lambda _c: d


CATALOG: tuple[CatalogEntry, ...] = (
    # ------------------------------------------------------------------ AI Act - horizontal
    CatalogEntry(
        "aia-literacy",
        "Train staff to a sufficient level of AI literacy",
        Regulation.AI_ACT,
        "Article 4",
        Priority.P1,
        lambda c: c.ai.is_ai_system,
        _const(AI_ACT_PROHIBITIONS),
    ),
    CatalogEntry(
        "aia-prohibited-stop",
        "Stop or redesign the prohibited practice identified",
        Regulation.AI_ACT,
        "Article 5",
        Priority.P0,
        lambda c: RiskCategory.PROHIBITED in c.categories,
        _const(AI_ACT_PROHIBITIONS),
    ),
    # ------------------------------------------------------------------ AI Act - high risk (provider)
    CatalogEntry(
        "aia-hr-risk-mgmt",
        "Establish a risk management system covering the whole lifecycle",
        Regulation.AI_ACT,
        "Article 9",
        Priority.P0,
        lambda c: _high_risk(c) and c.is_provider,
        _high_risk_due,
    ),
    CatalogEntry(
        "aia-hr-data-gov",
        "Govern training, validation and test data (quality, bias)",
        Regulation.AI_ACT,
        "Article 10",
        Priority.P0,
        lambda c: _high_risk(c) and c.is_provider,
        _high_risk_due,
    ),
    CatalogEntry(
        "aia-hr-tech-doc",
        "Draw up the technical documentation (Annex IV content)",
        Regulation.AI_ACT,
        "Article 11, Annex IV",
        Priority.P1,
        lambda c: _high_risk(c) and c.is_provider,
        _high_risk_due,
    ),
    CatalogEntry(
        "aia-hr-logging",
        "Ensure automatic recording of events (logs)",
        Regulation.AI_ACT,
        "Article 12",
        Priority.P1,
        lambda c: _high_risk(c) and c.is_provider,
        _high_risk_due,
    ),
    CatalogEntry(
        "aia-hr-transparency",
        "Provide instructions for use to deployers",
        Regulation.AI_ACT,
        "Article 13",
        Priority.P1,
        lambda c: _high_risk(c) and c.is_provider,
        _high_risk_due,
    ),
    CatalogEntry(
        "aia-hr-human-oversight",
        "Design effective human oversight measures",
        Regulation.AI_ACT,
        "Article 14",
        Priority.P0,
        _high_risk,
        _high_risk_due,
    ),
    CatalogEntry(
        "aia-hr-robustness",
        "Ensure accuracy, robustness and cybersecurity",
        Regulation.AI_ACT,
        "Article 15",
        Priority.P1,
        lambda c: _high_risk(c) and c.is_provider,
        _high_risk_due,
    ),
    CatalogEntry(
        "aia-hr-qms",
        "Put a quality management system in place",
        Regulation.AI_ACT,
        "Article 17",
        Priority.P1,
        lambda c: _high_risk(c) and c.is_provider,
        _high_risk_due,
    ),
    CatalogEntry(
        "aia-hr-conformity",
        "Carry out the conformity assessment, EU declaration and CE marking",
        Regulation.AI_ACT,
        "Articles 43, 47 and 48",
        Priority.P0,
        lambda c: _high_risk(c) and c.is_provider,
        _high_risk_due,
    ),
    CatalogEntry(
        "aia-hr-registration",
        "Register the system in the EU database",
        Regulation.AI_ACT,
        "Article 49, Annex VIII",
        Priority.P1,
        _high_risk,
        _high_risk_due,
    ),
    CatalogEntry(
        "aia-hr-pmm",
        "Post-market monitoring plan and serious incident reporting",
        Regulation.AI_ACT,
        "Articles 72 and 73",
        Priority.P2,
        lambda c: _high_risk(c) and c.is_provider,
        _high_risk_due,
    ),
    # ------------------------------------------------------------------ AI Act - high risk (deployer)
    CatalogEntry(
        "aia-hr-deployer",
        "Meet deployer obligations (intended use, monitoring, log retention)",
        Regulation.AI_ACT,
        "Article 26",
        Priority.P1,
        lambda c: _high_risk(c) and c.is_deployer,
        _high_risk_due,
    ),
    CatalogEntry(
        "aia-hr-fria",
        "Assess whether a fundamental rights impact assessment (FRIA) is required",
        Regulation.AI_ACT,
        "Article 27",
        Priority.P1,
        lambda c: _high_risk(c) and c.is_deployer,
        _high_risk_due,
    ),
    # ------------------------------------------------------------------ AI Act - transparency
    CatalogEntry(
        "aia-transp-interaction",
        "Inform people that they are interacting with an AI system",
        Regulation.AI_ACT,
        "Article 50(1)",
        Priority.P1,
        lambda c: c.profile.interacts_with_natural_persons,
        _const(AI_ACT_GENERAL_APPLICATION),
    ),
    CatalogEntry(
        "aia-transp-marking",
        "Mark generated content in a machine-readable format",
        Regulation.AI_ACT,
        "Article 50(2)",
        Priority.P1,
        lambda c: c.profile.generates_synthetic_content,
        _const(AI_ACT_ART50_2_GRACE_END),
    ),
    CatalogEntry(
        "aia-transp-emotion-biometric",
        "Inform people exposed to emotion recognition / biometric categorisation",
        Regulation.AI_ACT,
        "Article 50(3)",
        Priority.P1,
        lambda c: c.profile.biometric_processing,
        _const(AI_ACT_GENERAL_APPLICATION),
    ),
    # ------------------------------------------------------------------ AI Act - GPAI
    CatalogEntry(
        "aia-gpai-provider",
        "GPAI documentation, downstream information, copyright policy, training data summary",
        Regulation.AI_ACT,
        "Article 53, Annexes XI and XII",
        Priority.P1,
        lambda c: _gpai(c) and c.is_provider,
        _const(AI_ACT_GPAI_GOVERNANCE),
    ),
    CatalogEntry(
        "aia-gpai-systemic",
        "Model evaluations, adversarial testing, incident tracking and cybersecurity",
        Regulation.AI_ACT,
        "Article 55",
        Priority.P0,
        lambda c: RiskCategory.GPAI_SYSTEMIC_RISK in c.categories and c.is_provider,
        _const(AI_ACT_GPAI_GOVERNANCE),
    ),
    # ------------------------------------------------------------------ GDPR
    CatalogEntry(
        "gdpr-legal-basis",
        "Document the lawful basis of each processing operation",
        Regulation.GDPR,
        "Article 6",
        Priority.P0,
        _processes_data,
        _const(GDPR_APPLICATION),
    ),
    CatalogEntry(
        "gdpr-special-categories",
        "Justify an Article 9(2) exception for special categories of data",
        Regulation.GDPR,
        "Article 9",
        Priority.P0,
        lambda c: bool(c.profile.special_category_data) or c.profile.biometric_processing,
        _const(GDPR_APPLICATION),
    ),
    CatalogEntry(
        "gdpr-minimisation",
        "Limit data to what is necessary and set retention periods",
        Regulation.GDPR,
        "Article 5(1)(c) and (e)",
        Priority.P1,
        _processes_data,
        _const(GDPR_APPLICATION),
    ),
    CatalogEntry(
        "gdpr-information",
        "Inform data subjects (privacy notices)",
        Regulation.GDPR,
        "Articles 13 and 14",
        Priority.P1,
        _processes_data,
        _const(GDPR_APPLICATION),
    ),
    CatalogEntry(
        "gdpr-rights",
        "Set up a procedure for data subject rights (access, rectification, erasure, objection)",
        Regulation.GDPR,
        "Articles 15 to 21",
        Priority.P1,
        _processes_data,
        _const(GDPR_APPLICATION),
    ),
    CatalogEntry(
        "gdpr-art22",
        "Guarantee human intervention and the right to contest automated decisions",
        Regulation.GDPR,
        "Article 22",
        Priority.P0,
        lambda c: c.profile.automated_decisions,
        _const(GDPR_APPLICATION),
    ),
    CatalogEntry(
        "gdpr-privacy-by-design",
        "Apply data protection by design and by default",
        Regulation.GDPR,
        "Article 25",
        Priority.P2,
        _processes_data,
        _const(GDPR_APPLICATION),
    ),
    CatalogEntry(
        "gdpr-processors",
        "Put data processing agreements in place with processors (hosting, model vendor...)",
        Regulation.GDPR,
        "Article 28",
        Priority.P2,
        _processes_data,
        _const(GDPR_APPLICATION),
    ),
    CatalogEntry(
        "gdpr-register",
        "Add the processing to the records of processing activities",
        Regulation.GDPR,
        "Article 30",
        Priority.P2,
        _processes_data,
        _const(GDPR_APPLICATION),
    ),
    CatalogEntry(
        "gdpr-security",
        "Implement appropriate security measures (encryption, access control)",
        Regulation.GDPR,
        "Article 32",
        Priority.P1,
        _processes_data,
        _const(GDPR_APPLICATION),
    ),
    CatalogEntry(
        "gdpr-breach",
        "Personal data breach notification procedure (72 hours)",
        Regulation.GDPR,
        "Articles 33 and 34",
        Priority.P2,
        _processes_data,
        _const(GDPR_APPLICATION),
    ),
    CatalogEntry(
        "gdpr-dpia",
        "Carry out a data protection impact assessment (DPIA)",
        Regulation.GDPR,
        "Article 35",
        Priority.P0,
        lambda c: c.gdpr.dpia_required,
        _const(GDPR_APPLICATION),
    ),
    CatalogEntry(
        "gdpr-transfers",
        "Frame transfers outside the EU (adequacy decision, SCCs...)",
        Regulation.GDPR,
        "Articles 44 to 49",
        Priority.P1,
        lambda c: c.profile.international_transfers,
        _const(GDPR_APPLICATION),
    ),
)


@validated
def baseline_checklist(
    ai: AIActClassificationDraft, gdpr: GDPRAuditDraft, profile: ProjectProfile
) -> list[ChecklistItem]:
    """Build the baseline checklist applicable to the project."""
    ctx = _Context(ai=ai, gdpr=gdpr, profile=profile)
    return [
        ChecklistItem(
            item_id=entry.item_id,
            label=entry.label,
            regulation=entry.regulation,
            reference=entry.reference,
            priority=entry.priority,
            due_date=entry.due(ctx),
            source="catalog",
        )
        for entry in CATALOG
        if entry.condition(ctx)
    ]
