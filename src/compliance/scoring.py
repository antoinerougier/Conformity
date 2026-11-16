"""Deterministic compliance score (0-100) computed from the analysis statuses."""

from __future__ import annotations

from datetime import date, timedelta

from src.compliance.timeline import compliance_due_date
from src.core.types import validated
from src.models.schemas import (
    AIActClassificationDraft,
    ComplianceStatus,
    GDPRAuditDraft,
    Priority,
    Regulation,
    RiskCategory,
    Severity,
)

#: Points awarded per status (``None`` = excluded from the score).
STATUS_POINTS: dict[ComplianceStatus, float | None] = {
    ComplianceStatus.COMPLIANT: 1.0,
    ComplianceStatus.PARTIALLY_COMPLIANT: 0.5,
    ComplianceStatus.TO_VERIFY: 0.4,
    ComplianceStatus.NON_COMPLIANT: 0.0,
    ComplianceStatus.NOT_APPLICABLE: None,
}
SEVERITY_WEIGHT: dict[Severity, float] = {
    Severity.CRITICAL: 3.0,
    Severity.HIGH: 2.0,
    Severity.MEDIUM: 1.5,
    Severity.LOW: 1.0,
}
NEUTRAL_SCORE = 50.0
PROHIBITED_SCORE_CAP = 10.0


@validated
def compliance_score(ai: AIActClassificationDraft, gdpr: GDPRAuditDraft) -> int:
    """Weighted average of the statuses of AI Act obligations and GDPR assessments."""
    weighted: list[tuple[float, float]] = []  # (points, weight)

    for obligation in ai.obligations:
        points = STATUS_POINTS[obligation.status]
        if points is not None:
            weighted.append((points, 2.0))

    if gdpr.processes_personal_data:
        core = [
            (gdpr.legal_basis_status, 3.0),
            (gdpr.minimisation_status, 2.0),
            (gdpr.rights_status, 2.0),
        ]
        if gdpr.dpia_required:
            core.append((gdpr.dpia_status, 3.0))
        for status, weight in core:
            points = STATUS_POINTS[status]
            if points is not None:
                weighted.append((points, weight))
        for finding in gdpr.findings:
            points = STATUS_POINTS[finding.status]
            if points is not None:
                weighted.append((points, SEVERITY_WEIGHT[finding.severity]))

    score = (
        100.0 * sum(p * w for p, w in weighted) / sum(w for _, w in weighted) if weighted else NEUTRAL_SCORE
    )
    if ai.risk_category is RiskCategory.PROHIBITED:
        score = min(score, PROHIBITED_SCORE_CAP)
    return max(0, min(100, round(score)))


PRIORITY_WINDOW: dict[Priority, timedelta] = {
    Priority.P0: timedelta(days=30),
    Priority.P1: timedelta(days=90),
    Priority.P2: timedelta(days=180),
    Priority.P3: timedelta(days=365),
}


@validated
def recommendation_due_date(
    regulation: Regulation,
    priority: Priority,
    risk_category: RiskCategory,
    *,
    annex_i_product: bool = False,
    today: date | None = None,
) -> date:
    """Internal due date: a window based on priority, capped by the regulatory deadline when
    it lies in the future (otherwise the obligation is already enforceable: window only)."""
    today = today or date.today()
    target = today + PRIORITY_WINDOW[priority]
    if regulation is Regulation.GDPR:
        return target
    regulatory = compliance_due_date(risk_category, annex_i_product=annex_i_product)
    return min(target, regulatory) if regulatory > today else target
