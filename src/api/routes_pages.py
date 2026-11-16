"""Server-rendered HTML pages (Jinja2)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from src.api.services import AppServices, get_services
from src.compliance.timeline import AI_ACT_ENTRY_INTO_FORCE, all_deadlines
from src.models.schemas import ComplianceStatus, Priority, Regulation, RiskCategory

TEMPLATES_DIR = Path(__file__).parent / "templates"

router = APIRouter(include_in_schema=False)
Services = Annotated[AppServices, Depends(get_services)]

# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

RISK_TONE: dict[RiskCategory, str] = {
    RiskCategory.PROHIBITED: "critical",
    RiskCategory.HIGH_RISK: "high",
    RiskCategory.GPAI_SYSTEMIC_RISK: "high",
    RiskCategory.LIMITED_RISK: "medium",
    RiskCategory.GPAI: "medium",
    RiskCategory.MINIMAL_RISK: "low",
    RiskCategory.HORIZONTAL: "low",
    RiskCategory.NOT_APPLICABLE: "low",
}
STATUS_TONE: dict[ComplianceStatus, str] = {
    ComplianceStatus.COMPLIANT: "low",
    ComplianceStatus.PARTIALLY_COMPLIANT: "medium",
    ComplianceStatus.NON_COMPLIANT: "critical",
    ComplianceStatus.TO_VERIFY: "neutral",
    ComplianceStatus.NOT_APPLICABLE: "neutral",
}
PRIORITY_TONE: dict[Priority, str] = {
    Priority.P0: "critical",
    Priority.P1: "high",
    Priority.P2: "medium",
    Priority.P3: "low",
}


def fmt_date(value: date | None) -> str:
    return value.strftime("%d %b %Y") if value else "-"


def timeline_position(value: date, start: date, end: date) -> float:
    """Horizontal position (0-100 %) of a date on a timeline."""
    span = (end - start).days or 1
    return max(0.0, min(100.0, 100.0 * (value - start).days / span))


def build_templates() -> Jinja2Templates:
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    env = templates.env
    env.filters["fdate"] = fmt_date
    env.globals.update(
        risk_tone=RISK_TONE,
        status_tone=STATUS_TONE,
        priority_tone=PRIORITY_TONE,
        Regulation=Regulation,
        timeline_position=timeline_position,
    )
    return templates


templates = build_templates()


def _timeline_context(today: date) -> dict[str, object]:
    deadlines = all_deadlines()
    start = AI_ACT_ENTRY_INTO_FORCE.replace(month=1, day=1)
    end = max(d.date for d in deadlines).replace(month=12, day=31)
    return {
        "deadlines": deadlines,
        "timeline_deadlines": [d for d in deadlines if d.date >= start],
        "timeline_start": start,
        "timeline_end": end,
        "year_marks": [
            (year, round(timeline_position(date(year, 1, 1), start, end), 2))
            for year in range(start.year, end.year + 1)
        ],
        "today": today,
    }


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def overview(request: Request, services: Services) -> Response:
    context = {"page": "overview", "settings": services.settings, **_timeline_context(date.today())}
    return templates.TemplateResponse(request, "overview.html", context)


@router.get("/assessment", response_class=HTMLResponse)
def assessment_start(request: Request, services: Services) -> Response:
    context = {"page": "assessment", "session": None, "settings": services.settings}
    return templates.TemplateResponse(request, "assessment.html", context)


@router.get("/assessment/{session_id}", response_class=HTMLResponse)
def assessment(request: Request, session_id: str, services: Services) -> Response:
    session = services.sessions.get(session_id)
    if session.status in {"auditing", "done", "failed"}:
        return RedirectResponse(request.url_for("report", session_id=session_id), status_code=303)
    context = {"page": "assessment", "session": session, "settings": services.settings}
    return templates.TemplateResponse(request, "assessment.html", context)


@router.get("/report/{session_id}", response_class=HTMLResponse, name="report")
def report(request: Request, session_id: str, services: Services) -> Response:
    session = services.sessions.get(session_id)
    if session.status in {"interviewing", "ready"}:
        return RedirectResponse(request.url_for("assessment", session_id=session_id), status_code=303)
    context = {
        "page": "report",
        "session": session,
        "report": session.report,
        "checklist_state": session.checklist_state,
        "checklist_done": sum(
            1
            for item in (session.report.checklist if session.report else [])
            if session.checklist_state.get(item.item_id)
        ),
        **_timeline_context(session.report.generated_at.date() if session.report else date.today()),
    }
    return templates.TemplateResponse(request, "report.html", context)
