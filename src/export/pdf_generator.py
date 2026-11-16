"""Audit report export to PDF (ReportLab, no system dependency) and JSON."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, StyleSheet1, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable

from src.core.types import validated
from src.models.schemas import AuditReport, ComplianceStatus, Priority

PRIMARY = colors.HexColor("#1f4e79")
LIGHT = colors.HexColor("#eef3f8")
STATUS_COLORS = {
    ComplianceStatus.COMPLIANT: colors.HexColor("#2e7d32"),
    ComplianceStatus.PARTIALLY_COMPLIANT: colors.HexColor("#ef6c00"),
    ComplianceStatus.NON_COMPLIANT: colors.HexColor("#c62828"),
    ComplianceStatus.TO_VERIFY: colors.HexColor("#6d4c41"),
    ComplianceStatus.NOT_APPLICABLE: colors.HexColor("#757575"),
}
PRIORITY_COLORS = {
    Priority.P0: colors.HexColor("#c62828"),
    Priority.P1: colors.HexColor("#ef6c00"),
    Priority.P2: colors.HexColor("#f9a825"),
    Priority.P3: colors.HexColor("#2e7d32"),
}
MAX_EVIDENCE_ROWS = 40
EXCERPT_CHARS = 300
FOOTER = "EU AI Compliance Auditor - generated locally - not legal advice"

# Standard PDF fonts (Helvetica) cover Latin-1: the few characters outside it (bullets,
# typographic quotes...) are replaced to avoid rendering black boxes.
_REPLACEMENTS = {
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "…": "...",
    "•": "-",
    "–": "-",
    "—": "-",
    "→": "->",
    "≥": ">=",
    "≤": "<=",
    "œ": "oe",
    "Œ": "OE",
    " ": " ",
    " ": " ",
}

Row = list[str | Flowable]


def _clean(text: str | None) -> str:
    """Make text safe for ReportLab markup (XML escaping + Latin-1 fallback)."""
    value = text or ""
    for src, dst in _REPLACEMENTS.items():
        value = value.replace(src, dst)
    value = value.encode("latin-1", errors="replace").decode("latin-1")
    return escape(value).replace("\n", "<br/>")


def _fmt_date(d: date | None) -> str:
    return d.strftime("%d/%m/%Y") if d else "-"


def _styles() -> StyleSheet1:
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("CoverTitle", parent=ss["Title"], fontSize=24, textColor=PRIMARY, spaceAfter=18))
    ss.add(ParagraphStyle("H1", parent=ss["Heading1"], textColor=PRIMARY, spaceBefore=12, spaceAfter=8))
    ss.add(ParagraphStyle("H2", parent=ss["Heading2"], textColor=PRIMARY, spaceBefore=8, spaceAfter=6))
    ss.add(ParagraphStyle("Body", parent=ss["BodyText"], fontSize=9.5, leading=13))
    ss.add(ParagraphStyle("Cell", parent=ss["BodyText"], fontSize=8, leading=10))
    ss.add(ParagraphStyle("Small", parent=ss["BodyText"], fontSize=7.5, leading=9, textColor=colors.grey))
    ss.add(
        ParagraphStyle(
            "Score", parent=ss["Title"], fontSize=40, leading=48, alignment=TA_CENTER, textColor=PRIMARY
        )
    )
    return ss


def _table(rows: Sequence[Row], widths: Sequence[float], ss: StyleSheet1, extra: Sequence[Any] = ()) -> Table:
    data = [[Paragraph(_clean(c), ss["Cell"]) if isinstance(c, str) else c for c in row] for row in rows]
    table = Table(data, colWidths=list(widths), repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b0bec5")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                *extra,
            ]
        )
    )
    return table


def _header(labels: Sequence[str], ss: StyleSheet1) -> Row:
    style = ParagraphStyle("HeaderCell", parent=ss["Cell"], textColor=colors.white, fontName="Helvetica-Bold")
    return [Paragraph(_clean(label), style) for label in labels]


def _on_page(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, 1.2 * cm, FOOTER)
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


class _ReportBuilder:
    """Builds the story (list of flowables) section by section."""

    def __init__(self, report: AuditReport, done: Mapping[str, bool]) -> None:
        self.r = report
        self.done = done
        self.ss = _styles()
        self.width = A4[0] - 3.6 * cm
        self.story: list[Flowable] = []

    def _p(self, text: str, style: str = "Body") -> None:
        self.story.append(Paragraph(text, self.ss[style]))

    def cover(self) -> None:
        r, ai = self.r, self.r.ai_act
        self.story += [
            Spacer(1, 2 * cm),
            Paragraph("EU AI Act &amp; GDPR compliance audit", self.ss["CoverTitle"]),
            Paragraph(_clean(r.profile.project_name), self.ss["H1"]),
            Paragraph(_clean(r.profile.purpose), self.ss["Body"]),
            Spacer(1, 1 * cm),
            Paragraph(f"{r.compliance_score}/100", self.ss["Score"]),
            Paragraph(f"Compliance level: <b>{_clean(r.compliance_level)}</b>", self.ss["Body"]),
            Spacer(1, 0.6 * cm),
            _table(
                [
                    _header(["Item", "Value"], self.ss),
                    ["Audited document", r.file_name],
                    ["SHA-256 fingerprint", r.document_sha256[:32] + "..."],
                    ["Report date", r.generated_at.strftime("%d/%m/%Y %H:%M UTC")],
                    ["Report id", r.report_id],
                    ["AI Act category", ai.risk_category.label],
                    ["Role of the organisation", ai.operator_role.value],
                    ["DPIA required", _yes_no(r.gdpr.dpia_required)],
                    ["Models", ", ".join(f"{k}: {v}" for k, v in r.models.items())],
                    ["Rethinker iterations", str(r.rethink_iterations)],
                ],
                [5 * cm, self.width - 5 * cm],
                self.ss,
            ),
            Spacer(1, 0.8 * cm),
            Paragraph(_clean(r.disclaimer), self.ss["Small"]),
            PageBreak(),
        ]

    def summary_and_profile(self) -> None:
        r, p = self.r, self.r.profile
        self._p("1. Executive summary", "H1")
        self._p(_clean(r.executive_summary))
        if r.key_risks:
            self._p("Key risks", "H2")
            for risk in r.key_risks:
                self._p(f"- {_clean(risk)}")
        self._p("2. Project profile", "H1")
        rows: list[Row] = [
            _header(["Attribute", "Value"], self.ss),
            ["Sector", p.sector],
            ["Description", p.description],
            ["AI techniques", ", ".join(p.ai_techniques) or "-"],
            ["Affected persons", ", ".join(p.affected_persons) or "-"],
            ["Personal data", ", ".join(p.personal_data_categories) or "-"],
            ["Special categories (Art. 9)", ", ".join(p.special_category_data) or "-"],
            ["Automated decisions / profiling", f"{_yes_no(p.automated_decisions)} / {_yes_no(p.profiling)}"],
            ["Missing information", "; ".join(p.unknowns) or "-"],
        ]
        self.story.append(_table(rows, [5 * cm, self.width - 5 * cm], self.ss))

    def ai_act(self) -> None:
        ai, today = self.r.ai_act, self.r.generated_at.date()
        self._p("3. EU AI Act classification", "H1")
        secondary = ", ".join(c.label for c in ai.secondary_categories)
        self._p(
            f"<b>Category:</b> {_clean(ai.risk_category.label)} (confidence {ai.confidence:.0%})"
            + (f" - secondary: {_clean(secondary)}" if secondary else "")
        )
        if ai.annex_iii_areas:
            self._p(f"<b>Annex III areas:</b> {_clean(', '.join(ai.annex_iii_areas))}")
        self._p(f"<b>Rationale:</b> {_clean(ai.justification)}")
        if ai.obligations:
            self._p("Applicable obligations", "H2")
            rows: list[Row] = [_header(["Reference", "Obligation", "Status", "Gap"], self.ss)]
            extra = []
            for i, o in enumerate(ai.obligations, start=1):
                rows.append([o.reference, o.description, o.status.label, o.gap or "-"])
                extra.append(("TEXTCOLOR", (2, i), (2, i), STATUS_COLORS[o.status]))
            self.story.append(
                _table(rows, [2.8 * cm, 6.5 * cm, 2.6 * cm, self.width - 11.9 * cm], self.ss, extra)
            )
        self._p("Applicable deadlines", "H2")
        rows = [_header(["Date", "Deadline", "Legal basis", "Status"], self.ss)]
        for d in self.r.deadlines:
            status = "Applicable" if d.is_past(today) else f"In {d.days_remaining(today)} days"
            rows.append(
                [
                    _fmt_date(d.date),
                    d.title + (f" ({d.amended_by})" if d.amended_by else ""),
                    d.legal_basis,
                    status,
                ]
            )
        self.story.append(_table(rows, [2.3 * cm, 7.2 * cm, 4 * cm, self.width - 13.5 * cm], self.ss))

    def gdpr(self) -> None:
        g = self.r.gdpr
        self._p("4. GDPR audit", "H1")
        statuses = [g.legal_basis_status, g.minimisation_status, g.dpia_status, g.rights_status]
        dpia = ("Required: " + "; ".join(g.dpia_triggers)) if g.dpia_required else "Not required"
        rows: list[Row] = [
            _header(["Area", "Status", "Assessment"], self.ss),
            [
                "Lawful basis (Art. 6)",
                g.legal_basis_status.label,
                f"{g.identified_legal_basis.value} - {g.legal_basis_assessment}",
            ],
            ["Minimisation (Art. 5)", g.minimisation_status.label, g.minimisation_assessment],
            ["DPIA (Art. 35)", g.dpia_status.label, dpia],
            ["Rights (Art. 12-22)", g.rights_status.label, g.rights_assessment],
        ]
        extra = [("TEXTCOLOR", (1, i), (1, i), STATUS_COLORS[s]) for i, s in enumerate(statuses, start=1)]
        self.story.append(_table(rows, [3.8 * cm, 3 * cm, self.width - 6.8 * cm], self.ss, extra))
        if g.findings:
            self._p("Detailed findings", "H2")
            frows: list[Row] = [
                _header(["Topic", "Status / Severity", "Finding", "Recommendation", "Ref."], self.ss)
            ]
            for f in g.findings:
                refs = ", ".join(c.reference + ("" if c.verified else " (?)") for c in f.citations) or "-"
                frows.append(
                    [f.topic, f"{f.status.label} / {f.severity.value}", f.finding, f.recommendation, refs]
                )
            self.story.append(
                _table(frows, [2.6 * cm, 2.6 * cm, 4.7 * cm, 4.7 * cm, self.width - 14.6 * cm], self.ss)
            )

    def action_plan(self) -> None:
        self.story.append(PageBreak())
        self._p("5. Prioritised action plan", "H1")
        rows: list[Row] = [_header(["ID", "Priority", "Action", "Regulation", "Due", "Effort"], self.ss)]
        extra = []
        for i, r in enumerate(self.r.recommendations, start=1):
            action = Paragraph(f"<b>{_clean(r.title)}</b><br/>{_clean(r.description)}", self.ss["Cell"])
            rows.append(
                [
                    r.recommendation_id,
                    r.priority.value,
                    action,
                    f"{r.regulation.short} - {', '.join(r.references) or '-'}",
                    _fmt_date(r.due_date),
                    r.effort.value,
                ]
            )
            extra.append(("TEXTCOLOR", (1, i), (1, i), PRIORITY_COLORS[r.priority]))
        self.story.append(
            _table(
                rows, [1.6 * cm, 1.4 * cm, 7.6 * cm, 3.4 * cm, 2 * cm, self.width - 16 * cm], self.ss, extra
            )
        )

    def checklist(self) -> None:
        self._p("6. Compliance checklist", "H1")
        rows: list[Row] = [_header(["", "Action", "Reference", "Priority", "Due"], self.ss)]
        today = self.r.generated_at.date()
        completed = 0
        for item in self.r.checklist:
            checked = self.done.get(item.item_id, item.done)
            completed += checked
            rows.append(
                [
                    "[X]" if checked else "[ ]",
                    item.label,
                    f"{item.regulation.short}, {item.reference}",
                    item.priority.value,
                    "Applies now" if item.due_date and item.due_date <= today else _fmt_date(item.due_date),
                ]
            )
        self.story.append(_table(rows, [1 * cm, 8.4 * cm, 4 * cm, 1.6 * cm, self.width - 15 * cm], self.ss))
        self.story.append(Spacer(1, 0.2 * cm))
        self._p(f"Progress: {completed}/{len(self.r.checklist)} actions completed.")

    def annexes(self) -> None:
        self.story.append(PageBreak())
        self._p("Annex A - Legal provisions relied upon", "H1")
        rows: list[Row] = [_header(["Reference", "Title", "Excerpt", "Score"], self.ss)]
        for e in self.r.evidence[:MAX_EVIDENCE_ROWS]:
            excerpt = e.excerpt[:EXCERPT_CHARS] + ("..." if len(e.excerpt) > EXCERPT_CHARS else "")
            rows.append([e.citation, e.title, excerpt, f"{e.score:.3f}"])
        self.story.append(_table(rows, [3.2 * cm, 4 * cm, self.width - 8.7 * cm, 1.5 * cm], self.ss))
        if self.r.warnings:
            block: list[Flowable] = [Paragraph("Annex B - Quality-control warnings", self.ss["H1"])]
            block += [Paragraph(f"- {_clean(w)}", self.ss["Small"]) for w in self.r.warnings]
            self.story.append(KeepTogether(block))

    def build(self) -> list[Flowable]:
        self.cover()
        self.summary_and_profile()
        self.ai_act()
        self.gdpr()
        self.action_plan()
        self.checklist()
        self.annexes()
        return self.story


@validated
def generate_pdf(report: AuditReport, checklist_state: Mapping[str, bool] | None = None) -> bytes:
    """Render the report as PDF. ``checklist_state`` reflects the boxes ticked in the UI."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=2 * cm,
        title=f"Compliance audit - {report.profile.project_name}",
        author="EU AI Compliance Auditor",
    )
    doc.build(
        _ReportBuilder(report, checklist_state or {}).build(), onFirstPage=_on_page, onLaterPages=_on_page
    )
    return buffer.getvalue()


@validated
def export_json(report: AuditReport, checklist_state: Mapping[str, bool] | None = None) -> bytes:
    """Full JSON export (reflects the state of the interactive checklist)."""
    data = report.model_dump(mode="json")
    if checklist_state:
        for item in data["checklist"]:
            item["done"] = bool(checklist_state.get(item["item_id"], item["done"]))
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
