"""Page 3 - Detailed report, interactive checklist and PDF / JSON export."""

from __future__ import annotations

import re
from itertools import groupby

import streamlit as st
from pydantic import ValidationError

from src.export.pdf_generator import export_json, generate_pdf
from src.models.schemas import AuditReport
from src.ui import services

_UNSAFE_FILENAME_RE = re.compile(r"[^a-z0-9_]+")


def _import_json() -> None:
    uploaded = st.file_uploader("...or reload a previously exported JSON report", type=["json"])
    if uploaded is None:
        return
    try:
        report = AuditReport.model_validate_json(uploaded.getvalue())
    except ValidationError as exc:
        st.error(f"Invalid JSON file: {exc.error_count()} validation error(s).")
        return
    services.set_report(report)
    st.success("Report loaded.")
    st.rerun()


def _checklist(report: AuditReport) -> dict[str, bool]:
    state = services.checklist_state()
    total = len(report.checklist)
    done = sum(1 for item in report.checklist if state.get(item.item_id, item.done))
    st.progress(done / total if total else 0.0, text=f"Progress: {done}/{total} actions completed")

    items = sorted(report.checklist, key=lambda it: it.priority.rank)
    for priority, group in groupby(items, key=lambda it: it.priority):
        st.markdown(f"##### {priority.label}")
        for item in group:
            due = f" - due {item.due_date:%d/%m/%Y}" if item.due_date else ""
            label = f"**{item.label}** · {item.regulation.short}, {item.reference}{due}"
            if item.source == "recommendation":
                label += " · _recommendation_"
            state[item.item_id] = st.checkbox(
                label, value=state.get(item.item_id, item.done), key=f"chk-{report.report_id}-{item.item_id}"
            )
    return state


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _detailed(report: AuditReport) -> None:
    ai, gdpr = report.ai_act, report.gdpr
    st.markdown(f"### {report.profile.project_name}")
    st.write(report.profile.description)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### EU AI Act")
        st.markdown(
            f"- **Category:** {ai.risk_category.label}\n- **Role:** {ai.operator_role.value}\n"
            f"- **AI system (Art. 3(1)):** {_yes_no(ai.is_ai_system)}\n"
            f"- **Annex I product component:** {_yes_no(ai.annex_i_product)}\n"
            f"- **Art. 6(3) derogation:** {_yes_no(ai.article_6_3_exception)}"
        )
        st.write(ai.justification)
    with col2:
        st.markdown("#### GDPR")
        st.markdown(
            f"- **Lawful basis:** {gdpr.identified_legal_basis.value} ({gdpr.legal_basis_status.label})\n"
            f"- **Minimisation:** {gdpr.minimisation_status.label}\n"
            f"- **DPIA:** {'required' if gdpr.dpia_required else 'not required'} ({gdpr.dpia_status.label})\n"
            f"- **Rights:** {gdpr.rights_status.label}"
        )
        st.write(gdpr.legal_basis_assessment)
    st.markdown("#### Deadlines")
    for d in report.deadlines:
        st.markdown(f"- **{d.date:%d/%m/%Y}** - {d.regulation.short} - {d.title} ({d.legal_basis})")
    if report.warnings:
        with st.expander(f"⚠️ Quality-control warnings ({len(report.warnings)})"):
            for w in report.warnings:
                st.caption(w)


def _file_stem(report: AuditReport) -> str:
    slug = _UNSAFE_FILENAME_RE.sub("_", report.profile.project_name.lower()).strip("_")[:40] or "project"
    return f"audit_{slug}_{report.report_id}"


def render() -> None:
    st.title("📥 Export & Download")
    report = services.get_report()
    if report is None:
        st.info("No report in memory. Run an audit from the 'Audit & Chatbot' page.")
        _import_json()
        return

    tab_report, tab_checklist, tab_export = st.tabs(["📄 Detailed report", "☑️ Checklist", "⬇️ Export"])
    with tab_report:
        _detailed(report)
    with tab_checklist:
        state = _checklist(report)
    with tab_export:
        stem = _file_stem(report)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**PDF report**")
            st.caption(
                "Summary, classification, GDPR audit, action plan, checklist (current state), annexes."
            )
            st.download_button(
                "Download PDF",
                data=generate_pdf(report, state),
                file_name=f"{stem}.pdf",
                mime="application/pdf",
                type="primary",
                width="stretch",
            )
        with c2:
            st.markdown("**JSON report**")
            st.caption("Complete structured data (can be reloaded in this page).")
            st.download_button(
                "Download JSON",
                data=export_json(report, state),
                file_name=f"{stem}.json",
                mime="application/json",
                width="stretch",
            )
        st.divider()
        _import_json()


render()
