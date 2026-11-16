"""Page 2 - Project audit (.pdf / .pptx), agent progress, report and chatbot."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.agents.graph import ProgressEvent
from src.core.exceptions import ComplianceAuditorError
from src.models.schemas import AuditReport, ChatTurn, ComplianceStatus, Priority
from src.ui import services

STATUS_ICONS = {
    ComplianceStatus.COMPLIANT: "🟢",
    ComplianceStatus.PARTIALLY_COMPLIANT: "🟠",
    ComplianceStatus.NON_COMPLIANT: "🔴",
    ComplianceStatus.TO_VERIFY: "🟤",
    ComplianceStatus.NOT_APPLICABLE: "⚪",
}


def _status(status: ComplianceStatus) -> str:
    return f"{STATUS_ICONS[status]} {status.label}"


def _verified(verified: bool) -> str:
    return "✔️" if verified else "❔"


def _run_audit(uploaded_name: str, data: bytes) -> None:
    if not services.ollama_health().ready:
        st.error("Ollama is not ready (see Overview -> Health).")
        return
    if services.store_health().points == 0:
        st.error("The legal corpus is not indexed (Overview -> Health & indexing).")
        return

    path = services.save_upload(uploaded_name, data)
    trace: list[str] = []
    try:
        with st.status("Audit in progress...", expanded=True) as status:
            for event in services.auditor().stream(path):
                if isinstance(event, ProgressEvent):
                    status.update(label=event.label)
                    st.markdown(f"**{event.label}**")
                    for line in event.trace:
                        st.caption(line)
                        trace.append(f"{event.label} - {line}")
                    for err in event.errors:
                        st.caption(f"⚠️ {err}")
                else:
                    services.set_report(event.model_copy(update={"file_name": uploaded_name}), trace)
            status.update(label="Audit complete ✅", state="complete", expanded=False)
    except ComplianceAuditorError as exc:
        st.error(f"Audit failed: {exc}")
    except Exception as exc:  # noqa: BLE001 - surface any unexpected error to the user
        st.exception(exc)
    finally:
        path.unlink(missing_ok=True)  # data minimisation: the document is not kept


def _summary_tab(report: AuditReport) -> None:
    st.markdown(f"#### {report.profile.project_name}")
    st.write(report.executive_summary)
    if report.key_risks:
        st.markdown("**Key risks**\n" + "\n".join(f"- {r}" for r in report.key_risks))
    if report.profile.unknowns:
        st.info("**Information missing from the document:** " + "; ".join(report.profile.unknowns))


def _ai_act_tab(report: AuditReport) -> None:
    ai = report.ai_act
    st.markdown(f"**Category:** {ai.risk_category.label} - confidence {ai.confidence:.0%}")
    if ai.secondary_categories:
        st.markdown("**Secondary categories:** " + ", ".join(c.label for c in ai.secondary_categories))
    if ai.annex_iii_areas:
        st.markdown("**Annex III areas:** " + ", ".join(ai.annex_iii_areas))
    if ai.prohibited_practice_flags:
        st.error("**Potentially prohibited practices:** " + "; ".join(ai.prohibited_practice_flags))
    st.write(ai.justification)
    if ai.obligations:
        rows = [
            {
                "Reference": o.reference,
                "Obligation": o.description,
                "Status": _status(o.status),
                "Gap": o.gap or "",
            }
            for o in ai.obligations
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("**Applicable deadlines**")
    today = report.generated_at.date()
    for d in ai.applicable_deadlines:
        badge = "✅ applicable" if d.is_past(today) else f"⏳ D-{d.days_remaining(today)}"
        st.markdown(f"- **{d.date:%d/%m/%Y}** - {d.title} ({d.legal_basis}) · {badge}")
    if ai.citations:
        st.markdown(
            "**Citations:** " + " · ".join(f"{_verified(c.verified)} {c.reference}" for c in ai.citations)
        )


def _gdpr_tab(report: AuditReport) -> None:
    g = report.gdpr
    rows = [
        {
            "Area": "Lawful basis (Art. 6)",
            "Status": _status(g.legal_basis_status),
            "Assessment": f"{g.identified_legal_basis.value} - {g.legal_basis_assessment}",
        },
        {
            "Area": "Minimisation (Art. 5)",
            "Status": _status(g.minimisation_status),
            "Assessment": g.minimisation_assessment,
        },
        {
            "Area": "DPIA (Art. 35)",
            "Status": _status(g.dpia_status),
            "Assessment": "; ".join(g.dpia_triggers) or "-",
        },
        {
            "Area": "Rights (Art. 12-22)",
            "Status": _status(g.rights_status),
            "Assessment": g.rights_assessment,
        },
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    for f in g.findings:
        with st.expander(f"{_status(f.status)} · {f.topic} · severity {f.severity.value}"):
            st.write(f.finding)
            st.markdown(f"**Recommendation:** {f.recommendation}")
            if f.citations:
                st.caption(" · ".join(f"{_verified(c.verified)} {c.reference}" for c in f.citations))


def _plan_tab(report: AuditReport) -> None:
    rows = [
        {
            "ID": r.recommendation_id,
            "Priority": r.priority.label,
            "Action": r.title,
            "Details": r.description,
            "Regulation": r.regulation.short,
            "References": ", ".join(r.references),
            "Due": r.due_date.strftime("%d/%m/%Y") if r.due_date else "",
            "Effort": r.effort.value,
        }
        for r in report.recommendations
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("➡️ Interactive checklist and PDF/JSON export: see the 'Export & Download' page.")


def _summary(report: AuditReport) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Compliance score", f"{report.compliance_score}/100", report.compliance_level, delta_color="off"
    )
    c2.metric("AI Act category", report.ai_act.risk_category.label)
    c3.metric("DPIA required", "Yes" if report.gdpr.dpia_required else "No")
    blocking = sum(r.priority is Priority.P0 for r in report.recommendations)
    c4.metric("P0 actions / total", f"{blocking} / {len(report.recommendations)}")

    tabs = st.tabs(["🧾 Summary", "⚖️ AI Act", "🔐 GDPR", "✅ Action plan", "📚 Evidence", "🛠️ Trace"])
    with tabs[0]:
        _summary_tab(report)
    with tabs[1]:
        _ai_act_tab(report)
    with tabs[2]:
        _gdpr_tab(report)
    with tabs[3]:
        _plan_tab(report)
    with tabs[4]:
        st.dataframe(
            pd.DataFrame([e.model_dump() for e in report.evidence]), hide_index=True, width="stretch"
        )
    with tabs[5]:
        for line in services.audit_trace():
            st.caption(line)
        if report.warnings:
            st.warning("**Quality control:**\n" + "\n".join(f"- {w}" for w in report.warnings))


def _chat(report: AuditReport) -> None:
    st.subheader("💬 Compliance assistant")
    st.caption(
        "Ask questions to improve the compliance of the project. "
        "Answers rely on the audit report and the official legal texts."
    )
    history = services.chat_history()
    for turn in history:
        with st.chat_message(turn.role):
            st.markdown(turn.content)
            if turn.citations:
                st.caption("Sources: " + " · ".join(turn.citations))
    question = st.chat_input("E.g. Which human oversight measures should we implement first?")
    if not question or not question.strip():
        return
    with st.chat_message("user"):
        st.markdown(question)
    assistant = services.assistant(report)
    with st.chat_message("assistant"):
        stream, evidence = assistant.stream_answer(question, history)
        answer = st.write_stream(stream)
        text = answer if isinstance(answer, str) else "".join(str(a) for a in answer)
        sources = [e.chunk.metadata.citation for e in evidence]
        st.caption("Sources: " + " · ".join(sources))
        if unknown := assistant.validator.check_text(text):
            st.warning("References not found in the corpus (to verify): " + ", ".join(unknown))
    if text.strip():
        history.extend(
            [
                ChatTurn(role="user", content=question),
                ChatTurn(role="assistant", content=text, citations=sources),
            ]
        )


def render() -> None:
    st.title("🔍 Audit & interactive chatbot")
    uploaded = st.file_uploader(
        "Upload the presentation of your AI project",
        type=["pdf", "pptx"],
        help="The file is processed locally and deleted at the end of the audit.",
    )
    if uploaded is not None and st.button("🚀 Run the audit", type="primary"):
        _run_audit(uploaded.name, uploaded.getvalue())

    report = services.get_report()
    if report is None:
        st.info("No report yet: upload a document and run the audit.")
        return
    st.divider()
    st.caption(f"Report `{report.report_id}` - {report.file_name} - {report.generated_at:%d/%m/%Y %H:%M} UTC")
    _summary(report)
    st.divider()
    _chat(report)


render()
