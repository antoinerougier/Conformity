"""Streamlit entry point - multi-page application.

streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="EU AI Compliance Auditor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    st.Page("pages/presentation.py", title="Overview & Architecture", icon="🏛️", default=True),
    st.Page("pages/audit.py", title="Audit & Chatbot", icon="🔍"),
    st.Page("pages/export.py", title="Export & Download", icon="📥"),
]

navigation = st.navigation(PAGES)

with st.sidebar:
    st.markdown("### ⚖️ EU AI Compliance Auditor")
    st.caption(
        "**100% local** audit of AI projects against the EU AI Act (Regulation (EU) 2024/1689) "
        "and the GDPR (Regulation (EU) 2016/679). No document ever leaves your machine."
    )
    st.divider()
    st.caption("Decision-support tool - not legal advice.")

navigation.run()
