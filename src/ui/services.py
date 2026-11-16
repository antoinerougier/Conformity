"""Services shared by the Streamlit pages (cached resources, session state)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import streamlit as st

from src.agents.chat import ComplianceAssistant
from src.agents.graph import ComplianceAuditor
from src.compliance.citations import CitationValidator
from src.core.config import Settings, get_settings
from src.core.llm import OllamaHealth, OllamaLLMService, check_ollama
from src.core.logging import configure_logging
from src.core.types import NonEmptyStr, validated
from src.ingestion.parser import SUPPORTED_EXTENSIONS
from src.models.schemas import AuditReport, ChatTurn
from src.rag.retriever import HybridRetriever, build_retriever
from src.rag.vector_store import StoreHealth

REPORT_KEY = "audit_report"
CHAT_KEY = "chat_history"
CHECKLIST_KEY = "checklist_state"
TRACE_KEY = "audit_trace"


@st.cache_resource(show_spinner=False)
def settings() -> Settings:
    s = get_settings()
    configure_logging(s.log_level)
    return s


@st.cache_resource(show_spinner="Loading the legal corpus and the reranker...")
def retriever() -> HybridRetriever:
    return build_retriever(settings())


@st.cache_resource(show_spinner=False)
def auditor() -> ComplianceAuditor:
    return ComplianceAuditor.from_settings(settings(), retriever=retriever())


def assistant(report: AuditReport | None) -> ComplianceAssistant:
    r = retriever()
    return ComplianceAssistant(
        llm=OllamaLLMService(settings()),
        retriever=r,
        validator=CitationValidator.from_corpus(r.bm25.chunks),
        settings=settings(),
        report=report,
    )


@st.cache_data(ttl=15, show_spinner=False)
def ollama_health() -> OllamaHealth:
    return check_ollama(settings())


def store_health() -> StoreHealth:
    return retriever().store.health()


@validated
def save_upload(name: NonEmptyStr, data: bytes) -> Path:
    """Store an uploaded file under a random name (no path traversal, allowed types only)."""
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")
    if not data:
        raise ValueError("Uploaded file is empty")
    s = settings()
    s.uploads_dir.mkdir(parents=True, exist_ok=True)
    target = s.uploads_dir / f"{uuid.uuid4().hex}{suffix}"
    target.write_bytes(data)
    return target


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def get_report() -> AuditReport | None:
    value = st.session_state.get(REPORT_KEY)
    return value if isinstance(value, AuditReport) else None


def set_report(report: AuditReport, trace: list[str] | None = None) -> None:
    st.session_state[REPORT_KEY] = report
    st.session_state[CHAT_KEY] = []
    st.session_state[CHECKLIST_KEY] = {item.item_id: item.done for item in report.checklist}
    st.session_state[TRACE_KEY] = trace or []


def chat_history() -> list[ChatTurn]:
    history: Any = st.session_state.setdefault(CHAT_KEY, [])
    return history if isinstance(history, list) else []


def checklist_state() -> dict[str, bool]:
    state: Any = st.session_state.setdefault(CHECKLIST_KEY, {})
    return state if isinstance(state, dict) else {}


def audit_trace() -> list[str]:
    trace: Any = st.session_state.get(TRACE_KEY, [])
    return trace if isinstance(trace, list) else []
