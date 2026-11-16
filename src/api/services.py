"""Application services shared by the routes (built lazily, injectable for tests)."""

from __future__ import annotations

import threading
from typing import Literal

from fastapi import Request
from pydantic import BaseModel

from src.agents.chat import ComplianceAssistant
from src.agents.graph import ComplianceAuditor
from src.agents.interview import InterviewEngine
from src.api.sessions import SessionStore
from src.compliance.citations import CitationValidator
from src.core.config import Settings
from src.core.llm import LLMService, OllamaHealth, OllamaLLMService, check_ollama
from src.models.schemas import AuditReport
from src.rag.retriever import HybridRetriever, build_retriever
from src.rag.vector_store import StoreHealth


class IndexingStatus(BaseModel):
    state: Literal["idle", "running", "done", "failed"] = "idle"
    chunks: int = 0
    message: str = ""


class AppServices:
    """Container of the long-lived services used by the web application.

    The heavy objects (retriever, LLM client) are created on first use so the application
    starts instantly and the overview page works even when Ollama or Qdrant are down.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        llm: LLMService | None = None,
        retriever: HybridRetriever | None = None,
        ollama_health: OllamaHealth | None = None,
    ) -> None:
        self.settings = settings
        self.sessions = SessionStore(ttl_hours=settings.session_ttl_hours, max_sessions=settings.max_sessions)
        self.indexing = IndexingStatus()
        self._llm = llm
        self._retriever = retriever
        self._static_health = ollama_health
        self._lock = threading.Lock()

    @property
    def llm(self) -> LLMService:
        with self._lock:
            if self._llm is None:
                self._llm = OllamaLLMService(self.settings)
            return self._llm

    @property
    def retriever(self) -> HybridRetriever:
        with self._lock:
            if self._retriever is None:
                self._retriever = build_retriever(self.settings)
            return self._retriever

    def reset_retriever(self) -> None:
        """Drop the cached retriever (after re-indexing) so the corpus is reloaded."""
        with self._lock:
            self._retriever = None

    @property
    def validator(self) -> CitationValidator:
        return CitationValidator.from_corpus(self.retriever.bm25.chunks)

    def interviewer(self) -> InterviewEngine:
        return InterviewEngine(self.llm, self.settings)

    def auditor(self) -> ComplianceAuditor:
        return ComplianceAuditor.from_settings(self.settings, llm=self.llm, retriever=self.retriever)

    def assistant(self, report: AuditReport | None) -> ComplianceAssistant:
        return ComplianceAssistant(self.llm, self.retriever, self.validator, self.settings, report=report)

    def ollama_health(self) -> OllamaHealth:
        return self._static_health or check_ollama(self.settings, timeout=3.0)

    def store_health(self) -> StoreHealth:
        try:
            return self.retriever.store.health()
        except Exception as exc:  # noqa: BLE001 - health endpoint must never raise
            return StoreHealth(reachable=False, error=str(exc))


def get_services(request: Request) -> AppServices:
    """FastAPI dependency returning the services attached to the application."""
    services = request.app.state.services
    if not isinstance(services, AppServices):  # pragma: no cover - misconfiguration guard
        raise RuntimeError("Application services are not configured")
    return services
