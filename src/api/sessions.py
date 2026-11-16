"""In-memory assessment sessions.

A session follows one project through the three steps of the web application:
``interviewing`` -> ``ready`` -> ``auditing`` -> ``done`` (or ``failed``).

Sessions live only in memory (privacy by design: nothing is written to disk, the uploaded
file is deleted right after parsing) and expire after ``session_ttl_hours``.
"""

from __future__ import annotations

import secrets
import threading
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.models.schemas import AuditReport, ChatTurn, InterviewRound, ProjectDocument, ProjectProfile

SessionStatus = Literal["interviewing", "ready", "auditing", "done", "failed"]


def _now() -> datetime:
    return datetime.now(UTC)


class ProgressEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    node: str
    label: str
    trace: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    at: datetime = Field(default_factory=_now)


class AssessmentSession(BaseModel):
    """State of one assessment (validated on every assignment)."""

    model_config = ConfigDict(validate_assignment=True)

    session_id: str = Field(default_factory=lambda: secrets.token_urlsafe(16))
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    status: SessionStatus = "interviewing"
    document: ProjectDocument | None = None
    profile: ProjectProfile | None = None
    rounds: list[InterviewRound] = Field(default_factory=list)
    progress: list[ProgressEntry] = Field(default_factory=list)
    report: AuditReport | None = None
    checklist_state: dict[str, bool] = Field(default_factory=dict)
    chat_history: list[ChatTurn] = Field(default_factory=list)
    error: str | None = None

    @property
    def current_round(self) -> InterviewRound | None:
        """The round waiting for answers, if any."""
        return self.rounds[-1] if self.rounds and not self.rounds[-1].answered else None

    def touch(self) -> None:
        self.updated_at = _now()


class SessionNotFoundError(KeyError):
    """Unknown or expired session id."""


class SessionStore:
    """Thread-safe, bounded, expiring in-memory store."""

    def __init__(self, *, ttl_hours: int, max_sessions: int) -> None:
        self._ttl = timedelta(hours=ttl_hours)
        self._max = max_sessions
        self._items: OrderedDict[str, AssessmentSession] = OrderedDict()
        self._lock = threading.RLock()

    def _purge(self) -> None:
        cutoff = _now() - self._ttl
        for sid in [sid for sid, s in self._items.items() if s.updated_at < cutoff]:
            del self._items[sid]
        while len(self._items) > self._max:
            self._items.popitem(last=False)

    def create(self, session: AssessmentSession | None = None) -> AssessmentSession:
        with self._lock:
            session = session or AssessmentSession()
            self._items[session.session_id] = session
            self._purge()
            return session

    def get(self, session_id: str) -> AssessmentSession:
        with self._lock:
            self._purge()
            try:
                session = self._items[session_id]
            except KeyError as exc:
                raise SessionNotFoundError(session_id) from exc
            self._items.move_to_end(session_id)
            return session

    def save(self, session: AssessmentSession) -> None:
        with self._lock:
            session.touch()
            self._items[session.session_id] = session

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
