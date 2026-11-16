"""JSON API of the web application (consumed by the pages' JavaScript)."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from src.agents.graph import ComplianceAuditor, ProgressEvent
from src.agents.interview import document_from_text, merge_documents, transcript
from src.api.services import AppServices, IndexingStatus, get_services
from src.api.sessions import AssessmentSession, ProgressEntry, SessionStatus
from src.compliance.timeline import all_deadlines
from src.core.exceptions import UnsupportedDocumentError
from src.core.llm import OllamaHealth
from src.core.logging import get_logger
from src.export.pdf_generator import export_json, generate_pdf
from src.ingestion.parser import SUPPORTED_EXTENSIONS, parse_document
from src.models.schemas import (
    AuditReport,
    ChatTurn,
    Deadline,
    InterviewAnswer,
    InterviewQuestion,
    NonEmptyStr,
    ProjectDocument,
    ProjectProfile,
)
from src.rag.preprocess_laws import preprocess_all, write_jsonl
from src.rag.vector_store import StoreHealth

logger = get_logger(__name__)

UNPROCESSABLE = 422  # literal: the Starlette constant was renamed between versions

router = APIRouter(prefix="/api", tags=["api"])
Services = Annotated[AppServices, Depends(get_services)]

_UPLOAD_CHUNK = 1 << 20
_SSE_POLL_SECONDS = 0.4
_SSE_KEEPALIVE_SECONDS = 15.0

# =============================================================================
# Response / request models
# =============================================================================


class HealthView(BaseModel):
    ollama: OllamaHealth
    store: StoreHealth
    reranker: str | None
    llm_model: str
    embedding_model: str

    @property
    def ready(self) -> bool:
        return self.ollama.ready and self.store.reachable and self.store.points > 0


class SessionView(BaseModel):
    """Public view of a session (never exposes the raw document)."""

    session_id: str
    status: SessionStatus
    profile: ProjectProfile | None
    questions: list[InterviewQuestion]
    round_number: int
    max_rounds: int
    answered_rounds: int
    progress: list[ProgressEntry]
    has_report: bool
    error: str | None

    @classmethod
    def of(cls, session: AssessmentSession, max_rounds: int) -> SessionView:
        current = session.current_round
        return cls(
            session_id=session.session_id,
            status=session.status,
            profile=session.profile,
            questions=current.questions if current else [],
            round_number=current.number if current else len(session.rounds),
            max_rounds=max_rounds,
            answered_rounds=sum(1 for r in session.rounds if r.answered),
            progress=session.progress,
            has_report=session.report is not None,
            error=session.error,
        )


class AnswersPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answers: list[InterviewAnswer] = Field(default_factory=list, max_length=20)
    finish: bool = Field(default=False, description="Stop the interview after these answers")


class ChecklistUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    done: bool


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Annotated[str, Field(max_length=2000)]


# =============================================================================
# Helpers
# =============================================================================


def _session(services: AppServices, session_id: str) -> AssessmentSession:
    return services.sessions.get(session_id)  # SessionNotFoundError -> 404 (exception handler)


def _view(services: AppServices, session: AssessmentSession) -> SessionView:
    return SessionView.of(session, services.settings.max_interview_rounds)


async def _read_upload(upload: UploadFile, max_mb: int) -> bytes:
    limit = max_mb * 1024 * 1024
    data = bytearray()
    while chunk := await upload.read(_UPLOAD_CHUNK):
        data.extend(chunk)
        if len(data) > limit:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File larger than {max_mb} MB")
    return bytes(data)


def _parse_upload(services: AppServices, filename: str, data: bytes) -> ProjectDocument:
    """Parse an upload through a temporary file that is always deleted (data minimisation)."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(f"Unsupported file type '{suffix}': upload a .pdf or .pptx file")
    uploads = services.settings.uploads_dir
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / f"{uuid.uuid4().hex}{suffix}"
    try:
        path.write_bytes(data)
        document = parse_document(path, max_size_mb=services.settings.max_upload_mb)
    finally:
        path.unlink(missing_ok=True)
    return document.model_copy(update={"file_name": Path(filename).name})


def _require_report(session: AssessmentSession) -> AuditReport:
    if session.report is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "The report has not been generated yet")
    return session.report


# =============================================================================
# System
# =============================================================================


@router.get("/health", response_model=HealthView)
def health(services: Services) -> HealthView:
    store = services.store_health()
    reranker = services.retriever.reranker.name if store.reachable else None
    return HealthView(
        ollama=services.ollama_health(),
        store=store,
        reranker=reranker,
        llm_model=services.settings.llm_model,
        embedding_model=services.settings.embedding_model,
    )


@router.get("/deadlines", response_model=list[Deadline])
def deadlines() -> list[Deadline]:
    return all_deadlines()


def _run_indexing(services: AppServices) -> None:
    services.indexing = IndexingStatus(state="running", message="Splitting the official texts")
    try:
        chunks = preprocess_all(services.settings)
        write_jsonl(chunks, services.settings.processed_chunks_path)
        services.indexing = IndexingStatus(
            state="running", chunks=len(chunks), message="Computing embeddings"
        )
        count = services.retriever.store.index_chunks(chunks, recreate=True)
        services.reset_retriever()
        services.indexing = IndexingStatus(state="done", chunks=count, message=f"{count} chunks indexed")
    except Exception as exc:  # reported to the UI
        logger.exception("Indexing failed")
        services.indexing = IndexingStatus(state="failed", message=str(exc))


@router.post("/index", response_model=IndexingStatus, status_code=status.HTTP_202_ACCEPTED)
def start_indexing(services: Services, background: BackgroundTasks) -> IndexingStatus:
    if services.indexing.state == "running":
        return services.indexing
    services.indexing = IndexingStatus(state="running", message="Starting")
    background.add_task(_run_indexing, services)
    return services.indexing


@router.get("/index", response_model=IndexingStatus)
def indexing_status(services: Services) -> IndexingStatus:
    return services.indexing


# =============================================================================
# Assessment sessions (interview)
# =============================================================================


@router.post("/sessions", response_model=SessionView, status_code=status.HTTP_201_CREATED)
async def create_session(
    services: Services,
    file: Annotated[UploadFile | None, File()] = None,
    description: Annotated[str, Form(max_length=20000)] = "",
) -> SessionView:
    """Start an assessment from an uploaded deck and/or a free-text description."""
    document: ProjectDocument | None = None
    if file is not None and file.filename:
        data = await _read_upload(file, services.settings.max_upload_mb)
        document = _parse_upload(services, file.filename, data)
        document = merge_documents(document, description)
    elif description.strip():
        document = document_from_text(description)
    if document is None:
        raise HTTPException(UNPROCESSABLE, "Upload a document or describe the project")

    interviewer = services.interviewer()
    profile = interviewer.profile(document)
    first_round = interviewer.next_round(document, profile, [])
    session = AssessmentSession(
        document=document,
        profile=profile,
        rounds=[first_round] if first_round else [],
        status="interviewing" if first_round else "ready",
    )
    services.sessions.create(session)
    return _view(services, session)


@router.get("/sessions/{session_id}", response_model=SessionView)
def get_session(session_id: str, services: Services) -> SessionView:
    return _view(services, _session(services, session_id))


@router.post("/sessions/{session_id}/answers", response_model=SessionView)
def submit_answers(session_id: str, payload: AnswersPayload, services: Services) -> SessionView:
    """Record the answers of the current round, then ask follow-ups or mark the session ready."""
    session = _session(services, session_id)
    if session.status != "interviewing":
        raise HTTPException(status.HTTP_409_CONFLICT, f"The interview is closed (status: {session.status})")
    document = session.document
    if document is None:  # pragma: no cover - sessions are always created with a document
        raise HTTPException(status.HTTP_409_CONFLICT, "No project document in this session")
    interviewer = services.interviewer()
    current = session.current_round
    rounds = list(session.rounds)
    if current is not None:
        rounds[-1] = interviewer.record_answers(current, payload.answers)
    session.rounds = rounds
    session.profile = interviewer.profile(document, rounds)
    next_round = None if payload.finish else interviewer.next_round(document, session.profile, rounds)
    if next_round is not None:
        session.rounds = [*rounds, next_round]
    else:
        session.status = "ready"
    services.sessions.save(session)
    return _view(services, session)


# =============================================================================
# Audit
# =============================================================================


def _run_audit(services: AppServices, session_id: str) -> None:
    session = services.sessions.get(session_id)
    try:
        auditor: ComplianceAuditor = services.auditor()
        document = session.document
        if document is None:  # pragma: no cover - guarded by start_audit
            raise RuntimeError("No project document in this session")
        initial = ComplianceAuditor.state_for_context(document, session.profile, transcript(session.rounds))
        for event in auditor.iter_events(initial):
            if isinstance(event, ProgressEvent):
                entry = ProgressEntry(
                    node=event.node, label=event.label, trace=event.trace, errors=event.errors
                )
                session.progress = [*session.progress, entry]
            else:
                session.report = event
                session.checklist_state = {item.item_id: item.done for item in event.checklist}
                session.status = "done"
            services.sessions.save(session)
    except Exception as exc:  # reported to the UI
        logger.exception("Audit failed for session %s", session_id)
        session.error = str(exc)
        session.status = "failed"
        services.sessions.save(session)


@router.post("/sessions/{session_id}/audit", response_model=SessionView, status_code=status.HTTP_202_ACCEPTED)
def start_audit(session_id: str, services: Services, background: BackgroundTasks) -> SessionView:
    """Close the interview and run the audit in the background."""
    session = _session(services, session_id)
    if session.status in {"auditing", "done"}:
        return _view(services, session)
    if session.document is None:  # pragma: no cover
        raise HTTPException(status.HTTP_409_CONFLICT, "No project document in this session")
    session.status = "auditing"
    session.progress = []
    session.error = None
    services.sessions.save(session)
    background.add_task(_run_audit, services, session_id)
    return _view(services, session)


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.get("/sessions/{session_id}/events")
def audit_events(session_id: str, services: Services) -> StreamingResponse:
    """Server-sent events: one ``progress`` event per graph node, then ``done`` or ``failed``."""
    _session(services, session_id)

    def stream() -> Iterator[str]:
        sent = 0
        last_beat = time.monotonic()
        while True:
            session = services.sessions.get(session_id)
            for entry in session.progress[sent:]:
                yield _sse("progress", entry.model_dump(mode="json"))
            sent = len(session.progress)
            if session.status == "done":
                yield _sse("done", {"session_id": session_id})
                return
            if session.status == "failed":
                yield _sse("failed", {"error": session.error})
                return
            if time.monotonic() - last_beat > _SSE_KEEPALIVE_SECONDS:
                last_beat = time.monotonic()
                yield ": keep-alive\n\n"
            time.sleep(_SSE_POLL_SECONDS)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# =============================================================================
# Report
# =============================================================================


@router.get("/sessions/{session_id}/report", response_model=AuditReport)
def get_report(session_id: str, services: Services) -> AuditReport:
    return _require_report(_session(services, session_id))


@router.patch("/sessions/{session_id}/checklist/{item_id}", response_model=ChecklistUpdate)
def update_checklist(
    session_id: str, item_id: str, payload: ChecklistUpdate, services: Services
) -> ChecklistUpdate:
    session = _session(services, session_id)
    report = _require_report(session)
    if item_id not in {i.item_id for i in report.checklist}:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown checklist item '{item_id}'")
    session.checklist_state = {**session.checklist_state, item_id: payload.done}
    services.sessions.save(session)
    return payload


def _filename(report: AuditReport, ext: Literal["pdf", "json"]) -> str:
    slug = "".join(c if c.isalnum() else "_" for c in report.profile.project_name.lower()).strip("_")[:40]
    return f"audit_{slug or 'project'}_{report.report_id}.{ext}"


@router.get("/sessions/{session_id}/report.pdf")
def download_pdf(session_id: str, services: Services) -> Response:
    session = _session(services, session_id)
    report = _require_report(session)
    return Response(
        generate_pdf(report, session.checklist_state),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{_filename(report, "pdf")}"'},
    )


@router.get("/sessions/{session_id}/report.json")
def download_json(session_id: str, services: Services) -> Response:
    session = _session(services, session_id)
    report = _require_report(session)
    return Response(
        export_json(report, session.checklist_state),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{_filename(report, "json")}"'},
    )


@router.post("/reports/import", response_model=SessionView, status_code=status.HTTP_201_CREATED)
async def import_report(services: Services, file: Annotated[UploadFile, File()]) -> SessionView:
    """Re-open a previously exported JSON report."""
    data = await _read_upload(file, services.settings.max_upload_mb)
    report = AuditReport.model_validate_json(data)
    session = AssessmentSession(
        status="done",
        profile=report.profile,
        report=report,
        checklist_state={item.item_id: item.done for item in report.checklist},
    )
    services.sessions.create(session)
    return _view(services, session)


@router.post("/sessions/{session_id}/chat")
def chat(session_id: str, payload: ChatRequest, services: Services) -> StreamingResponse:
    """Stream a grounded answer about the report; sources are returned in ``X-Sources``."""
    session = _session(services, session_id)
    report = _require_report(session)
    question: NonEmptyStr = payload.question.strip()
    if not question:
        raise HTTPException(UNPROCESSABLE, "The question is empty")
    assistant = services.assistant(report)
    tokens, evidence = assistant.stream_answer(question, session.chat_history)
    sources = list(dict.fromkeys(e.chunk.metadata.citation.split(" (part")[0] for e in evidence))[:6]

    def stream() -> Iterator[str]:
        parts: list[str] = []
        for token in tokens:
            parts.append(token)
            yield token
        answer = "".join(parts).strip()
        if unknown := assistant.validator.check_text(answer):
            yield "\n\n> Not found in the legal corpus, verify before relying on it: " + ", ".join(unknown)
        if answer:
            session.chat_history = [
                *session.chat_history,
                ChatTurn(role="user", content=question),
                ChatTurn(role="assistant", content=answer, citations=sources),
            ]
            services.sessions.save(session)

    return StreamingResponse(
        stream(), media_type="text/plain; charset=utf-8", headers={"X-Sources": json.dumps(sources)}
    )
