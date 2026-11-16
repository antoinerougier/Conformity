"""FastAPI application factory.

Run locally:
    uvicorn src.api.app:app --reload
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from src.api.routes_api import router as api_router
from src.api.routes_pages import router as pages_router
from src.api.routes_pages import templates
from src.api.services import AppServices
from src.api.sessions import SessionNotFoundError
from src.core.config import get_settings
from src.core.exceptions import ComplianceAuditorError
from src.core.logging import configure_logging

STATIC_DIR = Path(__file__).parent / "static"
UNPROCESSABLE = 422  # literal: the Starlette constant was renamed between versions

# Everything is served by the application itself: no CDN, no third-party request.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
    "script-src 'self'; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'"
)


def _wants_html(request: Request) -> bool:
    return not request.url.path.startswith("/api") and "text/html" in request.headers.get("accept", "")


def _error(request: Request, code: int, message: str) -> Response:
    if _wants_html(request):
        return templates.TemplateResponse(
            request, "error.html", {"page": "error", "code": code, "message": message}, status_code=code
        )
    return JSONResponse({"detail": message}, status_code=code)


def create_app(services: AppServices | None = None) -> FastAPI:
    """Build the application. Tests inject ``services`` with fakes."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = app.state.services.settings
        configure_logging(settings.log_level)
        yield

    app = FastAPI(
        title="EU AI Compliance Auditor",
        description="Local compliance audit of AI projects against the EU AI Act and the GDPR.",
        version="2.0.0",
        lifespan=lifespan,
        # Swagger UI loads its assets from a CDN, which the CSP forbids: only the schema is served.
        docs_url=None,
        redoc_url=None,
    )
    app.state.services = services or AppServices(get_settings())

    @app.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("X-Frame-Options", "DENY")
        return response

    @app.exception_handler(SessionNotFoundError)
    async def session_not_found(request: Request, exc: SessionNotFoundError) -> Response:
        return _error(request, status.HTTP_404_NOT_FOUND, "This assessment does not exist or has expired.")

    @app.exception_handler(ComplianceAuditorError)
    async def domain_error(request: Request, exc: ComplianceAuditorError) -> Response:
        return _error(request, UNPROCESSABLE, str(exc))

    @app.exception_handler(ValidationError)
    async def validation_error(request: Request, exc: ValidationError) -> Response:
        return _error(request, UNPROCESSABLE, exc.errors()[0]["msg"])

    @app.exception_handler(ValueError)
    async def value_error(request: Request, exc: ValueError) -> Response:
        return _error(request, UNPROCESSABLE, str(exc))

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(request: Request, exc: RequestValidationError) -> Response:
        first = exc.errors()[0] if exc.errors() else {"msg": "Invalid request"}
        return _error(request, UNPROCESSABLE, str(first.get("msg")))

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, str]:
        """Liveness probe (does not touch Ollama or Qdrant)."""
        return {"status": "ok"}

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(api_router)
    app.include_router(pages_router)
    return app


app = create_app()
