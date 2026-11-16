# syntax=docker/dockerfile:1.7
# =============================================================================
# EU AI Compliance Auditor - application image (Streamlit + LangGraph)
# Multi-stage build: uv resolves and installs dependencies into an isolated venv;
# the final image contains neither uv, nor a compiler, nor any cache.
# =============================================================================

ARG PYTHON_VERSION=3.12

# ----------------------------------------------------------------------------- 1. builder
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# Dependencies only (application code is copied as-is into the final image and run from
# /app): this layer is invalidated only when pyproject.toml / uv.lock change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Optional pre-download of the FlashRank reranker (ONNX, ~100 MB) for a fully offline runtime.
ARG PRELOAD_RERANKER=true
ARG RERANKER_MODEL=ms-marco-MultiBERT-L-12
RUN if [ "$PRELOAD_RERANKER" = "true" ]; then \
      /opt/venv/bin/python -c "from flashrank import Ranker; Ranker(model_name='${RERANKER_MODEL}', cache_dir='/opt/rerankers')"; \
    else mkdir -p /opt/rerankers; fi

# ----------------------------------------------------------------------------- 2. runtime
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="eu-ai-compliance-auditor" \
      org.opencontainers.image.description="Local EU AI Act & GDPR audit (RAG + LangGraph agents)" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    RERANKER_CACHE_DIR=/opt/rerankers \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder --chown=app:app /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/rerankers /opt/rerankers
COPY --chown=app:app src ./src
COPY --chown=app:app pages ./pages
COPY --chown=app:app app.py ./
COPY --chown=app:app .streamlit ./.streamlit
COPY --chown=app:app data/regulations ./data/regulations
COPY --chown=app:app data/processed ./data/processed
COPY --chown=app:app --chmod=0755 docker/app/entrypoint.sh /usr/local/bin/app-entrypoint.sh

RUN mkdir -p /app/data/uploads /app/data/reports /app/.cache && chown -R app:app /app/data /app/.cache

USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/app-entrypoint.sh"]
CMD ["streamlit", "run", "app.py"]
