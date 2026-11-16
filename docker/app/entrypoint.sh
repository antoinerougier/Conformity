#!/usr/bin/env bash
# Application entrypoint: waits for Qdrant & Ollama, indexes the legal corpus when the
# collection is empty (AUTO_INDEX=true), then runs the command (Streamlit by default).
set -euo pipefail

QDRANT_URL="${QDRANT_URL:-http://qdrant:6333}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"
AUTO_INDEX="${AUTO_INDEX:-true}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-600}"

wait_for() {
  local name="$1" url="$2" waited=0
  echo "[entrypoint] Waiting for ${name} (${url})..."
  until curl -fsS "${url}" >/dev/null 2>&1; do
    sleep 3
    waited=$((waited + 3))
    if [ "${waited}" -ge "${WAIT_TIMEOUT}" ]; then
      echo "[entrypoint] ERROR: ${name} unavailable after ${WAIT_TIMEOUT}s" >&2
      exit 1
    fi
  done
  echo "[entrypoint] ${name} is available."
}

wait_for "Qdrant" "${QDRANT_URL%/}/readyz"
wait_for "Ollama" "${OLLAMA_BASE_URL%/}/api/tags"

if [ "${AUTO_INDEX}" = "true" ]; then
  echo "[entrypoint] Indexing the legal corpus (skipped if already present)..."
  python -m src.rag.preprocess_laws --index --if-empty || {
    echo "[entrypoint] WARNING: indexing failed - run it again from the Overview page." >&2
  }
fi

exec "$@"
