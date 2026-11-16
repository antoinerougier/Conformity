#!/usr/bin/env bash
# Ollama container entrypoint: starts the server, pulls the required models when missing,
# then signals readiness through a sentinel file used by the docker-compose healthcheck.
set -euo pipefail

READY_FILE="/tmp/ollama-models-ready"
# Comma- or space-separated list. Duplicates are ignored.
OLLAMA_MODELS_TO_PULL="${OLLAMA_MODELS_TO_PULL:-qwen2.5:14b,bge-m3}"
MAX_PULL_ATTEMPTS=3

rm -f "${READY_FILE}"

echo "[ollama-init] Starting the Ollama server..."
ollama serve &
SERVER_PID=$!

# Forward stop signals to the server for a clean shutdown.
trap 'echo "[ollama-init] Stopping..."; kill -TERM "${SERVER_PID}" 2>/dev/null; wait "${SERVER_PID}"; exit 0' TERM INT

echo "[ollama-init] Waiting for the API..."
for _ in $(seq 1 120); do
  if ollama list >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! ollama list >/dev/null 2>&1; then
  echo "[ollama-init] ERROR: the Ollama API is not responding." >&2
  exit 1
fi

declare -A SEEN=()
for model in ${OLLAMA_MODELS_TO_PULL//,/ }; do
  [ -z "${model}" ] && continue
  [ -n "${SEEN[${model}]:-}" ] && continue
  SEEN[${model}]=1
  if ollama list | awk 'NR>1 {print $1}' | grep -qx -e "${model}" -e "${model}:latest"; then
    echo "[ollama-init] Model already present: ${model}"
    continue
  fi
  echo "[ollama-init] Pulling model: ${model}"
  attempt=1
  until ollama pull "${model}"; do
    if [ "${attempt}" -ge "${MAX_PULL_ATTEMPTS}" ]; then
      echo "[ollama-init] ERROR: failed to pull ${model} after ${MAX_PULL_ATTEMPTS} attempts." >&2
      exit 1
    fi
    attempt=$((attempt + 1))
    echo "[ollama-init] Retrying (${attempt}/${MAX_PULL_ATTEMPTS}) in 10s..."
    sleep 10
  done
done

touch "${READY_FILE}"
echo "[ollama-init] All models are ready: ${OLLAMA_MODELS_TO_PULL}"

wait "${SERVER_PID}"
