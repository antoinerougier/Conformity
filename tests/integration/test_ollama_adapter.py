"""Tests of the Ollama adapter against a fake HTTP server implementing the Ollama API.

Validates, without GPU or model: structured output (JSON schema sent, Pydantic parsing), the
self-correction loop on invalid JSON, streaming, embeddings and the health check.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from src.core.config import Settings
from src.core.llm import (
    OllamaLLMService,
    PrefixedOllamaEmbeddings,
    StructuredOutputError,
    check_ollama,
)
from src.models.schemas import QueryPlan, Regulation

pytestmark = pytest.mark.integration


class _FakeOllama(BaseHTTPRequestHandler):
    chat_replies: ClassVar[list[str]] = []
    requests: ClassVar[list[dict[str, Any]]] = []

    def log_message(self, *args: Any) -> None:  # keep test output quiet
        return

    def _send(self, payload: str, content_type: str = "application/json") -> None:
        body = payload.encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/tags":
            self._send(json.dumps({"models": [{"name": "qwen2.5:14b"}, {"name": "bge-m3:latest"}]}))
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append({"path": self.path, **payload})
        if self.path == "/api/chat":
            content = type(self).chat_replies.pop(0) if type(self).chat_replies else "{}"
            base = {"model": payload["model"], "created_at": "2026-11-16T00:00:00Z"}
            if payload.get("stream", True):
                words = content.split(" ")
                lines = [
                    json.dumps({**base, "message": {"role": "assistant", "content": w + " "}, "done": False})
                    for w in words
                ]
                lines.append(
                    json.dumps(
                        {
                            **base,
                            "message": {"role": "assistant", "content": ""},
                            "done": True,
                            "done_reason": "stop",
                        }
                    )
                )
                self._send("\n".join(lines) + "\n", "application/x-ndjson")
            else:
                self._send(
                    json.dumps({**base, "message": {"role": "assistant", "content": content}, "done": True})
                )
        elif self.path == "/api/embed":
            inputs = payload["input"] if isinstance(payload["input"], list) else [payload["input"]]
            self._send(
                json.dumps({"model": payload["model"], "embeddings": [[0.1, 0.2, 0.3] for _ in inputs]})
            )
        else:
            self.send_error(404)


@pytest.fixture()
def ollama_server() -> Iterator[str]:
    _FakeOllama.chat_replies = []
    _FakeOllama.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _settings(url: str, **kw: Any) -> Settings:
    return Settings(_env_file=None, ollama_base_url=url, llm_max_retries=1, **kw)


VALID_PLAN = json.dumps(
    {
        "sub_queries": [
            {"question": "Is CV screening high-risk?", "target": "ai_act", "rationale": "Annex III"}
        ]
    }
)


def test_structured_output_sends_json_schema_and_parses(ollama_server: str) -> None:
    _FakeOllama.chat_replies = [VALID_PLAN]
    service = OllamaLLMService(_settings(ollama_server))
    plan = service.structured("system", "user", QueryPlan)
    assert plan.sub_queries[0].target is Regulation.AI_ACT
    request = _FakeOllama.requests[0]
    assert request["model"] == "qwen2.5:14b"
    assert request["format"]["properties"].keys() >= {"sub_queries"}  # JSON schema sent to Ollama
    assert request["options"]["num_ctx"] == 16384


def test_structured_output_self_corrects_after_invalid_json(ollama_server: str) -> None:
    _FakeOllama.chat_replies = ['{"sub_queries": "pas une liste"}', VALID_PLAN]
    plan = OllamaLLMService(_settings(ollama_server)).structured("s", "u", QueryPlan)
    assert len(plan.sub_queries) == 1
    second = _FakeOllama.requests[1]["messages"]
    assert "does not match the requested JSON schema" in second[-1]["content"]  # error fed back


def test_structured_output_validates_inputs(ollama_server: str) -> None:
    with pytest.raises(ValidationError):
        OllamaLLMService(_settings(ollama_server)).structured("  ", "u", QueryPlan)


def test_structured_output_gives_up_after_retries(ollama_server: str) -> None:
    _FakeOllama.chat_replies = ["{}", "{}", "{}"]
    with pytest.raises(StructuredOutputError, match="QueryPlan"):
        OllamaLLMService(_settings(ollama_server)).structured("s", "u", QueryPlan)


def test_streaming(ollama_server: str) -> None:
    _FakeOllama.chat_replies = ["Human oversight is required."]
    tokens = list(OllamaLLMService(_settings(ollama_server)).stream([HumanMessage("q")]))
    assert "".join(tokens).strip() == "Human oversight is required."


def test_embeddings_add_nomic_prefixes(ollama_server: str) -> None:
    emb = PrefixedOllamaEmbeddings(_settings(ollama_server, embedding_model="nomic-embed-text"))
    assert emb.embed_documents(["a", "b"]) == [[0.1, 0.2, 0.3]] * 2
    emb.embed_query("q")
    inputs = [r["input"] for r in _FakeOllama.requests if r["path"] == "/api/embed"]
    assert inputs[0] == ["search_document: a", "search_document: b"]
    assert inputs[1] in ("search_query: q", ["search_query: q"])


def test_healthcheck(ollama_server: str) -> None:
    health = check_ollama(_settings(ollama_server))
    assert health.ready
    missing = check_ollama(_settings(ollama_server, llm_model="mistral-nemo"))
    assert missing.missing_models == ["mistral-nemo"]
    down = check_ollama(_settings("http://127.0.0.1:9"), timeout=0.5)
    assert not down.reachable
    assert down.error
