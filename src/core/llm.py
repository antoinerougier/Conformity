"""Access to local models served by Ollama (reasoning LLM, judge, embeddings).

:class:`LLMService` is the interface consumed by the agents: it decouples LangGraph nodes
from the implementation (Ollama in production, a deterministic fake in tests).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, TypeVar, runtime_checkable

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel, ConfigDict, Field

from src.core.config import Settings
from src.core.exceptions import StructuredOutputError
from src.core.logging import get_logger
from src.core.types import NonEmptyStr, validated

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

__all__ = [
    "LLMService",
    "OllamaHealth",
    "OllamaLLMService",
    "PrefixedOllamaEmbeddings",
    "StructuredOutputError",
    "build_chat_model",
    "build_embeddings",
    "check_ollama",
]


@runtime_checkable
class LLMService(Protocol):
    """Minimal interface required by the agents."""

    model_name: str

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        """Call the model and return an instance of ``schema`` validated by Pydantic."""
        ...

    def stream(self, messages: Sequence[BaseMessage]) -> Iterator[str]:
        """Stream generated text (chatbot)."""
        ...


def build_chat_model(settings: Settings, *, model: str | None = None) -> ChatOllama:
    return ChatOllama(
        model=model or settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        num_ctx=settings.llm_num_ctx,
        client_kwargs={"timeout": settings.llm_timeout_s},
    )


class OllamaLLMService:
    """Ollama implementation with JSON-schema structured outputs and self-correction."""

    def __init__(self, settings: Settings, *, model: str | None = None) -> None:
        self.settings = settings
        self.model_name = model or settings.llm_model
        self._chat = build_chat_model(settings, model=self.model_name)

    @validated
    def structured(self, system: NonEmptyStr, user: NonEmptyStr, schema: type[T]) -> T:
        runnable = self._chat.with_structured_output(schema, method="json_schema", include_raw=True)
        messages: list[BaseMessage] = [SystemMessage(system), HumanMessage(user)]
        last_error = ""
        for attempt in range(self.settings.llm_max_retries + 1):
            result = runnable.invoke(messages)
            if not isinstance(result, dict):  # pragma: no cover - type guard
                raise StructuredOutputError(f"Unexpected response type: {type(result)!r}")
            parsed = result.get("parsed")
            if isinstance(parsed, schema):
                return parsed
            raw = result.get("raw")
            last_error = str(result.get("parsing_error"))
            logger.warning(
                "Invalid structured output (%s, attempt %d): %s",
                schema.__name__,
                attempt + 1,
                last_error[:300],
            )
            # Self-correction loop: feed the validation error back to the model.
            if isinstance(raw, AIMessage):
                messages = [
                    *messages,
                    raw,
                    HumanMessage(
                        "Your answer does not match the requested JSON schema. Error: "
                        f"{last_error[:1500]}\nReply only with a complete, valid JSON object."
                    ),
                ]
        raise StructuredOutputError(f"{schema.__name__}: failed after retries - {last_error}")

    def stream(self, messages: Sequence[BaseMessage]) -> Iterator[str]:
        if not messages:
            raise ValueError("messages must not be empty")
        for chunk in self._chat.stream(list(messages)):
            content = chunk.content
            if isinstance(content, str) and content:
                yield content


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class PrefixedOllamaEmbeddings(Embeddings):
    """Ollama embeddings, adding task prefixes for models that require them
    (e.g. ``nomic-embed-text``: ``search_document:`` / ``search_query:``)."""

    def __init__(self, settings: Settings) -> None:
        self.model_name = settings.embedding_model
        self._inner = OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url)
        self._batch = settings.embedding_batch_size
        needs_prefix = "nomic" in settings.embedding_model.lower()
        self._doc_prefix = "search_document: " if needs_prefix else ""
        self._query_prefix = "search_query: " if needs_prefix else ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            batch = [self._doc_prefix + t for t in texts[start : start + self._batch]]
            vectors.extend(self._inner.embed_documents(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(self._query_prefix + text)


def build_embeddings(settings: Settings) -> Embeddings:
    return PrefixedOllamaEmbeddings(settings)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class OllamaHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    reachable: bool
    available_models: list[str] = Field(default_factory=list)
    missing_models: list[str] = Field(default_factory=list)
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.reachable and not self.missing_models


def check_ollama(settings: Settings, timeout: float = 5.0) -> OllamaHealth:
    """Check that Ollama is reachable and that every configured model has been pulled."""
    required = {settings.llm_model, settings.judge_model, settings.embedding_model}
    try:
        response = httpx.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return OllamaHealth(reachable=False, missing_models=sorted(required), error=str(exc))
    names = sorted(m.get("name", "") for m in response.json().get("models", []))
    normalized = set(names) | {n.removesuffix(":latest") for n in names}
    missing = sorted(m for m in required if m not in normalized)
    return OllamaHealth(reachable=True, available_models=names, missing_models=missing)
