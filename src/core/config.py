"""Centralised application settings (pydantic-settings).

Every value can be overridden with an environment variable or a `.env` file at the
repository root. No data leaves the machine: default URLs point to local services
(Ollama, Qdrant).
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RerankerBackend(StrEnum):
    """Available reranker implementations."""

    FLASHRANK = "flashrank"
    CROSS_ENCODER = "cross_encoder"
    NONE = "none"


class Settings(BaseSettings):
    """Application settings, validated at startup."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_assignment=True,
    )

    # ------------------------------------------------------------------ Ollama
    ollama_base_url: str = Field(default="http://localhost:11434", pattern=r"^https?://")
    llm_model: str = Field(default="qwen2.5:14b", min_length=1)
    judge_model: str = Field(default="qwen2.5:14b", min_length=1)
    embedding_model: str = Field(default="bge-m3", min_length=1)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_num_ctx: int = Field(default=16384, ge=2048)
    llm_timeout_s: float = Field(default=600.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    embedding_batch_size: int = Field(default=16, ge=1, le=256)

    # ------------------------------------------------------------------ Qdrant
    qdrant_url: str | None = "http://localhost:6333"
    qdrant_path: Path | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = Field(default="eu_legal_corpus", pattern=r"^[A-Za-z0-9_\-]+$")

    # ------------------------------------------------------------------ Legal corpus
    regulations_dir: Path = PROJECT_ROOT / "data" / "regulations"
    ai_act_pdf: str = "ai_act_2024_1689_en.pdf"
    gdpr_pdf: str = "gdpr_2016_679_fr.pdf"
    processed_chunks_path: Path = PROJECT_ROOT / "data" / "processed" / "legal_chunks.jsonl"
    max_chunk_chars: int = Field(default=3500, ge=800)

    # ------------------------------------------------------------------ Retrieval
    dense_top_k: int = Field(default=20, ge=1)
    sparse_top_k: int = Field(default=20, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    rerank_candidates: int = Field(default=20, ge=1)
    rerank_top_n: int = Field(default=6, ge=1)
    reranker_backend: RerankerBackend = RerankerBackend.FLASHRANK
    reranker_model: str = "ms-marco-MultiBERT-L-12"
    cross_encoder_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_cache_dir: Path = PROJECT_ROOT / ".cache" / "rerankers"

    # ------------------------------------------------------------------ Agents
    max_rethink_iterations: int = Field(default=2, ge=0, le=5)
    max_sub_queries: int = Field(default=10, ge=2, le=30)
    project_context_chars: int = Field(default=24000, ge=2000)
    evidence_context_chars: int = Field(default=18000, ge=2000)

    # ------------------------------------------------------------------ Ingestion
    max_upload_mb: int = Field(default=50, ge=1, le=500)
    uploads_dir: Path = PROJECT_ROOT / "data" / "uploads"
    reports_dir: Path = PROJECT_ROOT / "data" / "reports"

    # ------------------------------------------------------------------ Ragas evaluation
    ragas_min_faithfulness: float = Field(default=0.70, ge=0.0, le=1.0)
    ragas_min_context_relevance: float = Field(default=0.60, ge=0.0, le=1.0)
    ragas_min_answer_relevancy: float = Field(default=0.60, ge=0.0, le=1.0)
    ragas_max_samples: int = Field(default=20, ge=1)
    ragas_testset_path: Path = PROJECT_ROOT / "tests" / "eval" / "data" / "golden_testset.jsonl"
    ragas_synthetic_testset_path: Path = PROJECT_ROOT / "tests" / "eval" / "data" / "synthetic_testset.jsonl"

    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    @field_validator("qdrant_url", "qdrant_api_key", mode="before")
    @classmethod
    def _empty_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_retrieval_budget(self) -> Settings:
        if self.rerank_top_n > self.rerank_candidates:
            raise ValueError("rerank_top_n must be <= rerank_candidates")
        return self

    @property
    def ai_act_path(self) -> Path:
        return self.regulations_dir / self.ai_act_pdf

    @property
    def gdpr_path(self) -> Path:
        return self.regulations_dir / self.gdpr_pdf


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton settings instance."""
    return Settings()
