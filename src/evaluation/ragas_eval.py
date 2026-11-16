"""Ragas evaluation of the RAG pipeline (hybrid retrieval + reranker + grounded generation).

The judge (LLM + embeddings) is served locally by Ollama and plugged into Ragas through the
LangChain wrappers (``LangchainLLMWrapper``, ``LangchainEmbeddingsWrapper``).

Metrics:
- ``faithfulness``         : is the answer fully supported by the context?
                             (no hallucination about the legal provisions)
- ``nv_context_relevance`` : is the retrieved context relevant to the question?
- ``answer_relevancy``     : does the answer address the question?
- ``reference_hit_rate``   : (deterministic) share of questions whose reference provision
                             appears in the retrieved context.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.agents.chat import ComplianceAssistant
from src.core.config import Settings
from src.core.llm import build_chat_model, build_embeddings
from src.core.logging import get_logger
from src.evaluation.dataset import EvalSample

logger = get_logger(__name__)

#: Ragas metric name -> Settings attribute holding its minimum threshold.
METRIC_THRESHOLDS = {
    "faithfulness": "ragas_min_faithfulness",
    "nv_context_relevance": "ragas_min_context_relevance",
    "answer_relevancy": "ragas_min_answer_relevancy",
}


def _unit_key(chunk_id: str) -> str:
    """``gdpr:article:35:p2`` -> ``gdpr:article:35`` (ignore the sub-chunk suffix)."""
    return chunk_id.rsplit(":p", 1)[0]


class PipelineRow(BaseModel):
    """Output of the RAG pipeline for one evaluation sample."""

    model_config = ConfigDict(frozen=True)

    sample: EvalSample
    response: str
    retrieved_contexts: list[str]
    retrieved_chunk_ids: list[str]
    unknown_references: list[str] = Field(default_factory=list)

    @property
    def reference_hit(self) -> bool:
        wanted = {_unit_key(cid) for cid in self.sample.reference_chunk_ids}
        return not wanted or bool(wanted & {_unit_key(cid) for cid in self.retrieved_chunk_ids})

    def to_ragas(self) -> dict[str, Any]:
        return {
            "user_input": self.sample.user_input,
            "response": self.response,
            "retrieved_contexts": self.retrieved_contexts,
            "reference": self.sample.reference,
        }


class EvaluationReport(BaseModel):
    """Aggregated scores, thresholds and per-sample details."""

    scores: dict[str, float]
    thresholds: dict[str, float]
    per_sample: list[dict[str, Any]] = Field(default_factory=list)
    failed_metrics: dict[str, int] = Field(default_factory=dict)
    models: dict[str, str] = Field(default_factory=dict)

    def violations(self) -> list[str]:
        """Metrics below their threshold (NaN counts as a violation)."""
        return [
            f"{metric} = {self.scores.get(metric, math.nan):.3f} < {minimum:.2f}"
            for metric, minimum in self.thresholds.items()
            if not self.scores.get(metric, math.nan) >= minimum
        ]

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"ragas_{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
        payload = {**self.model_dump(), "violations": self.violations()}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path


def run_pipeline(assistant: ComplianceAssistant, samples: Sequence[EvalSample]) -> list[PipelineRow]:
    rows: list[PipelineRow] = []
    for sample in samples:
        result = assistant.answer(sample.user_input)
        rows.append(
            PipelineRow(
                sample=sample,
                response=result.answer,
                retrieved_contexts=result.context_texts,
                retrieved_chunk_ids=[c.chunk.chunk_id for c in result.contexts],
                unknown_references=result.unknown_references,
            )
        )
        logger.info("[%s] answer generated (%d contexts)", sample.id, len(result.contexts))
    return rows


def _mean(values: Sequence[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else math.nan


class RagasComponents(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    metrics: list[Any]
    judge: Any
    embeddings: Any


def build_ragas_components(settings: Settings) -> RagasComponents:
    """Ollama judge plugged into Ragas through the LangChain wrappers + selected metrics."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import ContextRelevance, Faithfulness, ResponseRelevancy

        return RagasComponents(
            metrics=[Faithfulness(), ContextRelevance(), ResponseRelevancy(strictness=1)],
            judge=LangchainLLMWrapper(build_chat_model(settings, model=settings.judge_model)),
            embeddings=LangchainEmbeddingsWrapper(build_embeddings(settings)),
        )


def evaluate_rows(rows: Sequence[PipelineRow], settings: Settings) -> EvaluationReport:
    """Run Ragas on the pipeline outputs and aggregate the scores."""
    if not rows:
        raise ValueError("No rows to evaluate")
    components = build_ragas_components(settings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas import EvaluationDataset, evaluate
        from ragas.dataset_schema import EvaluationResult
        from ragas.run_config import RunConfig

        result = evaluate(
            dataset=EvaluationDataset.from_list([r.to_ragas() for r in rows]),
            metrics=components.metrics,
            llm=components.judge,
            embeddings=components.embeddings,
            run_config=RunConfig(timeout=900, max_retries=3, max_workers=2),
            raise_exceptions=False,
            show_progress=False,
        )
    if not isinstance(result, EvaluationResult):  # return_executor=False -> always a result
        raise TypeError(f"Unexpected Ragas result: {type(result)!r}")
    df = result.to_pandas()

    scores: dict[str, float] = {}
    failed: dict[str, int] = {}
    for metric in METRIC_THRESHOLDS:
        column = [float(v) for v in df[metric].tolist()] if metric in df.columns else []
        scores[metric] = _mean(column)
        failed[metric] = sum(1 for v in column if math.isnan(v)) + (len(rows) - len(column))
    scores["reference_hit_rate"] = sum(r.reference_hit for r in rows) / len(rows)

    per_sample = [
        {
            "id": row.sample.id,
            "regulation": row.sample.regulation.value,
            "question": row.sample.user_input,
            "response": row.response,
            "retrieved_chunk_ids": row.retrieved_chunk_ids,
            "reference_hit": row.reference_hit,
            "unknown_references": row.unknown_references,
            **{m: record.get(m) for m in METRIC_THRESHOLDS},
        }
        for row, record in zip(rows, df.to_dict(orient="records"), strict=False)
    ]
    return EvaluationReport(
        scores=scores,
        thresholds={m: float(getattr(settings, attr)) for m, attr in METRIC_THRESHOLDS.items()},
        per_sample=per_sample,
        failed_metrics=failed,
        models={
            "generator": settings.llm_model,
            "judge": settings.judge_model,
            "embeddings": settings.embedding_model,
        },
    )
