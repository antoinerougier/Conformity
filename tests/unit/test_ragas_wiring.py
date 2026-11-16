"""Compatibility of the Ragas integration (without running an LLM judge)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.core.config import Settings
from src.evaluation.dataset import EvalSample
from src.evaluation.ragas_eval import METRIC_THRESHOLDS, EvaluationReport, PipelineRow, build_ragas_components
from src.models.schemas import Regulation


def test_components_use_langchain_wrappers_and_expected_metrics() -> None:
    pytest.importorskip("ragas", reason="`eval` dependency group not installed")
    components = build_ragas_components(Settings(_env_file=None))
    assert {m.name for m in components.metrics} == set(METRIC_THRESHOLDS)
    assert type(components.judge).__name__ == "LangchainLLMWrapper"
    assert type(components.embeddings).__name__ == "LangchainEmbeddingsWrapper"


def test_reference_hit_ignores_part_suffix() -> None:
    sample = EvalSample(
        id="x",
        regulation=Regulation.GDPR,
        user_input="q",
        reference="r",
        reference_chunk_ids=["gdpr:article:35:p1"],
    )
    row = PipelineRow(
        sample=sample, response="a", retrieved_contexts=["ctx"], retrieved_chunk_ids=["gdpr:article:35:p2"]
    )
    assert row.reference_hit
    assert row.to_ragas()["retrieved_contexts"] == ["ctx"]
    miss = row.model_copy(update={"retrieved_chunk_ids": ["gdpr:article:6:p1"]})
    assert not miss.reference_hit


def test_quality_gate_violations(tmp_path: Path) -> None:
    report = EvaluationReport(
        scores={"faithfulness": 0.9, "nv_context_relevance": 0.4, "answer_relevancy": math.nan},
        thresholds={"faithfulness": 0.7, "nv_context_relevance": 0.6, "answer_relevancy": 0.6},
    )
    violations = report.violations()
    assert len(violations) == 2
    assert any(v.startswith("nv_context_relevance") for v in violations)
    assert any(v.startswith("answer_relevancy") for v in violations)  # NaN = failure
    assert report.save(tmp_path).is_file()
