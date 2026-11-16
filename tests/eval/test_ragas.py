"""Ragas evaluation of the legal RAG pipeline, run in CI (`pytest -m eval tests/eval`).

Prerequisites: Ollama (LLM, judge and embedding models) and a populated Qdrant index
(`python -m src.rag.preprocess_laws --index`). Without these services the test is skipped
locally and **fails** in CI (`REQUIRE_OLLAMA=1`).

Configurable thresholds: RAGAS_MIN_FAITHFULNESS, RAGAS_MIN_CONTEXT_RELEVANCE,
RAGAS_MIN_ANSWER_RELEVANCY; sample size: RAGAS_MAX_SAMPLES.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.agents.chat import ComplianceAssistant
from src.compliance.citations import CitationValidator
from src.core.config import Settings
from src.core.llm import OllamaLLMService, check_ollama
from src.evaluation.dataset import load_testset
from src.evaluation.ragas_eval import EvaluationReport, evaluate_rows, run_pipeline
from src.rag.retriever import build_retriever

pytestmark = pytest.mark.eval

REPORTS_DIR = Path(__file__).parent / "reports"
MIN_REFERENCE_HIT_RATE = 0.70
MAX_FAILED_SHARE = 0.25


def _require(condition: bool, reason: str) -> None:
    if condition:
        return
    if os.getenv("REQUIRE_OLLAMA") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="module")
def evaluation(settings: Settings) -> EvaluationReport:
    health = check_ollama(settings)
    _require(health.reachable, f"Ollama unreachable ({settings.ollama_base_url}): {health.error}")
    _require(not health.missing_models, f"Missing Ollama models: {health.missing_models}")

    retriever = build_retriever(settings)
    _require(
        retriever.store.count() > 0, "Empty Qdrant index: run `python -m src.rag.preprocess_laws --index`"
    )

    samples = load_testset(settings, max_samples=settings.ragas_max_samples)
    assert samples, "Empty test set"

    assistant = ComplianceAssistant(
        llm=OllamaLLMService(settings),
        retriever=retriever,
        validator=CitationValidator.from_corpus(retriever.bm25.chunks),
        settings=settings,
    )
    report = evaluate_rows(run_pipeline(assistant, samples), settings)
    path = report.save(REPORTS_DIR)
    print(f"\nRagas report: {path}\nScores: {report.scores}")  # noqa: T201 - surfaced in CI logs
    return report


def test_quality_gate(evaluation: EvaluationReport) -> None:
    """Quality thresholds: faithfulness, context relevance, answer relevancy."""
    violations = evaluation.violations()
    assert not violations, "Ragas thresholds not met: " + "; ".join(violations)


def test_metrics_computed_for_most_samples(evaluation: EvaluationReport) -> None:
    n = len(evaluation.per_sample)
    for metric, failed in evaluation.failed_metrics.items():
        assert failed <= MAX_FAILED_SHARE * n, f"{metric}: {failed}/{n} evaluations failed (LLM judge)"


def test_reference_articles_are_retrieved(evaluation: EvaluationReport) -> None:
    """Deterministic retrieval check: the reference provision is in the context."""
    rate = evaluation.scores["reference_hit_rate"]
    assert rate >= MIN_REFERENCE_HIT_RATE, f"reference_hit_rate = {rate:.2f} < {MIN_REFERENCE_HIT_RATE}"


def test_no_hallucinated_article_numbers(evaluation: EvaluationReport) -> None:
    """No answer cites an article/annex that does not exist in the corpus."""
    offenders = {s["id"]: s["unknown_references"] for s in evaluation.per_sample if s["unknown_references"]}
    assert not offenders, f"Non-existent provisions cited: {offenders}"
