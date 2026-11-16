"""Settings validation and the shared input-validation decorator."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from src.core.config import Settings
from src.core.types import NonEmptyStr, PositiveInt, validated


@validated
def _repeat(text: NonEmptyStr, times: PositiveInt) -> str:
    return text * times


def test_validated_decorator_checks_and_coerces_arguments() -> None:
    assert _repeat("  ab ", "2") == "abab"  # type: ignore[arg-type]  # stripped and coerced
    with pytest.raises(ValidationError):
        _repeat("", 2)
    with pytest.raises(ValidationError):
        _repeat("a", -1)


def test_settings_validation() -> None:
    s = Settings.model_validate({"qdrant_url": "", "qdrant_api_key": "", "log_level": "debug"})
    assert s.qdrant_url is None
    assert s.qdrant_api_key is None
    assert s.log_level == "DEBUG"
    with pytest.raises(ValidationError):
        Settings(_env_file=None, rerank_top_n=30, rerank_candidates=10)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ollama_base_url="localhost:11434")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, qdrant_collection="bad name!")


def test_api_key_is_secret() -> None:
    s = Settings(_env_file=None, qdrant_api_key=SecretStr("s3cr3t"))
    assert "s3cr3t" not in repr(s)
    assert s.qdrant_api_key is not None
    assert s.qdrant_api_key.get_secret_value() == "s3cr3t"


@pytest.mark.parametrize(
    ("raw", "expected"), [(0.8, 0.8), (3, 0.3), (85, 0.85), ("85%", 0.85), (100, 1.0), (250, 1.0)]
)
def test_llm_confidence_scales_are_normalised(raw: object, expected: float) -> None:
    from src.models.schemas import AIActClassificationDraft

    draft = AIActClassificationDraft.model_validate(
        {"is_ai_system": True, "risk_category": "minimal_risk", "justification": "x", "confidence": raw}
    )
    assert draft.confidence == pytest.approx(expected)


def test_llm_enum_spellings_are_normalised() -> None:
    from src.models.schemas import ComplianceStatus, RiskCategory

    assert RiskCategory("High-risk") is RiskCategory.HIGH_RISK
    assert ComplianceStatus("Partially compliant") is ComplianceStatus.PARTIALLY_COMPLIANT
    with pytest.raises(ValueError):
        RiskCategory("dangerous")
