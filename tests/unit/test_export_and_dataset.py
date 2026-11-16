from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest
from pydantic import ValidationError

from src.core.config import Settings
from src.evaluation.dataset import EvalSample, load_samples, load_testset, sample_chunks, write_samples
from src.export.pdf_generator import _clean, export_json, generate_pdf
from src.models.schemas import AuditReport, LegalChunk, Regulation, RiskCategory
from tests.conftest import ROOT, sample_report


@pytest.fixture()
def report() -> AuditReport:
    return sample_report()


def test_generate_pdf_produces_valid_document(report: AuditReport) -> None:
    pdf = generate_pdf(report, {"gdpr-dpia": True})
    assert pdf.startswith(b"%PDF")
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:  # type: ignore[no-untyped-call]
        text = "\n".join(page.get_text() for page in doc)
        assert doc.page_count >= 3
    assert "HireMatch" in text
    assert "Prioritised action plan" in text
    assert "[X]" in text  # ticked box reflected
    assert "02/12/2027" in text  # Omnibus Annex III deadline


def test_generate_pdf_rejects_invalid_input() -> None:
    with pytest.raises(ValidationError):
        generate_pdf({"not": "a report"})  # type: ignore[arg-type]


def test_clean_escapes_markup_and_non_latin1() -> None:
    assert _clean("<b>œ ≥ “x”</b>") == '&lt;b&gt;oe &gt;= "x"&lt;/b&gt;'


def test_export_json_round_trips(report: AuditReport) -> None:
    data = json.loads(export_json(report, {"gdpr-dpia": True}))
    assert data["checklist"][0]["done"] is True
    assert data["compliance_level"] == "Low"
    restored = AuditReport.model_validate_json(export_json(report))  # computed fields tolerated
    assert restored.ai_act.risk_category is RiskCategory.HIGH_RISK
    assert restored == report


class TestDatasets:
    def test_golden_set_references_exist_in_corpus(self, corpus: list[LegalChunk]) -> None:
        samples = load_samples(ROOT / "tests/eval/data/golden_testset.jsonl")
        assert len(samples) >= 12
        ids = {c.chunk_id for c in corpus}
        for s in samples:
            assert s.reference_chunk_ids, s.id
            assert set(s.reference_chunk_ids) <= ids, s.id
        assert {s.regulation for s in samples} == set(Regulation)

    def test_sample_chunks_is_stratified_and_reproducible(self, corpus: list[LegalChunk]) -> None:
        first = sample_chunks(corpus, per_source=5, seed=7)
        assert [c.chunk_id for c in first] == [
            c.chunk_id for c in sample_chunks(corpus, per_source=5, seed=7)
        ]
        assert sum(c.metadata.source is Regulation.AI_ACT for c in first) == 5
        assert all(c.metadata.topics for c in first)
        with pytest.raises(ValidationError):
            sample_chunks(corpus, per_source=0, seed=7)

    def test_load_testset_interleaves_regulations(self, tmp_path: Path) -> None:
        golden = tmp_path / "g.jsonl"
        samples = [
            EvalSample(id=f"a{i}", regulation=Regulation.AI_ACT, user_input="q?", reference="r")
            for i in range(3)
        ]
        samples.append(EvalSample(id="r0", regulation=Regulation.GDPR, user_input="q?", reference="r"))
        write_samples(samples, golden)
        s = Settings(
            _env_file=None,
            ragas_testset_path=golden,
            ragas_synthetic_testset_path=tmp_path / "absent.jsonl",
        )
        assert [x.id for x in load_testset(s)] == ["a0", "r0", "a1", "a2"]
        assert [x.id for x in load_testset(s, max_samples=2)] == ["a0", "r0"]

    def test_eval_sample_rejects_blank_question(self) -> None:
        with pytest.raises(ValidationError):
            EvalSample(id="x", regulation=Regulation.GDPR, user_input=" ", reference="r")
