from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.config import Settings
from src.core.exceptions import CorpusIntegrityError
from src.models.schemas import LegalChunk, Regulation, RiskCategory, UnitType
from src.rag import preprocess_laws as pl
from tests.conftest import ROOT, find_chunk


def _lines(*texts: str) -> list[pl._Line]:
    return [pl._Line(t, 1) for t in texts]


class TestHelpers:
    @pytest.mark.parametrize(("roman", "value"), [("I", 1), ("IV", 4), ("IX", 9), ("XIII", 13), ("XL", 40)])
    def test_roman_to_int(self, roman: str, value: int) -> None:
        assert pl.roman_to_int(roman) == value

    @pytest.mark.parametrize("bad", ["", "ABC", "12"])
    def test_roman_to_int_rejects_invalid_input(self, bad: str) -> None:
        with pytest.raises(ValueError, match="roman"):
            pl.roman_to_int(bad)

    def test_reflow_rebuilds_paragraphs_and_points(self) -> None:
        text = pl.reflow(
            [
                "1.",
                "The provider shall",
                "ensure that:",
                "(a) the system is",
                "safe;",
                "(b) logs are kept.",
                "2.",
                "Done.",
            ]
        )
        assert text.split("\n") == [
            "1. The provider shall ensure that:",
            "(a) the system is safe;",
            "(b) logs are kept.",
            "2. Done.",
        ]

    def test_reflow_joins_hyphenated_words(self) -> None:
        assert pl.reflow(["general-", "purpose model"]) == "general-purpose model"

    def test_clean_lines_removes_page_furniture_and_soft_hyphens(self) -> None:
        raw = _lines(
            "02016R0679 — FR — 04.05.2016 — 000.002 — 31", "▼B", "concer­", "nant les ►C1 personnes ◄"
        )
        assert [ln.text for ln in pl.clean_lines(raw, pl.GDPR_SPEC)] == ["concernant les personnes"]

    def test_sentence_case_keeps_acronyms(self) -> None:
        assert (
            pl._sentence_case("EU DATABASE FOR HIGH-RISK AI SYSTEMS")
            == "EU database for high-risk AI systems"
        )


class TestStateMachine:
    def test_sequential_numbering_rejects_inline_references(self) -> None:
        lines = _lines(
            "HAVE ADOPTED THIS REGULATION:",
            "CHAPTER I",
            "GENERAL PROVISIONS",
            "Article 1",
            "Subject matter",
            "This Regulation refers to",
            "Article 6",  # cross-reference at the start of a line: must NOT open an article
            "of another act.",
            "Article 2",
            "Scope",
            "1.",
            "It applies.",
            "Done at Brussels, 13 June 2024.",
            "For the European Parliament",
            "ANNEX I",
            "List of legislation",
            "1. Directive X.",
        )
        units = pl.parse_units(lines, pl.AI_ACT_SPEC)
        assert [(u.unit_type, u.number) for u in units] == [
            (UnitType.ARTICLE, "1"),
            (UnitType.ARTICLE, "2"),
            (UnitType.ANNEX, "I"),
        ]
        assert "Article 6 of another act." in units[0].body
        assert "Brussels" not in units[1].body  # signature block removed
        assert units[0].chapter_title == "General provisions"
        assert units[2].chapter_number is None

    def test_missing_body_marker_raises(self) -> None:
        with pytest.raises(CorpusIntegrityError, match="not found"):
            pl.parse_units(_lines("Article 1", "Title"), pl.AI_ACT_SPEC)

    def test_long_units_are_split_on_paragraph_boundaries(self) -> None:
        unit = pl.LegalUnit(
            source=Regulation.GDPR,
            unit_type=UnitType.ARTICLE,
            number="35",
            title="Analyse d'impact",
            body="\n".join(f"{i}. " + "mot " * 120 for i in range(1, 9)),
            page_start=30,
            chapter_number="IV",
            chapter_title="Responsable du traitement",
        )
        chunks = pl.unit_to_chunks(unit, pl.GDPR_SPEC, max_chars=1200)
        assert len(chunks) > 1
        assert all(c.metadata.total_parts == len(chunks) for c in chunks)
        assert all(c.text.startswith("GDPR - Article 35 - Analyse d'impact (part") for c in chunks)
        assert [c.metadata.part for c in chunks] == list(range(1, len(chunks) + 1))
        body_lines = [ln for c in chunks for ln in c.text.split("\n\n", 1)[1].split("\n")]
        assert all(re.match(r"^\d\. ", ln) for ln in body_lines)  # no paragraph cut in half

    def test_unit_to_chunks_validates_inputs(self) -> None:
        unit = pl.LegalUnit(
            source=Regulation.GDPR, unit_type=UnitType.ARTICLE, number="1", title="t", body="b", page_start=1
        )
        with pytest.raises(ValidationError):
            pl.unit_to_chunks(unit, pl.GDPR_SPEC, max_chars=0)

    def test_legal_unit_rejects_bad_number(self) -> None:
        with pytest.raises(ValidationError):
            pl.LegalUnit(
                source=Regulation.GDPR,
                unit_type=UnitType.ARTICLE,
                number="1a",
                title="",
                body="",
                page_start=1,
            )


class TestRealCorpus:
    def test_all_articles_and_annexes_present(self, corpus: list[LegalChunk]) -> None:
        pl.validate_corpus(corpus)
        ai_articles = {c.metadata.article_number for c in corpus if c.metadata.source is Regulation.AI_ACT}
        assert set(range(1, 114)) <= ai_articles

    def test_validate_corpus_detects_missing_articles(self, corpus: list[LegalChunk]) -> None:
        with pytest.raises(CorpusIntegrityError, match="missing articles"):
            pl.validate_corpus([c for c in corpus if c.metadata.article_number != 42])

    def test_chunk_ids_are_unique_and_point_ids_deterministic(self, corpus: list[LegalChunk]) -> None:
        assert len({c.chunk_id for c in corpus}) == len(corpus)
        assert corpus[0].point_id == LegalChunk.model_validate(corpus[0].model_dump()).point_id

    def test_chunks_respect_size_budget(self, corpus: list[LegalChunk]) -> None:
        assert max(len(c.text) for c in corpus) <= Settings().max_chunk_chars + 400  # + header

    @pytest.mark.parametrize(
        ("chunk_id", "risk", "when", "topic"),
        [
            ("ai_act:article:5:p1", RiskCategory.PROHIBITED, date(2025, 2, 2), "prohibited_practices"),
            ("ai_act:article:14:p1", RiskCategory.HIGH_RISK, date(2027, 12, 2), "human_oversight"),
            ("ai_act:article:50:p1", RiskCategory.LIMITED_RISK, date(2026, 8, 2), "transparency"),
            ("ai_act:article:53:p1", RiskCategory.GPAI, date(2025, 8, 2), "gpai"),
            ("ai_act:article:31:p1", RiskCategory.HIGH_RISK, date(2025, 8, 2), None),
            ("ai_act:annex:III:p1", RiskCategory.HIGH_RISK, date(2027, 12, 2), "high_risk_classification"),
            ("ai_act:annex:I:p1", RiskCategory.HIGH_RISK, date(2028, 8, 2), "high_risk_classification"),
            ("gdpr:article:35:p1", RiskCategory.NOT_APPLICABLE, date(2018, 5, 25), "dpia"),
            ("gdpr:article:6:p1", RiskCategory.NOT_APPLICABLE, date(2018, 5, 25), "legal_basis"),
        ],
    )
    def test_metadata_enrichment(
        self, corpus: list[LegalChunk], chunk_id: str, risk: RiskCategory, when: date, topic: str | None
    ) -> None:
        m = find_chunk(corpus, chunk_id).metadata
        assert m.risk_category is risk
        assert m.enforcement_date == when
        if topic:
            assert topic in m.topics

    def test_structure_metadata(self, corpus: list[LegalChunk]) -> None:
        art9 = find_chunk(corpus, "ai_act:article:9:p1").metadata
        assert (art9.chapter_number, art9.section_number) == ("III", "2")
        assert art9.title == "Risk management system"
        gdpr1 = find_chunk(corpus, "gdpr:article:1:p1")
        assert gdpr1.metadata.title == "Objet et objectifs"  # title of the official French text
        assert gdpr1.metadata.citation == "GDPR, Art. 1"

    def test_extraction_artifacts_are_fixed(self, corpus: list[LegalChunk]) -> None:
        assert "greater than 10^25" in find_chunk(corpus, "ai_act:article:51:p1").text
        assert not any("­" in c.text or "►" in c.text or "◄" in c.text for c in corpus)
        assert "72 heures au plus tard" in find_chunk(corpus, "gdpr:article:33:p1").text

    def test_pdf_preprocessing_matches_committed_corpus(self, tmp_path: Path) -> None:
        """The committed JSONL must be exactly reproducible from the official PDFs."""
        out = tmp_path / "chunks.jsonl"
        assert pl.main(["--out", str(out)]) == 0
        assert out.read_text() == (ROOT / "data/processed/legal_chunks.jsonl").read_text()

    def test_preprocess_regulation_requires_existing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            pl.preprocess_regulation(tmp_path / "missing.pdf", pl.AI_ACT_SPEC, 1000)

    def test_jsonl_round_trip(self, corpus: list[LegalChunk], tmp_path: Path) -> None:
        path = tmp_path / "rt.jsonl"
        assert pl.write_jsonl(corpus[:5], path) == 5
        assert pl.read_jsonl(path) == corpus[:5]

    def test_corpus_stats(self, corpus: list[LegalChunk]) -> None:
        stats = pl.corpus_stats(corpus)
        assert stats.chunks == len(corpus)
        assert set(stats.by_source) == {"ai_act", "gdpr"}
