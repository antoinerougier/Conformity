from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest
from pydantic import ValidationError

from src.core.exceptions import UnsupportedDocumentError
from src.ingestion.parser import condense_markdown, count_markdown_tables, parse_document
from src.models.schemas import ProjectDocument, ProjectSection
from tests.conftest import build_pptx


def _pdf(path: Path, pages: list[str]) -> Path:
    doc = pymupdf.open()  # type: ignore[no-untyped-call]
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    doc.save(str(path))  # type: ignore[no-untyped-call]
    doc.close()  # type: ignore[no-untyped-call]
    return path


def test_parse_pptx_extracts_titles_bullets_tables_and_notes(tmp_path: Path) -> None:
    path = build_pptx(
        tmp_path / "project.pptx",
        [
            {
                "title": "HireMatch - CV screening",
                "bullets": [(0, "Automatic ranking of candidates"), (1, "Scoring model")],
                "notes": "Data kept for 5 years",
            },
            {
                "title": "Data",
                "bullets": [(0, "CVs, cover letters")],
                "table": [["Data", "Source"], ["Name", "Candidate"], ["Degrees", "CV"]],
            },
        ],
    )
    doc = parse_document(path)
    assert doc.file_type == "pptx"
    assert doc.title == "HireMatch - CV screening"
    assert len(doc.sections) == 2
    s1, s2 = doc.sections
    assert "- Automatic ranking of candidates" in s1.markdown
    assert "  - Scoring model" in s1.markdown  # bullet level preserved
    assert s1.notes == "Data kept for 5 years"
    assert s2.table_count == 1
    assert "| Data | Source |" in s2.markdown
    assert "| --- | --- |" in s2.markdown
    assert "> Notes: Data kept for 5 years" in doc.markdown
    assert len(doc.sha256) == 64


def test_parse_pdf_page_by_page(tmp_path: Path) -> None:
    doc = parse_document(
        _pdf(tmp_path / "project.pdf", ["Project ScoreCredit", "AI-based credit risk analysis"])
    )
    assert doc.file_type == "pdf"
    assert [s.kind for s in doc.sections] == ["page", "page"]
    assert "ScoreCredit" in doc.sections[0].markdown
    assert "credit risk" in doc.markdown


def test_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "notes.docx"
    path.write_bytes(b"PK\x03\x04data")
    with pytest.raises(UnsupportedDocumentError, match="extension"):
        parse_document(path)


def test_rejects_spoofed_signature(tmp_path: Path) -> None:
    path = tmp_path / "fake.pdf"
    path.write_bytes(b"<html>not a pdf</html>")
    with pytest.raises(UnsupportedDocumentError, match="signature"):
        parse_document(path)


def test_rejects_missing_file_and_invalid_size_limit(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        parse_document(tmp_path / "missing.pdf")
    path = _pdf(tmp_path / "big.pdf", ["x"])
    with pytest.raises(ValidationError):
        parse_document(path, max_size_mb=0)


def test_rejects_empty_document(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedDocumentError, match="No usable text"):
        parse_document(_pdf(tmp_path / "blank.pdf", [""]))


def test_count_markdown_tables() -> None:
    assert count_markdown_tables("| a | b |\n|---|---|\n| 1 | 2 |\n\ntext\n| c |\n| :---: |") == 2


def test_condense_markdown_respects_budget_and_keeps_every_section() -> None:
    sections = [
        ProjectSection(index=1, kind="slide", title="Short", markdown="tiny text"),
        ProjectSection(index=2, kind="slide", title="Long", markdown="x" * 20_000),
        ProjectSection(index=3, kind="slide", title="Medium", markdown="y" * 3_000),
    ]
    doc = ProjectDocument(file_name="d.pptx", file_type="pptx", sections=sections, sha256="0" * 64)
    out = condense_markdown(doc, 6_000)
    assert len(out) <= 6_300
    assert "tiny text" in out  # short sections kept in full
    assert "## Slide 2 - Long" in out
    assert "## Slide 3 - Medium" in out
    assert condense_markdown(doc, 100_000) == doc.markdown


def test_project_document_requires_sections_and_valid_hash() -> None:
    with pytest.raises(ValidationError):
        ProjectDocument(file_name="d.pdf", file_type="pdf", sections=[], sha256="0" * 64)
    with pytest.raises(ValidationError):
        ProjectDocument(
            file_name="d.pdf",
            file_type="pdf",
            sections=[ProjectSection(index=1, kind="page", markdown="x")],
            sha256="not-a-hash",
        )
