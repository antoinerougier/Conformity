"""Structured ingestion of project documents (.pdf, .pptx) into Markdown.

- PDF : ``pymupdf4llm`` (headings, lists, Markdown tables), page by page.
- PPTX: ``python-pptx`` - titles, text frames with bullet levels, tables, shape groups
  (recursive), charts (title/series) and speaker notes.

:func:`condense_markdown` produces a size-bounded view, fairly distributed across sections,
so the document fits the local LLM context window without any extra LLM call.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import FilePath

from src.core.exceptions import UnsupportedDocumentError
from src.core.logging import get_logger
from src.core.types import PositiveInt, validated
from src.models.schemas import ProjectDocument, ProjectSection

logger = get_logger(__name__)

__all__ = ["UnsupportedDocumentError", "condense_markdown", "count_markdown_tables", "parse_document"]

SUPPORTED_EXTENSIONS: dict[str, Literal["pdf", "pptx"]] = {".pdf": "pdf", ".pptx": "pptx"}
_MAGIC_BYTES: dict[str, bytes] = {"pdf": b"%PDF", "pptx": b"PK\x03\x04"}
_MD_TABLE_SEPARATOR_RE = re.compile(r"^\|(\s*:?-{3,}:?\s*\|)+\s*$")
_BYTES_PER_MB = 1024 * 1024


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_signature(path: Path, file_type: str) -> None:
    """Reject files whose content does not match their extension (spoofed uploads)."""
    with path.open("rb") as fh:
        head = fh.read(4)
    if not head.startswith(_MAGIC_BYTES[file_type]):
        raise UnsupportedDocumentError(f"Invalid file signature for a .{file_type}: {path.name}")


@validated
def parse_document(path: FilePath, *, max_size_mb: PositiveInt = 50) -> ProjectDocument:
    """Entry point: detect the type and return a :class:`ProjectDocument`."""
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(f"Unsupported extension: {ext} (expected .pdf or .pptx)")
    size_mb = path.stat().st_size / _BYTES_PER_MB
    if size_mb > max_size_mb:
        raise UnsupportedDocumentError(f"File too large ({size_mb:.1f} MB > {max_size_mb} MB)")
    file_type = SUPPORTED_EXTENSIONS[ext]
    _check_signature(path, file_type)

    title, sections = _parse_pdf(path) if file_type == "pdf" else _parse_pptx(path)
    sections = [s for s in sections if s.markdown.strip() or (s.notes or "").strip()]
    if not sections:
        raise UnsupportedDocumentError("No usable text (scanned document? OCR is required first).")
    logger.info("Document '%s' parsed: %d sections", path.name, len(sections))
    return ProjectDocument(
        file_name=path.name, file_type=file_type, title=title, sections=sections, sha256=sha256_of(path)
    )


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def count_markdown_tables(markdown: str) -> int:
    """Number of Markdown tables (one ``|---|`` separator line per table)."""
    return sum(1 for line in markdown.splitlines() if _MD_TABLE_SEPARATOR_RE.match(line.strip()))


def _parse_pdf(path: Path) -> tuple[str | None, list[ProjectSection]]:
    import pymupdf
    import pymupdf4llm

    with pymupdf.open(path) as doc:  # type: ignore[no-untyped-call]
        meta_title = (doc.metadata or {}).get("title") or None
    try:
        # No OCR: text-based documents only (scanned files are rejected with a clear message).
        pages: list[dict[str, Any]] = pymupdf4llm.to_markdown(
            str(path), page_chunks=True, show_progress=False, use_ocr=False
        )
    except TypeError:  # older pymupdf4llm without the layout engine / OCR option
        pages = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)
    sections: list[ProjectSection] = []
    for index, page in enumerate(pages, start=1):
        md = str(page.get("text", "")).strip()
        heading = next((ln.lstrip("#").strip() for ln in md.splitlines() if ln.startswith("#")), None)
        sections.append(
            ProjectSection(
                index=index, kind="page", title=heading, markdown=md, table_count=count_markdown_tables(md)
            )
        )
    title = meta_title or next((s.title for s in sections if s.title), None)
    return title, sections


# ---------------------------------------------------------------------------
# PPTX
# ---------------------------------------------------------------------------


def _md_escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>").strip()


def _table_to_markdown(table: Any) -> str:
    rows = [[_md_escape_cell(cell.text) for cell in row.cells] for row in table.rows]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _text_frame_to_markdown(text_frame: Any) -> str:
    lines = []
    for paragraph in text_frame.paragraphs:
        text = "".join(run.text for run in paragraph.runs).strip()
        if text:
            level = int(getattr(paragraph, "level", 0) or 0)
            lines.append(f"{'  ' * level}- {text}")
    return "\n".join(lines)


def _chart_to_markdown(chart: Any) -> str:
    parts = ["**Chart**"]
    try:
        if chart.has_title and chart.chart_title.has_text_frame:
            parts[0] += f": {chart.chart_title.text_frame.text.strip()}"
        categories = [str(c) for c in chart.plots[0].categories] if chart.plots else []
        if categories:
            parts.append(f"Categories: {', '.join(categories)}")
        for plot in chart.plots:
            for series in plot.series:
                parts.append(f"Series '{series.name}': {', '.join(str(v) for v in series.values)}")
    except (AttributeError, IndexError, KeyError, ValueError) as exc:
        logger.debug("Chart only partially readable: %s", exc)
    return "\n".join(parts)


def _iter_shapes(shapes: Any) -> Iterator[Any]:
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from _iter_shapes(shape.shapes)
        else:
            yield shape


def _parse_pptx(path: Path) -> tuple[str | None, list[ProjectSection]]:
    from pptx import Presentation

    prs = Presentation(str(path))
    core_title = (prs.core_properties.title or "").strip() or None
    sections: list[ProjectSection] = []
    for index, slide in enumerate(prs.slides, start=1):
        title_shape = slide.shapes.title
        title = title_shape.text.strip() if title_shape is not None and title_shape.has_text_frame else None
        blocks: list[str] = []
        tables = 0
        # Reading order: top -> bottom, left -> right.
        for shape in sorted(_iter_shapes(slide.shapes), key=lambda s: (s.top or 0, s.left or 0)):
            if title_shape is not None and shape.shape_id == title_shape.shape_id:
                continue
            if getattr(shape, "has_table", False):
                blocks.append(_table_to_markdown(shape.table))
                tables += 1
            elif getattr(shape, "has_chart", False):
                blocks.append(_chart_to_markdown(shape.chart))
            elif getattr(shape, "has_text_frame", False) and (
                md := _text_frame_to_markdown(shape.text_frame)
            ):
                blocks.append(md)
        notes = None
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text.strip() or None
        sections.append(
            ProjectSection(
                index=index,
                kind="slide",
                title=title,
                markdown="\n\n".join(b for b in blocks if b.strip()),
                notes=notes,
                table_count=tables,
            )
        )
    title = core_title or next((s.title for s in sections if s.title), None)
    return title, sections


# ---------------------------------------------------------------------------
# Condensation for the LLM context
# ---------------------------------------------------------------------------


@validated
def condense_markdown(document: ProjectDocument, max_chars: PositiveInt) -> str:
    """Bounded Markdown view: fair budget split, short sections kept in full."""
    full = document.markdown
    if len(full) <= max_chars:
        return full
    blocks: list[str] = []
    for s in document.sections:
        header = f"## {s.kind.capitalize()} {s.index}" + (f" - {s.title}" if s.title else "")
        body = s.markdown.strip() + (f"\n\n> Notes: {s.notes.strip()}" if s.notes else "")
        blocks.append(f"{header}\n\n{body}")
    # Fair max-min allocation of the budget (water-filling).
    remaining = max(max_chars - 200, 0)
    sizes = [len(b) for b in blocks]
    alloc = [0] * len(blocks)
    pending = sorted(range(len(blocks)), key=lambda i: sizes[i])
    while pending:
        share = remaining // len(pending)
        i = pending[0]
        if sizes[i] <= share:
            alloc[i] = sizes[i]
            remaining -= sizes[i]
            pending.pop(0)
        else:
            for j in pending:
                alloc[j] = share
            break
    out = [
        block if len(block) <= budget else block[: max(budget - 20, 0)].rstrip() + " [...]"
        for block, budget in zip(blocks, alloc, strict=True)
    ]
    return f"> Condensed document ({len(full)} -> ~{max_chars} characters)\n\n" + "\n\n".join(out)
