"""Legal preprocessing: structure-aware splitting of the AI Act and the GDPR.

Splitting follows the **official structure** of the regulations (chapter -> section ->
article, then annexes) rather than an arbitrary character count. Very long articles (e.g.
Art. 3 "Definitions") are sub-split *on numbered-paragraph boundaries*; every sub-chunk
keeps the article header and the same metadata (``part/total_parts``).

Each chunk is enriched with strict metadata that can be filtered in Qdrant:
``source``, ``article_number``/``annex_number``, ``chapter_*``, ``section_*``,
``risk_category``, ``enforcement_date`` (after Omnibus 2026/1744), ``topics``.

Usage:
    python -m src.rag.preprocess_laws                       # writes data/processed/legal_chunks.jsonl
    python -m src.rag.preprocess_laws --index               # + Qdrant indexing
    python -m src.rag.preprocess_laws --index --if-empty    # index only if the collection is empty
    python -m src.rag.preprocess_laws --index --recreate    # full re-indexing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

import pymupdf
from pydantic import BaseModel, ConfigDict, Field, FilePath

from src.compliance.timeline import (
    GDPR_APPLICATION,
    ai_act_annex_rules,
    ai_act_article_enforcement,
    ai_act_article_risk,
)
from src.core.config import Settings, get_settings
from src.core.exceptions import CorpusIntegrityError
from src.core.logging import configure_logging, get_logger
from src.core.types import PositiveInt, validated
from src.models.schemas import (
    LegalChunk,
    LegalChunkMetadata,
    Regulation,
    RiskCategory,
    UnitType,
)

logger = get_logger(__name__)

# =============================================================================
# Regulation specifications
# =============================================================================


@dataclass(frozen=True)
class LawSpec:
    """Typographic conventions of a regulation as published on EUR-Lex.

    The regexes match the *source* document, hence the French keywords for the GDPR PDF.
    """

    source: Regulation
    title: str
    celex: str
    language: Literal["en", "fr"]
    chapter_re: re.Pattern[str]
    section_re: re.Pattern[str]
    article_re: re.Pattern[str]
    annex_re: re.Pattern[str] | None
    body_start_re: re.Pattern[str]
    body_start_inclusive: bool
    noise_line_res: tuple[re.Pattern[str], ...]
    inline_noise_res: tuple[re.Pattern[str], ...] = ()
    signature_start_re: re.Pattern[str] | None = None
    #: Targeted fixes of extraction artefacts (e.g. lost superscripts: 10^25 -> "1025").
    text_fixes: tuple[tuple[re.Pattern[str], str], ...] = ()


AI_ACT_SPEC = LawSpec(
    source=Regulation.AI_ACT,
    title="Regulation (EU) 2024/1689 (Artificial Intelligence Act)",
    celex="32024R1689",
    language="en",
    chapter_re=re.compile(r"^CHAPTER ([IVXLC]+)$"),
    section_re=re.compile(r"^SECTION (\d+)$"),
    article_re=re.compile(r"^Article (\d+)$"),
    annex_re=re.compile(r"^ANNEX ([IVXLC]+)$"),
    body_start_re=re.compile(r"^HAVE ADOPTED THIS REGULATION:?$"),
    body_start_inclusive=False,
    noise_line_res=(
        re.compile(r"^EN$"),
        re.compile(r"^OJ L, \d{1,2}\.\d{1,2}\.\d{4}$"),
        re.compile(r"^ELI: http://data\.europa\.eu/eli/reg/2024/1689/oj$"),
        re.compile(r"^\d{1,3}/144$"),
    ),
    signature_start_re=re.compile(r"^Done at Brussels"),
    text_fixes=((re.compile(r"(floating point operations is greater than )1025"), r"\g<1>10^25"),),
)

GDPR_SPEC = LawSpec(
    source=Regulation.GDPR,
    title="Regulation (EU) 2016/679 (General Data Protection Regulation)",
    celex="32016R0679",
    language="fr",
    chapter_re=re.compile(r"^CHAPITRE ([IVXLC]+)$"),
    section_re=re.compile(r"^Section (\d+)$"),
    article_re=re.compile(r"^Article (\d+|premier)$"),
    annex_re=None,
    body_start_re=re.compile(r"^CHAPITRE I$"),
    body_start_inclusive=True,
    noise_line_res=(
        re.compile(r"^02016R0679 — FR — .*$"),
        re.compile(r"^[▼►◄][A-Z]?\d*$"),
    ),
    inline_noise_res=(re.compile(r"►[A-Z]\d*\s?"), re.compile(r"\s?◄")),
)

SPECS: dict[Regulation, LawSpec] = {Regulation.AI_ACT: AI_ACT_SPEC, Regulation.GDPR: GDPR_SPEC}

# =============================================================================
# Topics (``topics`` metadata)
# =============================================================================


def _expand(mapping: dict[tuple[int, ...] | range, tuple[str, ...]]) -> dict[int, tuple[str, ...]]:
    out: dict[int, tuple[str, ...]] = {}
    for keys, topics in mapping.items():
        for k in keys:
            out[k] = tuple(sorted(set(out.get(k, ())) | set(topics)))
    return out


GDPR_TOPICS = _expand(
    {
        (4,): ("definitions",),
        (5,): ("principles", "data_minimisation", "storage_limitation", "purpose_limitation"),
        (6,): ("legal_basis",),
        (7, 8): ("consent", "legal_basis"),
        (9,): ("special_categories",),
        (10,): ("criminal_data",),
        range(12, 15): ("transparency",),
        range(15, 22): ("data_subject_rights",),
        (22,): ("automated_decision_making", "profiling", "data_subject_rights"),
        (24,): ("accountability",),
        (25,): ("privacy_by_design",),
        (26,): ("joint_controllers",),
        (28, 29): ("processors",),
        (30,): ("records_of_processing",),
        (32,): ("security",),
        (33, 34): ("data_breach", "security"),
        (35, 36): ("dpia",),
        range(37, 40): ("dpo",),
        range(40, 44): ("certification_codes",),
        range(44, 51): ("international_transfers",),
        range(77, 85): ("remedies_penalties",),
    }
)

AI_ACT_TOPICS = _expand(
    {
        (2,): ("scope",),
        (3,): ("definitions",),
        (4,): ("ai_literacy",),
        (5,): ("prohibited_practices",),
        (6, 7): ("high_risk_classification",),
        (8, 9): ("risk_management",),
        (10,): ("data_governance", "bias"),
        (11,): ("technical_documentation",),
        (12,): ("record_keeping",),
        (13,): ("transparency",),
        (14,): ("human_oversight",),
        (15,): ("robustness_cybersecurity",),
        range(16, 26): ("operator_obligations",),
        (26,): ("deployer_obligations", "human_oversight"),
        (27,): ("fundamental_rights_impact_assessment",),
        range(40, 50): ("conformity_registration",),
        (50,): ("transparency", "synthetic_content"),
        range(51, 57): ("gpai",),
        range(57, 64): ("innovation_sandbox",),
        (72, 73): ("post_market_monitoring",),
        (86,): ("right_to_explanation",),
        range(99, 102): ("penalties",),
    }
)

# =============================================================================
# Helpers
# =============================================================================

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
_PARA_NUM_RE = re.compile(r"^\d{1,2}\.$")
_ITEM_START_RE = re.compile(r"^(\(\s?[a-z0-9]{1,5}\s?\)|[a-z]\)|\d{1,2}\)|—|-\s|•)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:])\s+(?=[A-Z(À-Ý])")
_ACRONYMS_RE = re.compile(r"\b(ai|eu|union)\b", re.IGNORECASE)


def roman_to_int(value: str) -> int:
    """Convert a roman numeral (I..C range used by EU acts) to an integer."""
    if not value or any(ch not in _ROMAN for ch in value.upper()):
        raise ValueError(f"Invalid roman numeral: {value!r}")
    total, prev = 0, 0
    for ch in reversed(value.upper()):
        n = _ROMAN[ch]
        total = total - n if n < prev else total + n
        prev = max(prev, n)
    return total


def _sentence_case(text: str) -> str:
    """``GENERAL-PURPOSE AI MODELS`` -> ``General-purpose AI models``."""
    lowered = text.capitalize()
    return _ACRONYMS_RE.sub(lambda m: {"union": "Union"}.get(m.group(1).lower(), m.group(1).upper()), lowered)


def _is_upper_title(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


@dataclass(slots=True)
class _Line:
    """Raw text line with its page number (internal, hot path: plain dataclass)."""

    text: str
    page: int


def extract_lines(pdf_path: Path) -> list[_Line]:
    """Extract raw text line by line, keeping the (1-based) page number."""
    lines: list[_Line] = []
    with pymupdf.open(pdf_path) as doc:  # type: ignore[no-untyped-call]
        for page_index, page in enumerate(doc, start=1):
            text: str = page.get_text("text")
            lines.extend(_Line(raw.strip(), page_index) for raw in text.split("\n"))
    return lines


def clean_lines(lines: Sequence[_Line], spec: LawSpec) -> list[_Line]:
    """Remove page furniture and re-join words split by soft hyphens."""
    cleaned: list[_Line] = []
    pending: _Line | None = None
    for line in lines:
        text = line.text.replace(" ", " ").replace("`", "").strip()
        for pattern in spec.inline_noise_res:
            text = pattern.sub("", text)
        text = text.strip()
        if not text or any(p.match(text) for p in spec.noise_line_res):
            continue
        page = line.page
        if pending is not None:
            text, page = pending.text + text, pending.page
            pending = None
        if text.endswith("­"):  # soft hyphen at end of line
            pending = _Line(text[:-1], page)
            continue
        cleaned.append(_Line(text.replace("­", ""), page))
    if pending is not None:
        cleaned.append(pending)
    return cleaned


def reflow(lines: Iterable[str]) -> str:
    """Rebuild paragraphs from hard-wrapped lines (one output line = one paragraph/point)."""
    paragraphs: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if _PARA_NUM_RE.match(line):
            paragraphs.append(line)
            continue
        if paragraphs and _PARA_NUM_RE.match(paragraphs[-1]):
            paragraphs[-1] = f"{paragraphs[-1]} {line}"
            continue
        if not paragraphs or _ITEM_START_RE.match(line):
            paragraphs.append(line)
            continue
        prev = paragraphs[-1]
        if prev.endswith("-") and line[:1].islower():
            paragraphs[-1] = prev + line
        elif prev.endswith((".", ":", ";")) and line[:1].isupper():
            paragraphs.append(line)
        else:
            paragraphs[-1] = f"{prev} {line}"
    return "\n".join(paragraphs)


# =============================================================================
# Structural parsing
# =============================================================================


class LegalUnit(BaseModel):
    """Legal unit (article or annex) before final chunking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Regulation
    unit_type: UnitType
    number: str = Field(pattern=r"^(\d+|[IVXLC]+)$")
    title: str
    body: str
    page_start: int = Field(ge=1)
    chapter_number: str | None = None
    chapter_title: str | None = None
    section_number: str | None = None
    section_title: str | None = None

    @property
    def article_number(self) -> int | None:
        return int(self.number) if self.unit_type is UnitType.ARTICLE else None


@dataclass(slots=True)
class _ParserState:
    """Mutable state of the parsing state machine (internal)."""

    chapter: str | None = None
    chapter_title: str | None = None
    section: str | None = None
    section_title: str | None = None
    next_article: int = 1
    next_chapter: int = 1
    next_annex: int = 1
    in_annexes: bool = False
    in_signature: bool = False
    unit_type: UnitType | None = None
    unit_number: str = ""
    unit_title: str = ""
    unit_page: int = 1
    body: list[str] = field(default_factory=list)


def _is_heading(text: str, spec: LawSpec) -> bool:
    patterns = [spec.chapter_re, spec.section_re, spec.article_re]
    if spec.annex_re is not None:
        patterns.append(spec.annex_re)
    return any(p.match(text) for p in patterns)


def _read_title(lines: Sequence[_Line], start: int, spec: LawSpec, *, upper: bool) -> tuple[str, int]:
    """Read the title following a heading. Returns (title, index of the next line)."""
    if start >= len(lines):
        return "", start
    parts = [lines[start].text]
    i = start + 1
    while i < len(lines):
        nxt = lines[i].text
        if _is_heading(nxt, spec):
            break
        continuation = _is_upper_title(nxt) if upper else nxt[:1].islower()
        if not continuation or len(" ".join(parts)) > 250:
            break
        parts.append(nxt)
        i += 1
    return " ".join(parts).strip(), i


def _article_number(raw: str) -> int:
    return 1 if raw == "premier" else int(raw)


def parse_units(lines: Sequence[_Line], spec: LawSpec) -> list[LegalUnit]:
    """State machine: chapters -> sections -> articles -> annexes.

    Article/chapter/annex numbers must be **strictly sequential**, which neutralises false
    positives (e.g. a cross-reference "Article 6" that happens to start a line in the body).
    """
    start = next((i for i, ln in enumerate(lines) if spec.body_start_re.match(ln.text)), None)
    if start is None:
        raise CorpusIntegrityError(f"Start-of-enacting-terms marker not found for {spec.source.value}")
    if not spec.body_start_inclusive:
        start += 1

    units: list[LegalUnit] = []
    st = _ParserState()

    def close_unit() -> None:
        if st.unit_type is None:
            return
        body = reflow(st.body)
        for pattern, replacement in spec.text_fixes:
            body = pattern.sub(replacement, body)
        if body.strip():
            in_annex = st.unit_type is UnitType.ANNEX
            units.append(
                LegalUnit(
                    source=spec.source,
                    unit_type=st.unit_type,
                    number=st.unit_number,
                    title=st.unit_title,
                    body=body,
                    page_start=st.unit_page,
                    chapter_number=None if in_annex else st.chapter,
                    chapter_title=None if in_annex else st.chapter_title,
                    section_number=None if in_annex else st.section,
                    section_title=None if in_annex else st.section_title,
                )
            )
        st.unit_type, st.body = None, []

    i = start
    while i < len(lines):
        line = lines[i]
        text = line.text

        annex = spec.annex_re.match(text) if spec.annex_re is not None else None
        if annex is not None and roman_to_int(annex.group(1)) == st.next_annex:
            close_unit()
            st.in_annexes, st.in_signature = True, False
            st.next_annex += 1
            title, i = _read_title(lines, i + 1, spec, upper=False)
            st.unit_type, st.unit_number, st.unit_title = UnitType.ANNEX, annex.group(1), title
            st.unit_page = line.page
            continue

        chapter = None if st.in_annexes else spec.chapter_re.match(text)
        if chapter is not None and roman_to_int(chapter.group(1)) == st.next_chapter:
            close_unit()
            st.next_chapter += 1
            st.chapter = chapter.group(1)
            st.chapter_title, i = _read_title(lines, i + 1, spec, upper=True)
            if spec.language == "en":
                st.chapter_title = _sentence_case(st.chapter_title)
            st.section, st.section_title = None, None
            continue

        section = None if st.in_annexes else spec.section_re.match(text)
        expected_section = 1 if st.section is None else int(st.section) + 1
        if section is not None and int(section.group(1)) == expected_section:
            close_unit()
            st.section = section.group(1)
            st.section_title, i = _read_title(lines, i + 1, spec, upper=False)
            continue

        article = None if st.in_annexes else spec.article_re.match(text)
        if article is not None and _article_number(article.group(1)) == st.next_article:
            close_unit()
            st.next_article += 1
            title, i = _read_title(lines, i + 1, spec, upper=False)
            st.unit_type, st.unit_title = UnitType.ARTICLE, title
            st.unit_number = str(_article_number(article.group(1)))
            st.unit_page = line.page
            continue

        if spec.signature_start_re is not None and spec.signature_start_re.match(text):
            st.in_signature = True
        if st.unit_type is not None and not st.in_signature:
            st.body.append(text)
        i += 1

    close_unit()
    return units


# =============================================================================
# Final chunking and metadata
# =============================================================================


def _split_long(paragraphs: Sequence[str], max_chars: int) -> list[list[str]]:
    """Group paragraphs into blocks of at most ``max_chars`` (an over-long paragraph is
    split on sentence boundaries)."""
    expanded: list[str] = []
    for p in paragraphs:
        if len(p) <= max_chars:
            expanded.append(p)
            continue
        buf = ""
        for sentence in _SENTENCE_SPLIT_RE.split(p):
            if buf and len(buf) + len(sentence) + 1 > max_chars:
                expanded.append(buf)
                buf = sentence
            else:
                buf = f"{buf} {sentence}".strip()
        if buf:
            expanded.append(buf)

    groups: list[list[str]] = [[]]
    size = 0
    for p in expanded:
        if groups[-1] and size + len(p) + 1 > max_chars:
            groups.append([])
            size = 0
        groups[-1].append(p)
        size += len(p) + 1
    return [g for g in groups if g]


def _unit_metadata(unit: LegalUnit) -> tuple[RiskCategory, date, str | None, tuple[str, ...]]:
    if unit.source is Regulation.GDPR:
        return (
            RiskCategory.NOT_APPLICABLE,
            GDPR_APPLICATION,
            None,
            GDPR_TOPICS.get(unit.article_number or 0, ()),
        )
    if unit.unit_type is UnitType.ANNEX:
        risk, when, note = ai_act_annex_rules(unit.number)
        topics: tuple[str, ...] = ("high_risk_classification",) if unit.number in {"I", "III"} else ()
        return risk, when, note, topics
    article = unit.article_number or 1
    when, note = ai_act_article_enforcement(article, unit.chapter_number)
    return ai_act_article_risk(article, unit.chapter_number), when, note, AI_ACT_TOPICS.get(article, ())


@validated
def unit_to_chunks(unit: LegalUnit, spec: LawSpec, max_chars: PositiveInt) -> list[LegalChunk]:
    """Turn a legal unit into one or more chunks carrying the full header and metadata."""
    risk, when, note, topics = _unit_metadata(unit)
    label = "Annex" if unit.unit_type is UnitType.ANNEX else "Article"
    header = f"{unit.source.short} - {label} {unit.number} - {unit.title}"
    context_bits = []
    if unit.chapter_number:
        context_bits.append(f"Chapter {unit.chapter_number}: {unit.chapter_title}")
    if unit.section_number:
        context_bits.append(f"Section {unit.section_number}: {unit.section_title}")
    context = " | ".join(context_bits)

    groups = _split_long(unit.body.split("\n"), max_chars)
    total = len(groups)
    chunks: list[LegalChunk] = []
    for part, group in enumerate(groups, start=1):
        part_suffix = f" (part {part}/{total})" if total > 1 else ""
        text = header + part_suffix + ("\n" + context if context else "") + "\n\n" + "\n".join(group)
        metadata = LegalChunkMetadata(
            source=unit.source,
            regulation_title=spec.title,
            celex=spec.celex,
            language=spec.language,
            unit_type=unit.unit_type,
            article_number=unit.article_number,
            annex_number=unit.number if unit.unit_type is UnitType.ANNEX else None,
            title=unit.title,
            chapter_number=unit.chapter_number,
            chapter_title=unit.chapter_title,
            section_number=unit.section_number,
            section_title=unit.section_title,
            risk_category=risk,
            enforcement_date=when,
            enforcement_note=note,
            topics=topics,
            part=part,
            total_parts=total,
            page_start=unit.page_start,
        )
        chunk_id = f"{unit.source.value}:{unit.unit_type.value}:{unit.number}:p{part}"
        chunks.append(LegalChunk(chunk_id=chunk_id, text=text, metadata=metadata))
    return chunks


@validated
def preprocess_regulation(pdf_path: FilePath, spec: LawSpec, max_chars: PositiveInt) -> list[LegalChunk]:
    """Full pipeline for one official PDF: extract -> clean -> parse -> chunk."""
    lines = clean_lines(extract_lines(pdf_path), spec)
    units = parse_units(lines, spec)
    chunks = [c for u in units for c in unit_to_chunks(u, spec, max_chars)]
    n_art = sum(1 for u in units if u.unit_type is UnitType.ARTICLE)
    n_anx = sum(1 for u in units if u.unit_type is UnitType.ANNEX)
    logger.info("%s: %d articles, %d annexes -> %d chunks", spec.source.short, n_art, n_anx, len(chunks))
    return chunks


def preprocess_all(settings: Settings) -> list[LegalChunk]:
    """Preprocess both regulations and check the integrity of the resulting corpus."""
    for path in (settings.ai_act_path, settings.gdpr_path):
        if not path.is_file():
            raise FileNotFoundError(f"Official text not found: {path}")
    chunks = preprocess_regulation(settings.ai_act_path, AI_ACT_SPEC, settings.max_chunk_chars)
    chunks += preprocess_regulation(settings.gdpr_path, GDPR_SPEC, settings.max_chunk_chars)
    validate_corpus(chunks)
    return chunks


EXPECTED_ARTICLES = {Regulation.AI_ACT: 113, Regulation.GDPR: 99}
EXPECTED_ANNEXES = {Regulation.AI_ACT: 13, Regulation.GDPR: 0}


def validate_corpus(chunks: Sequence[LegalChunk]) -> None:
    """Integrity check: every expected article and annex is present, ids are unique."""
    for source, expected in EXPECTED_ARTICLES.items():
        found = {
            c.metadata.article_number
            for c in chunks
            if c.metadata.source is source and c.metadata.unit_type is UnitType.ARTICLE
        }
        missing = set(range(1, expected + 1)) - {n for n in found if n is not None}
        if missing:
            raise CorpusIntegrityError(f"{source.short}: missing articles {sorted(missing)}")
        annexes = {
            c.metadata.annex_number
            for c in chunks
            if c.metadata.source is source and c.metadata.unit_type is UnitType.ANNEX
        }
        if len(annexes) != EXPECTED_ANNEXES[source]:
            raise CorpusIntegrityError(
                f"{source.short}: {len(annexes)} annexes instead of {EXPECTED_ANNEXES[source]}"
            )
    ids = Counter(c.chunk_id for c in chunks)
    if dupes := [k for k, v in ids.items() if v > 1]:
        raise CorpusIntegrityError(f"Duplicate chunk ids: {dupes[:5]}")


# =============================================================================
# JSONL serialisation
# =============================================================================

_DUMP_EXCLUDE: dict[str, object] = {"point_id": True, "metadata": {"citation", "unit_key"}}


def write_jsonl(chunks: Iterable[LegalChunk], path: Path) -> int:
    """Write chunks as deterministic JSONL (sorted keys), returning the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for chunk in chunks:
            payload = chunk.model_dump(mode="json", exclude=_DUMP_EXCLUDE)  # type: ignore[arg-type]
            fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def iter_jsonl(path: Path) -> Iterator[LegalChunk]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield LegalChunk.model_validate_json(line)


def read_jsonl(path: Path) -> list[LegalChunk]:
    return list(iter_jsonl(path))


def load_or_build_corpus(settings: Settings) -> list[LegalChunk]:
    """Load the preprocessed corpus, or rebuild it from the official PDFs."""
    if settings.processed_chunks_path.is_file():
        return read_jsonl(settings.processed_chunks_path)
    chunks = preprocess_all(settings)
    write_jsonl(chunks, settings.processed_chunks_path)
    return chunks


class CorpusStats(BaseModel):
    chunks: int
    by_source: dict[str, int]
    by_risk_category: dict[str, int]
    by_enforcement_date: dict[str, int]
    max_chars: int


def corpus_stats(chunks: Sequence[LegalChunk]) -> CorpusStats:
    return CorpusStats(
        chunks=len(chunks),
        by_source=dict(sorted(Counter(c.metadata.source.value for c in chunks).items())),
        by_risk_category=dict(sorted(Counter(c.metadata.risk_category.value for c in chunks).items())),
        by_enforcement_date=dict(
            sorted(Counter(c.metadata.enforcement_date.isoformat() for c in chunks).items())
        ),
        max_chars=max((len(c.text) for c in chunks), default=0),
    )


# =============================================================================
# CLI
# =============================================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ai-act", type=Path, help="Official AI Act PDF (EN)")
    parser.add_argument("--gdpr", type=Path, help="Official GDPR PDF (FR)")
    parser.add_argument("--out", type=Path, help="Output JSONL file")
    parser.add_argument("--index", action="store_true", help="Index into Qdrant")
    parser.add_argument("--if-empty", action="store_true", help="Index only if the collection is empty")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate the collection")
    parser.add_argument("--stats", action="store_true", help="Print corpus statistics")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)
    overrides: dict[str, object] = {}
    if args.ai_act:
        overrides["regulations_dir"] = args.ai_act.parent
        overrides["ai_act_pdf"] = args.ai_act.name
    if args.gdpr:
        if args.ai_act and args.gdpr.parent != args.ai_act.parent:
            parser.error("--ai-act and --gdpr must be in the same directory")
        overrides["regulations_dir"] = args.gdpr.parent
        overrides["gdpr_pdf"] = args.gdpr.name
    if args.out:
        overrides["processed_chunks_path"] = args.out
    if overrides:
        settings = Settings.model_validate({**settings.model_dump(), **overrides})

    chunks = preprocess_all(settings)
    n = write_jsonl(chunks, settings.processed_chunks_path)
    logger.info("%d chunks written to %s", n, settings.processed_chunks_path)
    if args.stats:
        print(corpus_stats(chunks).model_dump_json(indent=2))  # noqa: T201 - CLI output

    if args.index:
        from src.rag.vector_store import build_vector_store  # lazy import (requires Ollama)

        store = build_vector_store(settings)
        if args.if_empty and not args.recreate and store.count() > 0:
            logger.info("Collection '%s' already populated: indexing skipped.", settings.qdrant_collection)
            return 0
        indexed = store.index_chunks(chunks, recreate=args.recreate)
        logger.info("%d chunks indexed in Qdrant (%s)", indexed, settings.qdrant_collection)
    return 0


if __name__ == "__main__":
    sys.exit(main())
