"""Validation of legal citations produced by the agents (anti-hallucination guardrail).

Each citation is checked against:
1. the evidence actually retrieved for the audit -> ``verified=True``;
2. otherwise the full corpus -> valid citation but *not backed by the evidence*;
3. otherwise -> non-existent provision (hallucination), reported as a warning.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.core.types import validated
from src.models.schemas import LegalChunk, LegalCitation, Regulation, RetrievedChunk, UnitType

_ARTICLE_RE = re.compile(r"\bart(?:icles?|\.)?\s*(\d{1,3})", re.IGNORECASE)
_ANNEX_RE = re.compile(r"\bannex(?:e)?\s+([IVXLC]+)\b", re.IGNORECASE)
_BRACKET_RE = re.compile(r"\[(AI Act|GDPR|RGPD)\s*,\s*([^\]]+)\]", re.IGNORECASE)


@validated
def parse_reference(regulation: Regulation, reference: str) -> list[str]:
    """``"Article 6(2) and Annex III"`` -> ``['ai_act:article:6', 'ai_act:annex:III']``."""
    keys = [f"{regulation.value}:{UnitType.ARTICLE.value}:{n}" for n in _ARTICLE_RE.findall(reference)]
    keys += [f"{regulation.value}:{UnitType.ANNEX.value}:{a.upper()}" for a in _ANNEX_RE.findall(reference)]
    return list(dict.fromkeys(keys))


class CitationValidator(BaseModel):
    """Validate citations against the corpus and the audit's evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    corpus_units: frozenset[str]
    evidence_units: dict[str, list[str]] = Field(default_factory=dict)

    @classmethod
    def from_corpus(cls, chunks: Iterable[LegalChunk]) -> CitationValidator:
        return cls(corpus_units=frozenset(c.metadata.unit_key for c in chunks))

    def with_evidence(self, evidence: Sequence[RetrievedChunk]) -> CitationValidator:
        units: dict[str, list[str]] = {}
        for item in evidence:
            units.setdefault(item.chunk.metadata.unit_key, []).append(item.chunk.chunk_id)
        return CitationValidator(corpus_units=self.corpus_units, evidence_units=units)

    @validated
    def validate_citations(self, citations: Sequence[LegalCitation]) -> tuple[list[LegalCitation], list[str]]:
        """Return the citations with ``verified``/``chunk_ids`` filled in, plus warnings."""
        validated_citations: list[LegalCitation] = []
        warnings: list[str] = []
        for citation in citations:
            keys = parse_reference(citation.regulation, citation.reference)
            label = f"{citation.regulation.short} - {citation.reference}"
            if not keys:
                warnings.append(f"Unparseable citation: {label}")
                validated_citations.append(citation.model_copy(update={"verified": False}))
                continue
            if any(k not in self.corpus_units for k in keys):
                warnings.append(f"Provision not found in the corpus (probable hallucination): {label}")
                validated_citations.append(citation.model_copy(update={"verified": False}))
                continue
            chunk_ids = [cid for k in keys for cid in self.evidence_units.get(k, [])]
            supported = all(k in self.evidence_units for k in keys)
            if not supported:
                warnings.append(f"Valid citation but missing from the retrieved evidence: {label}")
            validated_citations.append(
                citation.model_copy(update={"verified": supported, "chunk_ids": chunk_ids})
            )
        return validated_citations, warnings

    @validated
    def check_text(self, text: str) -> list[str]:
        """Check bracketed references in free text (chatbot answers)."""
        problems: list[str] = []
        for reg_label, ref in _BRACKET_RE.findall(text):
            regulation = Regulation.AI_ACT if reg_label.lower() == "ai act" else Regulation.GDPR
            for key in parse_reference(regulation, ref):
                if key not in self.corpus_units:
                    problems.append(f"[{reg_label}, {ref.strip()}]")
        return problems
