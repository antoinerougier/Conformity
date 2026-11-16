"""Generation of a synthetic test set from the indexed legal corpus.

Stratified, reproducible (fixed seed) sampling of the topic-tagged chunks of each regulation,
then generation by the local LLM of a (question, reference answer) pair grounded exclusively
on the chunk. The source chunk is stored as ``reference_contexts``.

Usage:
    python -m src.evaluation.dataset --per-source 10 --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.core.config import Settings, get_settings
from src.core.llm import LLMService, OllamaLLMService
from src.core.logging import configure_logging, get_logger
from src.core.types import NonNegativeInt, PositiveInt, validated
from src.models.schemas import LegalChunk, NonEmptyStr, Regulation
from src.rag.preprocess_laws import load_or_build_corpus

logger = get_logger(__name__)

MIN_CHUNK_CHARS_FOR_SAMPLING = 400


class EvalSample(BaseModel):
    """Evaluation sample (curated "golden" set or synthetic set)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: NonEmptyStr
    regulation: Regulation
    user_input: NonEmptyStr
    reference: NonEmptyStr
    reference_chunk_ids: list[str] = Field(default_factory=list)
    reference_contexts: list[str] = Field(default_factory=list)
    origin: str = "golden"


class SyntheticQA(BaseModel):
    question: str = Field(min_length=10, description="Question an AI project manager could ask, in English")
    answer: str = Field(min_length=20, description="Reference answer based only on the excerpt")


GENERATION_SYSTEM = """\
You generate evaluation data for an EU AI Act / GDPR legal assistant.
From the EXCERPT provided:
- write ONE realistic question, in English, that an AI project manager could ask and whose
  answer is fully contained in the excerpt (without mentioning "the excerpt");
- write the reference answer, in English, factual, complete and based ONLY on the excerpt,
  citing the provision concerned.
"""


@validated
def sample_chunks(chunks: Sequence[LegalChunk], per_source: PositiveInt, seed: int) -> list[LegalChunk]:
    """Sampling stratified by regulation, restricted to topic-tagged, substantial chunks."""
    rng = random.Random(seed)  # noqa: S311 - reproducibility, not cryptography
    selected: list[LegalChunk] = []
    for source in Regulation:
        pool = sorted(
            (
                c
                for c in chunks
                if c.metadata.source is source
                and c.metadata.topics
                and len(c.text) > MIN_CHUNK_CHARS_FOR_SAMPLING
            ),
            key=lambda c: c.chunk_id,
        )
        selected.extend(rng.sample(pool, min(per_source, len(pool))))
    return selected


def generate_synthetic(
    llm: LLMService, chunks: Sequence[LegalChunk], *, per_source: int, seed: int
) -> list[EvalSample]:
    """Generate one grounded QA pair per sampled chunk (failures are skipped and logged)."""
    samples: list[EvalSample] = []
    for i, chunk in enumerate(sample_chunks(chunks, per_source, seed), start=1):
        try:
            qa = llm.structured(
                GENERATION_SYSTEM, f"EXCERPT ({chunk.metadata.citation}):\n{chunk.text}", SyntheticQA
            )
        except Exception as exc:  # noqa: BLE001 - one failure does not invalidate the whole set
            logger.warning("Generation skipped for %s: %s", chunk.chunk_id, exc)
            continue
        samples.append(
            EvalSample(
                id=f"synthetic-{i:03d}",
                regulation=chunk.metadata.source,
                user_input=qa.question,
                reference=qa.answer,
                reference_chunk_ids=[chunk.chunk_id],
                reference_contexts=[chunk.text],
                origin="synthetic",
            )
        )
        logger.info("[%d] %s -> %s", i, chunk.metadata.citation, qa.question)
    return samples


def write_samples(samples: Sequence[EvalSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(s.model_dump_json() + "\n")


def load_samples(path: Path) -> list[EvalSample]:
    """Load a JSONL test set (empty list if the file does not exist)."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return [EvalSample.model_validate_json(line) for line in fh if line.strip()]


@validated
def load_testset(settings: Settings, max_samples: NonNegativeInt | None = None) -> list[EvalSample]:
    """Golden set (first) completed with the synthetic set when present; regulations are
    interleaved so that a small CI sub-sample stays balanced."""
    pool = load_samples(settings.ragas_testset_path) + load_samples(settings.ragas_synthetic_testset_path)
    by_reg = {reg: [s for s in pool if s.regulation is reg] for reg in Regulation}
    interleaved: list[EvalSample] = []
    while any(by_reg.values()):
        for reg in Regulation:
            if by_reg[reg]:
                interleaved.append(by_reg[reg].pop(0))
    return interleaved[:max_samples] if max_samples else interleaved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--per-source", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.per_source < 1:
        parser.error("--per-source must be >= 1")

    settings = get_settings()
    configure_logging(settings.log_level)
    chunks = load_or_build_corpus(settings)
    samples = generate_synthetic(
        OllamaLLMService(settings), chunks, per_source=args.per_source, seed=args.seed
    )
    out = args.out or settings.ragas_synthetic_testset_path
    write_samples(samples, out)
    logger.info("%d synthetic samples written to %s", len(samples), out)
    return 0 if samples else 1


if __name__ == "__main__":
    sys.exit(main())
