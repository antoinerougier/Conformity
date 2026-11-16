"""Assessment interview: the LLM asks the project team for the facts the documentation lacks.

Flow:
1. :meth:`InterviewEngine.start` profiles the project from the uploaded document and/or a
   free-text description, then generates the first round of questions.
2. :meth:`InterviewEngine.submit_answers` records the answers, refreshes the project profile
   with the new facts, then asks follow-up questions or declares the context sufficient.

The interview is bounded (``max_interview_rounds``) and every LLM call has a deterministic
fallback: if the model fails, questions come from a catalogue keyed on profile gaps.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from src.agents import prompts
from src.core.config import Settings
from src.core.llm import LLMService
from src.core.logging import get_logger
from src.core.types import validated
from src.ingestion.parser import condense_markdown
from src.models.schemas import (
    InterviewAnswer,
    InterviewQuestion,
    InterviewRound,
    InterviewRoundDraft,
    InterviewTopic,
    OperatorRole,
    ProjectDocument,
    ProjectProfile,
    ProjectSection,
    QuestionDraft,
    Regulation,
)

logger = get_logger(__name__)

MIN_DESCRIPTION_CHARS = 40


@validated
def document_from_text(description: str, *, title: str | None = None) -> ProjectDocument:
    """Wrap a free-text project description as a :class:`ProjectDocument`."""
    text = description.strip()
    if len(text) < MIN_DESCRIPTION_CHARS:
        raise ValueError(f"The project description must be at least {MIN_DESCRIPTION_CHARS} characters long")
    return ProjectDocument(
        file_name="project-description.txt",
        file_type="text",
        title=title,
        sections=[ProjectSection(index=1, kind="text", title=title, markdown=text)],
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def merge_documents(document: ProjectDocument, description: str) -> ProjectDocument:
    """Append a free-text description to an uploaded document as an extra section."""
    extra = description.strip()
    if not extra:
        return document
    section = ProjectSection(
        index=len(document.sections) + 1, kind="text", title="Additional context", markdown=extra
    )
    return document.model_copy(update={"sections": [*document.sections, section]})


def transcript(rounds: Sequence[InterviewRound]) -> str:
    """Full Q&A transcript of the answered rounds (injected into the audit prompts)."""
    return "\n\n".join(f"Round {r.number}\n{r.transcript()}" for r in rounds if r.answered)


# ---------------------------------------------------------------------------
# Deterministic fallback catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _CatalogQuestion:
    draft: QuestionDraft
    needed: Callable[[ProjectProfile], bool]


_CATALOG: tuple[_CatalogQuestion, ...] = (
    _CatalogQuestion(
        QuestionDraft(
            question="Does your organisation develop the AI system itself, "
            "or use a system built by a third party?",
            why="Providers and deployers have different obligations under the AI Act.",
            topic=InterviewTopic.ROLE,
            regulation=Regulation.AI_ACT,
            answer_type="choice",
            options=["We develop it (provider)", "We use a third-party system (deployer)", "Both"],
        ),
        lambda p: p.operator_role is OperatorRole.UNDETERMINED,
    ),
    _CatalogQuestion(
        QuestionDraft(
            question="Are the system's outputs used to take or prepare decisions about individuals "
            "(hiring, credit, access to a service, education, benefits...)?",
            why="Decisions about people can make the system high-risk (Annex III) and trigger GDPR Art. 22.",
            topic=InterviewTopic.DECISIONS,
            regulation=Regulation.AI_ACT,
            answer_type="yes_no",
        ),
        lambda p: not p.automated_decisions,
    ),
    _CatalogQuestion(
        QuestionDraft(
            question="Does a person review each output before it affects someone, and can they override it?",
            why="Human oversight is a core requirement for high-risk systems and for Art. 22 GDPR.",
            topic=InterviewTopic.HUMAN_OVERSIGHT,
            regulation=Regulation.AI_ACT,
            answer_type="yes_no",
        ),
        lambda p: not p.human_oversight_mentioned,
    ),
    _CatalogQuestion(
        QuestionDraft(
            question="Which categories of personal data does the system process, "
            "and where do they come from?",
            why="Determines the lawful basis, minimisation and information obligations.",
            topic=InterviewTopic.PERSONAL_DATA,
            regulation=Regulation.GDPR,
        ),
        lambda p: not p.personal_data_categories,
    ),
    _CatalogQuestion(
        QuestionDraft(
            question="Does the system process health, biometric or other special-category data, "
            "even indirectly?",
            why="Special categories require an Art. 9(2) exception and usually a DPIA.",
            topic=InterviewTopic.SENSITIVE_DATA,
            regulation=Regulation.GDPR,
            answer_type="yes_no",
        ),
        lambda p: not p.special_category_data and not p.biometric_processing,
    ),
    _CatalogQuestion(
        QuestionDraft(
            question="How long are personal data and model outputs kept, and how are they deleted?",
            why="Storage limitation (Art. 5(1)(e) GDPR).",
            topic=InterviewTopic.RETENTION,
            regulation=Regulation.GDPR,
        ),
        lambda p: not p.retention_policy,
    ),
    _CatalogQuestion(
        QuestionDraft(
            question="Are data hosted or accessed outside the European Union "
            "(cloud provider, model API, support)?",
            why="Transfers outside the EU need an adequacy decision or safeguards (Art. 44-49 GDPR).",
            topic=InterviewTopic.TRANSFERS,
            regulation=Regulation.GDPR,
            answer_type="yes_no",
        ),
        lambda p: not p.international_transfers,
    ),
    _CatalogQuestion(
        QuestionDraft(
            question="Has a data protection impact assessment (DPIA) been carried out for this processing?",
            why="A DPIA is mandatory for high-risk processing (Art. 35 GDPR).",
            topic=InterviewTopic.GOVERNANCE,
            regulation=Regulation.GDPR,
            answer_type="choice",
            options=["Yes, completed", "In progress", "No", "Not sure"],
        ),
        lambda p: not p.dpia_mentioned,
    ),
)


def fallback_questions(profile: ProjectProfile, asked: set[str], limit: int) -> list[QuestionDraft]:
    """Catalogue questions matching the profile's gaps that were not asked yet."""
    picked = [c.draft for c in _CATALOG if c.needed(profile) and c.draft.question not in asked]
    return picked[:limit]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class InterviewEngine:
    """Generates interview rounds and keeps the project profile up to date."""

    def __init__(self, llm: LLMService, settings: Settings) -> None:
        self.llm = llm
        self.settings = settings

    # ------------------------------------------------------------------ profile
    def profile(self, document: ProjectDocument, rounds: Sequence[InterviewRound] = ()) -> ProjectProfile:
        """Extract (or refresh) the factual project profile from the document and the answers."""
        content = condense_markdown(document, self.settings.project_context_chars)
        qa = transcript(rounds)
        user = f"PROJECT DOCUMENT:\n\n{content}" + (
            f"\n\nINTERVIEW WITH THE PROJECT TEAM:\n{qa}" if qa else ""
        )
        try:
            return self.llm.structured(prompts.PROFILE_SYSTEM, user, ProjectProfile)
        except Exception:  # deterministic fallback, logged
            logger.exception("Profiling failed during the interview")
            return ProjectProfile(
                project_name=document.title or "Untitled project",
                purpose="Undetermined (automatic extraction failed)",
                description=content[:500],
                sector="undetermined",
                unknowns=["Automatic profile extraction failed: answer the questions in detail"],
            )

    # ------------------------------------------------------------------ rounds
    def next_round(
        self, document: ProjectDocument, profile: ProjectProfile, rounds: Sequence[InterviewRound]
    ) -> InterviewRound | None:
        """Return the next round of questions, or ``None`` when the context is sufficient."""
        number = len(rounds) + 1
        if number > self.settings.max_interview_rounds:
            return None
        limit = self.settings.max_questions_per_round
        asked = {q.question for r in rounds for q in r.questions}
        profile_json = json.dumps(profile.model_dump(mode="json"), ensure_ascii=False, indent=1)
        user = (
            f"PROJECT PROFILE:\n{profile_json}\n\n"
            f"DOCUMENTATION EXCERPT:\n{condense_markdown(document, 6000)}\n\n"
            f"QUESTIONS ALREADY ASKED AND ANSWERS:\n{transcript(rounds) or '(none yet)'}"
        )
        try:
            draft = self.llm.structured(
                prompts.INTERVIEW_SYSTEM.format(max_q=limit), user, InterviewRoundDraft
            )
            questions = [q for q in draft.questions if q.question not in asked][:limit]
            if draft.ready or not questions:
                # Always ask at least one round so the team can add context the document lacks.
                if rounds:
                    return None
                questions = fallback_questions(profile, asked, limit)
        except Exception:  # deterministic fallback, logged
            logger.exception("Interviewer failed: using the fallback catalogue")
            questions = fallback_questions(profile, asked, limit)
        if not questions:
            return None
        return InterviewRound(
            number=number,
            questions=[
                InterviewQuestion(**q.model_dump(), question_id=f"r{number}-q{i}")
                for i, q in enumerate(questions, 1)
            ],
        )

    @validated
    def record_answers(self, current: InterviewRound, answers: Sequence[InterviewAnswer]) -> InterviewRound:
        """Attach answers to a round, ignoring ids that do not belong to it."""
        valid_ids = {q.question_id for q in current.questions}
        by_id = {a.question_id: a for a in answers if a.question_id in valid_ids}
        completed = [
            by_id.get(qid, InterviewAnswer(question_id=qid, skipped=True)) for qid in sorted(valid_ids)
        ]
        return current.model_copy(update={"answers": completed})
