from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agents.interview import (
    InterviewEngine,
    document_from_text,
    fallback_questions,
    merge_documents,
    transcript,
)
from src.core.config import Settings
from src.core.exceptions import StructuredOutputError
from src.models.schemas import (
    InterviewAnswer,
    InterviewRoundDraft,
    InterviewTopic,
    OperatorRole,
    ProjectProfile,
    QuestionDraft,
    Regulation,
)
from tests.conftest import FakeLLM

DESCRIPTION = "A chatbot answering customer questions about bicycle parts and order status, hosted in the EU."


def _profile(**kw: object) -> ProjectProfile:
    return ProjectProfile.model_validate(
        {"project_name": "Atlas", "purpose": "Support", "description": "d", "sector": "retail", **kw}
    )


def _question(text: str, **kw: object) -> QuestionDraft:
    return QuestionDraft.model_validate({"question": text, "topic": "purpose", **kw})


def _engine(llm: FakeLLM, **overrides: int) -> InterviewEngine:
    return InterviewEngine(llm, Settings.model_validate(overrides))


class TestDocuments:
    def test_document_from_text(self) -> None:
        doc = document_from_text(DESCRIPTION, title="Atlas")
        assert doc.file_type == "text"
        assert doc.sections[0].markdown == DESCRIPTION
        assert len(doc.sha256) == 64

    def test_document_from_text_rejects_short_descriptions(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            document_from_text("Too short")

    def test_merge_documents_appends_context(self) -> None:
        doc = merge_documents(document_from_text(DESCRIPTION), "Retention is 30 days.")
        assert [s.title for s in doc.sections] == [None, "Additional context"]
        assert merge_documents(doc, "  ") is doc


class TestQuestions:
    def test_choice_question_needs_options(self) -> None:
        with pytest.raises(ValidationError):
            _question("Which hosting region do you use?", answer_type="choice", options=["EU"])

    def test_fallback_questions_follow_profile_gaps(self) -> None:
        gaps = fallback_questions(_profile(), set(), limit=8)
        assert {q.topic for q in gaps} >= {
            InterviewTopic.ROLE,
            InterviewTopic.RETENTION,
            InterviewTopic.TRANSFERS,
        }
        known = _profile(operator_role=OperatorRole.PROVIDER, retention_policy="30 days")
        topics = {q.topic for q in fallback_questions(known, set(), limit=8)}
        assert InterviewTopic.ROLE not in topics
        assert InterviewTopic.RETENTION not in topics
        assert len(fallback_questions(_profile(), set(), limit=2)) == 2


class TestEngine:
    def test_first_round_uses_llm_questions(self, fake_llm: FakeLLM) -> None:
        fake_llm.queue(
            InterviewRoundDraft,
            InterviewRoundDraft(
                ready=False,
                questions=[
                    _question("Who reviews the answers?", regulation=Regulation.AI_ACT, answer_type="yes_no")
                ],
            ),
        )
        engine = _engine(fake_llm)
        doc = document_from_text(DESCRIPTION)
        round_1 = engine.next_round(doc, _profile(), [])
        assert round_1 is not None
        assert [q.question_id for q in round_1.questions] == ["r1-q1"]
        assert round_1.questions[0].answer_type == "yes_no"

    def test_ready_on_first_round_still_asks_fallback_questions(self, fake_llm: FakeLLM) -> None:
        fake_llm.queue(InterviewRoundDraft, InterviewRoundDraft(ready=True))
        round_1 = _engine(fake_llm, max_questions_per_round=3).next_round(
            document_from_text(DESCRIPTION), _profile(), []
        )
        assert round_1 is not None
        assert len(round_1.questions) == 3

    def test_llm_failure_falls_back_to_catalogue(self, fake_llm: FakeLLM) -> None:
        fake_llm.queue(InterviewRoundDraft, StructuredOutputError("bad JSON"))
        round_1 = _engine(fake_llm).next_round(document_from_text(DESCRIPTION), _profile(), [])
        assert round_1 is not None
        assert round_1.questions

    def test_answers_transcript_and_follow_up(self, fake_llm: FakeLLM) -> None:
        fake_llm.queue(
            InterviewRoundDraft,
            InterviewRoundDraft(
                ready=False, questions=[_question("Who uses it?"), _question("Where is it hosted?")]
            ),
            InterviewRoundDraft(ready=True),
        )
        engine = _engine(fake_llm)
        doc = document_from_text(DESCRIPTION)
        first = engine.next_round(doc, _profile(), [])
        assert first is not None
        answered = engine.record_answers(
            first,
            [
                InterviewAnswer(question_id="r1-q1", answer="Customers"),
                InterviewAnswer(question_id="r9-q9", answer="x"),
            ],
        )
        assert [a.question_id for a in answered.answers] == ["r1-q1", "r1-q2"]
        assert answered.answers[1].skipped
        text = transcript([answered])
        assert "Q: Who uses it?\nA: Customers" in text
        assert "A: (skipped)" in text
        assert engine.next_round(doc, _profile(), [answered]) is None  # LLM says ready

    def test_interview_is_bounded(self, fake_llm: FakeLLM) -> None:
        fake_llm.queue(
            InterviewRoundDraft, InterviewRoundDraft(ready=False, questions=[_question("Anything else?")])
        )
        engine = _engine(fake_llm, max_interview_rounds=1)
        doc = document_from_text(DESCRIPTION)
        first = engine.next_round(doc, _profile(), [])
        assert first is not None
        assert engine.next_round(doc, _profile(), [engine.record_answers(first, [])]) is None

    def test_profile_includes_answers_and_falls_back(self, fake_llm: FakeLLM) -> None:
        fake_llm.queue(ProjectProfile, StructuredOutputError("timeout"))
        profile = _engine(fake_llm).profile(document_from_text(DESCRIPTION, title="Atlas"))
        assert profile.project_name == "Atlas"
        assert profile.unknowns
