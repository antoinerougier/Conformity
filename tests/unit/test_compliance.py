from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from src.compliance import timeline as tl
from src.compliance.citations import CitationValidator, parse_reference
from src.compliance.obligations import baseline_checklist
from src.compliance.scoring import compliance_score, recommendation_due_date
from src.models.schemas import (
    AIActClassificationDraft,
    ComplianceStatus,
    GDPRAuditDraft,
    GDPRFinding,
    LegalBasis,
    LegalChunk,
    LegalCitation,
    Obligation,
    OperatorRole,
    Priority,
    ProjectProfile,
    Regulation,
    RetrievedChunk,
    RiskCategory,
    Severity,
)
from tests.conftest import find_chunk


def _ai(category: RiskCategory = RiskCategory.HIGH_RISK, **kw: object) -> AIActClassificationDraft:
    data: dict[str, object] = {
        "is_ai_system": True,
        "risk_category": category,
        "operator_role": OperatorRole.PROVIDER,
        "justification": "test",
        "confidence": 0.8,
    }
    return AIActClassificationDraft.model_validate({**data, **kw})


def _gdpr(**kw: object) -> GDPRAuditDraft:
    ok = ComplianceStatus.COMPLIANT
    data: dict[str, object] = {
        "processes_personal_data": True,
        "identified_legal_basis": LegalBasis.LEGITIMATE_INTERESTS,
        "legal_basis_assessment": "",
        "legal_basis_status": ok,
        "minimisation_status": ok,
        "minimisation_assessment": "",
        "dpia_required": True,
        "dpia_status": ok,
        "rights_status": ok,
        "rights_assessment": "",
        "confidence": 0.7,
    }
    return GDPRAuditDraft.model_validate({**data, **kw})


def _profile(**kw: object) -> ProjectProfile:
    return ProjectProfile.model_validate(
        {"project_name": "P", "purpose": "x", "description": "x", "sector": "HR", **kw}
    )


class TestTimeline:
    def test_deadlines_sorted_and_omnibus_dates(self) -> None:
        deadlines = tl.all_deadlines()
        assert [d.date for d in deadlines] == sorted(d.date for d in deadlines)
        by_id = {d.deadline_id: d for d in deadlines}
        assert by_id["ai-act-high-risk-annex-iii"].date == date(2027, 12, 2)
        assert by_id["ai-act-high-risk-annex-i"].date == date(2028, 8, 2)
        assert by_id["ai-act-high-risk-annex-iii"].amended_by == tl.OMNIBUS

    def test_high_risk_deadlines_depend_on_annex_i(self) -> None:
        ids = {d.deadline_id for d in tl.deadlines_for([RiskCategory.HIGH_RISK])}
        assert "ai-act-high-risk-annex-iii" in ids
        assert "ai-act-high-risk-annex-i" not in ids
        assert "gdpr-application" in ids
        ids_i = {d.deadline_id for d in tl.deadlines_for([RiskCategory.HIGH_RISK], annex_i_product=True)}
        assert "ai-act-high-risk-annex-i" in ids_i

    def test_minimal_risk_has_no_high_risk_deadline(self) -> None:
        ids = {d.deadline_id for d in tl.deadlines_for([RiskCategory.MINIMAL_RISK], include_gdpr=False)}
        assert not {i for i in ids if "high-risk" in i}
        assert "gdpr-application" not in ids

    @pytest.mark.parametrize(
        ("category", "annex_i", "expected"),
        [
            (RiskCategory.PROHIBITED, False, date(2025, 2, 2)),
            (RiskCategory.HIGH_RISK, False, date(2027, 12, 2)),
            (RiskCategory.HIGH_RISK, True, date(2028, 8, 2)),
            (RiskCategory.GPAI, False, date(2025, 8, 2)),
            (RiskCategory.LIMITED_RISK, False, date(2026, 8, 2)),
        ],
    )
    def test_compliance_due_date(self, category: RiskCategory, annex_i: bool, expected: date) -> None:
        assert tl.compliance_due_date(category, annex_i_product=annex_i) == expected

    def test_inputs_are_validated(self) -> None:
        with pytest.raises(ValidationError):
            tl.deadlines_for(["not-a-category"])  # type: ignore[list-item]
        with pytest.raises(ValidationError):
            tl.ai_act_article_risk(0, None)
        assert tl.compliance_due_date("high_risk") == date(2027, 12, 2)  # type: ignore[arg-type]  # coerced

    def test_deadline_helpers(self) -> None:
        d = tl.all_deadlines()[0]
        assert d.is_past(date(2030, 1, 1))
        assert d.days_remaining(d.date) == 0


class TestCitations:
    def test_parse_reference(self) -> None:
        assert parse_reference(Regulation.AI_ACT, "Article 6(2) and Annex III, point 4") == [
            "ai_act:article:6",
            "ai_act:annex:III",
        ]
        assert parse_reference(Regulation.GDPR, "Art. 35(3)(a)") == ["gdpr:article:35"]
        assert parse_reference(Regulation.GDPR, "recital") == []

    def test_validator_classifies_citations(self, corpus: list[LegalChunk]) -> None:
        validator = CitationValidator.from_corpus(corpus).with_evidence(
            [RetrievedChunk(chunk=find_chunk(corpus, "ai_act:article:14:p1"))]
        )
        cites = [
            LegalCitation(regulation=Regulation.AI_ACT, reference="Article 14"),
            LegalCitation(regulation=Regulation.AI_ACT, reference="Article 26"),
            LegalCitation(regulation=Regulation.AI_ACT, reference="Article 250"),
            LegalCitation(regulation=Regulation.GDPR, reference="Recital 71"),
        ]
        validated, warnings = validator.validate_citations(cites)
        assert [c.verified for c in validated] == [True, False, False, False]
        assert validated[0].chunk_ids == ["ai_act:article:14:p1"]
        assert any("missing from the retrieved evidence" in w for w in warnings)
        assert any("not found in the corpus" in w and "250" in w for w in warnings)
        assert any("Unparseable" in w for w in warnings)

    def test_check_text_detects_hallucinated_articles(self, corpus: list[LegalChunk]) -> None:
        validator = CitationValidator.from_corpus(corpus)
        assert validator.check_text("See [AI Act, Art. 14] and [GDPR, Art. 35].") == []
        assert validator.check_text("See [GDPR, Art. 120] and [AI Act, Annex XIV].") == [
            "[GDPR, Art. 120]",
            "[AI Act, Annex XIV]",
        ]

    def test_empty_reference_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LegalCitation(regulation=Regulation.GDPR, reference="   ")


class TestScoring:
    def test_all_compliant_scores_100(self) -> None:
        ai = _ai(
            obligations=[Obligation(reference="Article 9", description="", status=ComplianceStatus.COMPLIANT)]
        )
        assert compliance_score(ai, _gdpr()) == 100

    def test_non_compliant_lowers_score(self) -> None:
        bad = _gdpr(
            legal_basis_status=ComplianceStatus.NON_COMPLIANT,
            findings=[
                GDPRFinding(
                    topic="dpia",
                    status=ComplianceStatus.NON_COMPLIANT,
                    severity=Severity.CRITICAL,
                    finding="",
                    recommendation="",
                )
            ],
        )
        assert compliance_score(_ai(), bad) < 70

    def test_prohibited_practice_caps_score(self) -> None:
        assert compliance_score(_ai(RiskCategory.PROHIBITED), _gdpr()) <= 10

    def test_no_signal_defaults_to_50(self) -> None:
        assert compliance_score(_ai(), _gdpr(processes_personal_data=False)) == 50

    def test_due_date_capped_by_regulation(self) -> None:
        today = date(2027, 11, 1)
        assert recommendation_due_date(
            Regulation.AI_ACT, Priority.P3, RiskCategory.HIGH_RISK, today=today
        ) == date(2027, 12, 2)
        # Already enforceable: 30-day window only.
        assert recommendation_due_date(
            Regulation.AI_ACT, Priority.P0, RiskCategory.PROHIBITED, today=today
        ) == date(2027, 12, 1)
        assert recommendation_due_date(
            Regulation.GDPR, Priority.P1, RiskCategory.HIGH_RISK, today=today
        ) == date(2028, 1, 30)

    def test_llm_output_schema_rejects_out_of_range_confidence(self) -> None:
        with pytest.raises(ValidationError):
            _ai(confidence=1.5)
        with pytest.raises(ValidationError):
            _ai(unexpected_field=True)  # extra="forbid"


class TestChecklist:
    def test_high_risk_provider_gets_core_obligations(self) -> None:
        items = baseline_checklist(_ai(), _gdpr(), _profile())
        ids = {i.item_id for i in items}
        assert {"aia-hr-risk-mgmt", "aia-hr-human-oversight", "aia-hr-conformity", "gdpr-dpia"} <= ids
        assert "aia-hr-deployer" not in ids
        assert next(i for i in items if i.item_id == "aia-hr-risk-mgmt").due_date == date(2027, 12, 2)

    def test_conditional_entries(self) -> None:
        profile = _profile(
            interacts_with_natural_persons=True, automated_decisions=True, international_transfers=True
        )
        ids = {
            i.item_id
            for i in baseline_checklist(
                _ai(RiskCategory.MINIMAL_RISK, operator_role=OperatorRole.DEPLOYER),
                _gdpr(dpia_required=False),
                profile,
            )
        }
        assert {"aia-transp-interaction", "gdpr-art22", "gdpr-transfers"} <= ids
        assert not {i for i in ids if i.startswith("aia-hr")}
        assert "gdpr-dpia" not in ids
