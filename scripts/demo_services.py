"""Offline demo services: the real web application wired to a *scripted* language model.

Used to capture the README screenshots reproducibly (CI, machines without Ollama). The
scripted answers describe the fictitious WorkPulse 360 project of
``examples/workpulse360_high_risk_project.pdf``. Everything else (parsing, legal corpus,
retrieval, citation checks, deadlines, scoring, checklist, PDF export) is the real code.

For real model output, run the application normally (``uvicorn src.api.app:app``) or
``scripts/capture_screenshots.py --live``.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Iterator, Sequence
from typing import Any, TypeVar

from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from qdrant_client import QdrantClient

from src.api.services import AppServices
from src.core.config import RerankerBackend, Settings
from src.core.llm import OllamaHealth
from src.models.schemas import (
    AIActClassificationDraft,
    ComplianceStatus,
    Effort,
    GDPRAuditDraft,
    GDPRFinding,
    InterviewRoundDraft,
    InterviewTopic,
    LegalBasis,
    LegalCitation,
    Obligation,
    OperatorRole,
    Priority,
    ProjectProfile,
    QueryPlan,
    QuestionDraft,
    RecommendationDraft,
    Regulation,
    RethinkerVerdict,
    RiskCategory,
    Severity,
    SubQuery,
    SynthesisDraft,
)
from src.rag.preprocess_laws import load_or_build_corpus
from src.rag.retriever import BM25Index, HybridRetriever, NoOpReranker, tokenize
from src.rag.vector_store import LegalVectorStore

T = TypeVar("T", bound=BaseModel)
AI, GDPR = Regulation.AI_ACT, Regulation.GDPR


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings (no model download)."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class DemoReranker(NoOpReranker):
    name = "demo (RRF order, no reranker model)"


def _cite(reg: Regulation, ref: str, why: str = "") -> LegalCitation:
    return LegalCitation(regulation=reg, reference=ref, justification=why)


PROFILE = ProjectProfile(
    project_name="WorkPulse 360",
    purpose="Continuously score employees' engagement and loyalty and recommend promotions, warnings and dismissals.",
    description="SaaS sold to employers. Webcam and microphone emotion analysis, biometric categorisation, "
    "a Trust Score per employee built from productivity, sick leave, social media and location data, and "
    "automatic dismissal flags.",
    sector="Human resources / workforce analytics",
    ai_techniques=[
        "Emotion recognition (CNN)",
        "Biometric categorisation",
        "Gradient boosting",
        "Third-party LLM",
    ],
    operator_role=OperatorRole.PROVIDER,
    users=["HR departments", "Managers"],
    affected_persons=["Employees of customer companies"],
    personal_data_categories=[
        "Video and voice",
        "Keystrokes and screenshots",
        "GPS location",
        "Social media posts",
    ],
    special_category_data=[
        "Health (sick-leave reasons)",
        "Biometric data",
        "Ethnic origin",
        "Trade-union membership",
    ],
    data_sources=["Workstations", "Personal smartphones", "HR system", "Web scraping"],
    automated_decisions=True,
    profiling=True,
    biometric_processing=True,
    uses_general_purpose_model=True,
    international_transfers=True,
    retention_policy="Indefinite",
    unknowns=[
        "Whether a manager can see and overturn the reasons behind a dismissal flag",
        "Contractual basis for sharing Trust Scores with the partner insurer",
        "Whether employee data is used to train the third-party model",
    ],
)

ROUND_1 = InterviewRoundDraft(
    ready=False,
    rationale="Human review, defaults and data sharing decide between high-risk and prohibited uses.",
    questions=[
        QuestionDraft(
            question="When a dismissal letter is generated, can the manager see why the score dropped and cancel it?",
            why="Decides whether decisions are solely automated (GDPR Art. 22) and what oversight exists.",
            topic=InterviewTopic.HUMAN_OVERSIGHT,
            regulation=GDPR,
            answer_type="yes_no",
        ),
        QuestionDraft(
            question="Is emotion analysis switched on for every employee, or only for those who opt in?",
            why="Emotion recognition at work is prohibited unless used for medical or safety reasons.",
            topic=InterviewTopic.SENSITIVE_DATA,
            regulation=AI,
            answer_type="choice",
            options=["On for everyone", "Opt-in per employee", "Each customer decides"],
        ),
        QuestionDraft(
            question="On what contractual basis does the partner insurer receive Trust Scores?",
            why="Disclosure to third parties needs its own lawful basis and must be in the privacy notice.",
            topic=InterviewTopic.PERSONAL_DATA,
            regulation=GDPR,
        ),
        QuestionDraft(
            question="Which model writes the performance narratives, and is employee data used to train it?",
            why="Determines whether obligations for general-purpose model providers or deployers apply.",
            topic=InterviewTopic.MODEL,
            regulation=AI,
        ),
    ],
)

AI_ACT = AIActClassificationDraft(
    is_ai_system=True,
    risk_category=RiskCategory.PROHIBITED,
    secondary_categories=[RiskCategory.HIGH_RISK],
    operator_role=OperatorRole.PROVIDER,
    annex_iii_areas=["4(b) Decisions on promotion and termination, monitoring and evaluation of workers"],
    prohibited_practice_flags=[
        "Emotion recognition of employees in the workplace, with no medical or safety purpose (Art. 5(1)(f))",
        "Biometric categorisation inferring ethnic origin and trade-union membership (Art. 5(1)(g))",
    ],
    justification="WorkPulse 360 infers the emotions of employees from webcam and microphone during working "
    "hours and categorises them by ethnic origin and likely union membership from their faces. Both practices "
    "are prohibited by Article 5 regardless of consent. The Trust Score, used to decide promotions and "
    "dismissals, also falls under Annex III point 4(b), so the remaining features are high-risk.",
    obligations=[
        Obligation(
            reference="Article 5(1)(f)",
            description="Stop emotion recognition in the workplace",
            status=ComplianceStatus.NON_COMPLIANT,
            gap="Active for every employee by default.",
        ),
        Obligation(
            reference="Article 5(1)(g)",
            description="Stop biometric categorisation by sensitive attributes",
            status=ComplianceStatus.NON_COMPLIANT,
            gap="Ethnicity and union membership inferred from faces.",
        ),
        Obligation(
            reference="Article 9",
            description="Risk management system",
            status=ComplianceStatus.NON_COMPLIANT,
            gap="No risk management process documented.",
        ),
        Obligation(
            reference="Article 10",
            description="Data governance and bias examination",
            status=ComplianceStatus.NON_COMPLIANT,
            gap="Face model error rates by skin tone unknown.",
        ),
        Obligation(
            reference="Article 14",
            description="Effective human oversight",
            status=ComplianceStatus.NON_COMPLIANT,
            gap="Managers cannot see the reasons behind a flag.",
        ),
        Obligation(
            reference="Article 4",
            description="AI literacy of HR staff",
            status=ComplianceStatus.TO_VERIFY,
            gap="No training mentioned.",
        ),
    ],
    citations=[
        _cite(AI, "Article 5", "Prohibited practices"),
        _cite(AI, "Annex III", "Employment use cases"),
        _cite(AI, "Article 6", "High-risk classification"),
        _cite(AI, "Article 14", "Human oversight"),
    ],
    confidence=0.86,
)

GDPR_AUDIT = GDPRAuditDraft(
    processes_personal_data=True,
    identified_legal_basis=LegalBasis.CONSENT,
    legal_basis_assessment="Consent imposed in the employment contract is not freely given: refusing it prevents "
    "hiring. No other lawful basis covers 24/7 location tracking or social-media scraping.",
    legal_basis_status=ComplianceStatus.NON_COMPLIANT,
    minimisation_status=ComplianceStatus.NON_COMPLIANT,
    minimisation_assessment="Off-work behaviour, private social media and indefinite retention go far beyond what "
    "is necessary for workforce management.",
    dpia_required=True,
    dpia_triggers=["Systematic evaluation with significant effects", "Large-scale special-category data"],
    dpia_status=ComplianceStatus.NON_COMPLIANT,
    rights_status=ComplianceStatus.NON_COMPLIANT,
    rights_assessment="Employees are not informed of the emotion analysis and cannot access or erase their score.",
    findings=[
        GDPRFinding(
            topic="automated_decision_making",
            status=ComplianceStatus.NON_COMPLIANT,
            severity=Severity.CRITICAL,
            finding="Dismissal letters are sent automatically after ten days unless a manager intervenes.",
            recommendation="Require a documented human decision before any HR measure; let employees contest it.",
            citations=[_cite(GDPR, "Article 22")],
        ),
        GDPRFinding(
            topic="special_categories",
            status=ComplianceStatus.NON_COMPLIANT,
            severity=Severity.CRITICAL,
            finding="Health data, biometric templates, ethnic origin and union membership are processed.",
            recommendation="Delete these data and remove the features; no Art. 9(2) exception applies.",
            citations=[_cite(GDPR, "Article 9")],
        ),
        GDPRFinding(
            topic="international_transfers",
            status=ComplianceStatus.NON_COMPLIANT,
            severity=Severity.HIGH,
            finding="Raw video and scores are stored in the United States without transfer safeguards.",
            recommendation="Host in the EU or put standard contractual clauses and a transfer impact assessment in place.",
            citations=[_cite(GDPR, "Article 44")],
        ),
        GDPRFinding(
            topic="storage_limitation",
            status=ComplianceStatus.NON_COMPLIANT,
            severity=Severity.HIGH,
            finding="Video, biometric data and scores are kept indefinitely for model training.",
            recommendation="Define retention periods per category and purge historical data.",
            citations=[_cite(GDPR, "Article 5")],
        ),
    ],
    confidence=0.82,
)

SYNTHESIS = SynthesisDraft(
    executive_summary="WorkPulse 360 cannot be placed on the EU market in its current form. Two of its core "
    "features, emotion recognition at work and biometric categorisation by ethnicity and union membership, are "
    "prohibited AI practices that no consent or safeguard can cure. Without them, the Trust Score remains a "
    "high-risk system under Annex III, whose obligations apply from 2 December 2027, and it relies on "
    "automated dismissal decisions and special-category data that the GDPR does not allow. The January 2027 "
    "launch should be postponed until the prohibited features are removed and a DPIA, a valid lawful basis "
    "and human review of every HR decision are in place.",
    key_risks=[
        "Fines of up to EUR 35 million or 7% of worldwide turnover for the prohibited practices",
        "Unlawful processing of health, biometric and union data",
        "Dismissals decided without human review",
    ],
    recommendations=[
        RecommendationDraft(
            title="Remove emotion recognition and biometric categorisation",
            description="Disable the features for all customers and delete the related data and models.",
            regulation=AI,
            priority=Priority.P0,
            effort=Effort.MEDIUM,
            owner="CTO",
            references=["Article 5(1)(f)", "Article 5(1)(g)"],
        ),
        RecommendationDraft(
            title="Stop automatic dismissal letters",
            description="Every HR measure must be a documented decision by a person who sees the reasons.",
            regulation=GDPR,
            priority=Priority.P0,
            effort=Effort.LOW,
            owner="Product",
            references=["Article 22"],
        ),
        RecommendationDraft(
            title="Run a DPIA before any EU launch",
            description="Cover the Trust Score, monitoring and data sharing; consult the authority if risks remain.",
            regulation=GDPR,
            priority=Priority.P0,
            effort=Effort.HIGH,
            owner="DPO",
            references=["Article 35", "Article 36"],
        ),
        RecommendationDraft(
            title="Stop selling scores to insurers and recruiters",
            description="No lawful basis and incompatible purposes.",
            regulation=GDPR,
            priority=Priority.P0,
            effort=Effort.LOW,
            owner="CEO",
            references=["Article 6", "Article 5"],
        ),
        RecommendationDraft(
            title="Build the high-risk compliance file",
            description="Risk management, data governance and bias testing, technical documentation, logging, "
            "human oversight design, conformity assessment and EU registration.",
            regulation=AI,
            priority=Priority.P1,
            effort=Effort.HIGH,
            owner="Compliance",
            references=["Articles 9-15", "Article 43", "Article 49"],
        ),
        RecommendationDraft(
            title="Move hosting to the EU and set retention periods",
            description="Purge indefinite archives; document retention per data category.",
            regulation=GDPR,
            priority=Priority.P1,
            effort=Effort.MEDIUM,
            owner="Infrastructure",
            references=["Article 5(1)(e)", "Article 44"],
        ),
        RecommendationDraft(
            title="Train HR users on the system's limits",
            description="AI literacy programme for customer HR teams and internal staff.",
            regulation=AI,
            priority=Priority.P2,
            effort=Effort.LOW,
            owner="Customer success",
            references=["Article 4"],
        ),
    ],
)

CHAT_ANSWER = (
    "Start with the two prohibited practices: emotion recognition of employees and biometric categorisation by "
    "ethnic origin and union membership are banned outright [AI Act, Art. 5], so they must be switched off and "
    "their data deleted before anything else. Next, stop the automatic dismissal letters: a person must take each "
    "decision and employees must be able to contest it [GDPR, Art. 22]. Then run the DPIA [GDPR, Art. 35], which "
    "will drive the rest of the plan, including the high-risk documentation due by 2 December 2027."
)


class ScriptedLLM:
    """Deterministic stand-in for the local model (see module docstring)."""

    model_name = "qwen2.5:14b (scripted demo)"

    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.interview_calls = 0

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        answers: dict[type[BaseModel], Any] = {
            ProjectProfile: PROFILE,
            QueryPlan: QueryPlan(
                sub_queries=[
                    SubQuery(question="Emotion recognition in the workplace prohibited practice", target=AI),
                    SubQuery(
                        question="Biometric categorisation inferring race or trade union membership",
                        target=AI,
                    ),
                    SubQuery(
                        question="AI systems for promotion and termination of work relationships", target=AI
                    ),
                    SubQuery(question="Décision individuelle automatisée effets juridiques", target=GDPR),
                    SubQuery(question="Données concernant la santé et données biométriques", target=GDPR),
                ]
            ),
            RethinkerVerdict: RethinkerVerdict(sufficient=True, confidence=0.84),
            AIActClassificationDraft: AI_ACT,
            GDPRAuditDraft: GDPR_AUDIT,
            SynthesisDraft: SYNTHESIS,
        }
        if schema is InterviewRoundDraft:
            self.interview_calls += 1
            result: Any = ROUND_1 if self.interview_calls == 1 else InterviewRoundDraft(ready=True)
        else:
            result = answers[schema]
        if self.delay_s and schema not in {ProjectProfile, InterviewRoundDraft}:
            time.sleep(self.delay_s)  # lets the progress page show each step
        assert isinstance(result, schema)
        return result

    def stream(self, messages: Sequence[BaseMessage]) -> Iterator[str]:
        for word in CHAT_ANSWER.split(" "):
            time.sleep(self.delay_s / 40)
            yield word + " "


def build_demo_services(*, delay_s: float = 0.0) -> AppServices:
    """Real application services with the scripted model and an in-memory index."""
    settings = Settings(
        _env_file=None, qdrant_url=None, qdrant_path=None, reranker_backend=RerankerBackend.NONE
    )
    corpus = load_or_build_corpus(settings)
    store = LegalVectorStore(
        QdrantClient(location=":memory:"), HashingEmbeddings(), settings.qdrant_collection
    )
    store.index_chunks(corpus)
    retriever = HybridRetriever(
        store=store, bm25=BM25Index(corpus), reranker=DemoReranker(), settings=settings
    )
    health = OllamaHealth(reachable=True, available_models=[settings.llm_model, settings.embedding_model])
    return AppServices(settings, llm=ScriptedLLM(delay_s), retriever=retriever, ollama_health=health)
