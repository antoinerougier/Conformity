"""End-to-end tests of the web application (FastAPI) with the scripted demo model."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scripts.demo_services import build_demo_services
from src.api.app import create_app
from src.api.services import AppServices
from tests.conftest import ROOT

pytestmark = pytest.mark.integration

WORKPULSE = ROOT / "examples" / "workpulse360_high_risk_project.pdf"
HTML = {"accept": "text/html"}


@pytest.fixture(scope="module")
def services() -> AppServices:
    return build_demo_services()


@pytest.fixture()
def client(services: AppServices) -> Iterator[TestClient]:
    services.llm.interview_calls = 0  # type: ignore[attr-defined]  # reset the scripted interviewer
    with TestClient(create_app(services)) as test_client:
        yield test_client


def _start(client: TestClient, path: Path = WORKPULSE) -> str:
    with path.open("rb") as fh:
        response = client.post("/api/sessions", files={"file": (path.name, fh, "application/pdf")})
    assert response.status_code == 201, response.text
    return str(response.json()["session_id"])


def _complete(client: TestClient) -> str:
    sid = _start(client)
    client.post(f"/api/sessions/{sid}/answers", json={"answers": [{"question_id": "r1-q1", "answer": "No"}]})
    assert client.post(f"/api/sessions/{sid}/audit").status_code == 202
    return sid


class TestPages:
    def test_overview_renders_calendar_and_security_headers(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "The regulatory calendar" in response.text
        assert "02 Dec 2027" in response.text
        assert "default-src 'self'" in response.headers["content-security-policy"]
        assert response.headers["x-frame-options"] == "DENY"

    def test_assessment_start_page(self, client: TestClient) -> None:
        assert "Start the interview" in client.get("/assessment").text

    def test_unknown_session_gives_html_404(self, client: TestClient) -> None:
        response = client.get("/report/unknown", headers=HTML)
        assert response.status_code == 404
        assert "Page not found" in response.text

    def test_static_assets_are_local(self, client: TestClient) -> None:
        assert client.get("/static/css/app.css").status_code == 200
        assert client.get("/static/fonts/ibm-plex-sans-latin-400-normal.woff2").status_code == 200


class TestInterview:
    def test_upload_starts_an_interview(self, client: TestClient) -> None:
        sid = _start(client)
        view = client.get(f"/api/sessions/{sid}").json()
        assert view["status"] == "interviewing"
        assert view["profile"]["project_name"] == "WorkPulse 360"
        assert [q["question_id"] for q in view["questions"]] == ["r1-q1", "r1-q2", "r1-q3", "r1-q4"]
        page = client.get(f"/assessment/{sid}").text
        assert "Send answers" in page
        assert "Is emotion analysis switched on" in page

    def test_text_only_session(self, client: TestClient) -> None:
        response = client.post(
            "/api/sessions",
            data={"description": "An HR tool that ranks job candidates from their CVs automatically."},
        )
        assert response.status_code == 201

    @pytest.mark.parametrize(
        ("files", "data", "message"),
        [
            (None, {"description": ""}, "Upload a document"),
            (None, {"description": "too short"}, "at least"),
            ({"file": ("notes.txt", b"hello", "text/plain")}, None, "Unsupported file type"),
            ({"file": ("fake.pdf", b"<html></html>", "application/pdf")}, None, "signature"),
        ],
    )
    def test_invalid_inputs_are_rejected(
        self,
        client: TestClient,
        files: dict[str, tuple[str, bytes, str]] | None,
        data: dict[str, str] | None,
        message: str,
    ) -> None:
        response = client.post("/api/sessions", files=files, data=data)
        assert response.status_code == 422
        assert message.lower() in response.json()["detail"].lower()

    def test_answers_close_the_interview(self, client: TestClient) -> None:
        sid = _start(client)
        response = client.post(
            f"/api/sessions/{sid}/answers",
            json={
                "answers": [
                    {"question_id": "r1-q1", "answer": "No"},
                    {"question_id": "r1-q2", "answer": "On for everyone"},
                ]
            },
        )
        assert response.json()["status"] == "ready"  # the scripted interviewer is satisfied after round 1
        assert "Write the report" in client.get(f"/assessment/{sid}").text
        closed = client.post(f"/api/sessions/{sid}/answers", json={"answers": []})
        assert closed.status_code == 409

    def test_answers_payload_is_validated(self, client: TestClient) -> None:
        sid = _start(client)
        bad = client.post(
            f"/api/sessions/{sid}/answers", json={"answers": [{"question_id": "q1", "answer": "x"}]}
        )
        assert bad.status_code == 422


class TestAuditAndReport:
    def test_full_flow(self, client: TestClient) -> None:
        sid = _complete(client)
        events = client.get(f"/api/sessions/{sid}/events").text
        assert events.count("event: progress") >= 8
        assert "event: done" in events

        report = client.get(f"/api/sessions/{sid}/report").json()
        assert report["ai_act"]["risk_category"] == "prohibited"
        assert report["compliance_score"] <= 10
        assert report["recommendations"][0]["priority"] == "P0"

        page = client.get(f"/report/{sid}").text
        assert "Prohibited practice" in page
        assert "Compliance checklist" in page
        assert client.get(f"/assessment/{sid}", follow_redirects=False).status_code == 303

    def test_checklist_and_exports(self, client: TestClient) -> None:
        sid = _complete(client)
        assert (
            client.patch(f"/api/sessions/{sid}/checklist/gdpr-dpia", json={"done": True}).status_code == 200
        )
        assert client.patch(f"/api/sessions/{sid}/checklist/unknown", json={"done": True}).status_code == 404
        pdf = client.get(f"/api/sessions/{sid}/report.pdf")
        assert pdf.content.startswith(b"%PDF")
        assert "attachment" in pdf.headers["content-disposition"]
        exported = client.get(f"/api/sessions/{sid}/report.json").json()
        assert next(i for i in exported["checklist"] if i["item_id"] == "gdpr-dpia")["done"] is True

        reopened = client.post(
            "/api/reports/import",
            files={"file": ("r.json", json.dumps(exported).encode(), "application/json")},
        )
        assert reopened.status_code == 201
        assert reopened.json()["status"] == "done"

    def test_chat_streams_with_sources(self, client: TestClient) -> None:
        sid = _complete(client)
        response = client.post(f"/api/sessions/{sid}/chat", json={"question": "What should we do first?"})
        assert response.status_code == 200
        assert "prohibited practices" in response.text
        assert json.loads(response.headers["x-sources"])
        assert "What should we do first?" in client.get(f"/report/{sid}").text  # history kept

    def test_report_is_not_available_before_the_audit(self, client: TestClient) -> None:
        sid = _start(client)
        assert client.get(f"/api/sessions/{sid}/report").status_code == 409
        assert client.get(f"/report/{sid}", follow_redirects=False).status_code == 303

    def test_health(self, client: TestClient) -> None:
        health = client.get("/api/health").json()
        assert health["ollama"]["reachable"] is True
        assert health["store"]["points"] > 200
