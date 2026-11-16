"""Rendering of the Streamlit pages (AppTest): no external service, with and without a report."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.ui import services
from tests.conftest import sample_report

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "")
    monkeypatch.setenv("RERANKER_BACKEND", "none")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")


def _run(page: str, with_report: bool = False) -> AppTest:
    at = AppTest.from_file(str(ROOT / page), default_timeout=60)
    if with_report:
        report = sample_report()
        at.session_state[services.REPORT_KEY] = report
        at.session_state[services.CHECKLIST_KEY] = {i.item_id: False for i in report.checklist}
    return at.run()


def test_app_entrypoint_renders_presentation() -> None:
    at = _run("app.py")
    assert not at.exception
    assert at.title[0].value.startswith("⚖️")


def test_presentation_reports_unreachable_ollama() -> None:
    at = _run("pages/presentation.py")
    assert not at.exception
    assert any("Unreachable" in e.value for e in at.error)
    assert at.button[0].label.startswith("📚 Pre-index")


@pytest.mark.parametrize("page", ["pages/audit.py", "pages/export.py"])
def test_pages_without_report(page: str) -> None:
    at = _run(page)
    assert not at.exception
    assert at.info


def test_audit_page_displays_report() -> None:
    at = _run("pages/audit.py", with_report=True)
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Compliance score" in labels
    assert "AI Act category" in labels
    assert at.metric[0].value == "42/100"


def test_export_page_checklist_updates_state() -> None:
    at = _run("pages/export.py", with_report=True)
    assert not at.exception
    assert at.checkbox[0].value is False
    at.checkbox[0].check().run()
    assert not at.exception
    assert at.session_state[services.CHECKLIST_KEY]["gdpr-dpia"] is True
