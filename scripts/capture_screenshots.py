"""Capture the README screenshots by driving the web application in a headless browser.

Default (offline, reproducible): starts the real application wired to the scripted demo
model of ``scripts/demo_services.py`` and walks through the WorkPulse 360 example.

With real models: start the application yourself (``uvicorn src.api.app:app``) and run
    python scripts/capture_screenshots.py --url http://127.0.0.1:8000
The interview answers below are typed into whatever questions your model asks.

Requires Playwright and a Chromium-based browser (``playwright install chromium``, or an
installed Google Chrome, which is detected automatically).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import threading
import time
from pathlib import Path

import uvicorn
from playwright.sync_api import Browser, Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "workpulse360_high_risk_project.pdf"
VIEWPORT = {"width": 1440, "height": 900}
# Answers typed into the interview, matched on keywords of each question (first match wins).
ANSWERS: list[tuple[tuple[str, ...], str]] = [
    (
        ("dismissal", "manager", "review", "human"),
        "Managers only see the score. Letters go out automatically after ten days.",
    ),
    (
        ("insurer", "share", "third part", "recipient"),
        "A framework contract with a group-insurance provider; employees are not told.",
    ),
    (
        ("model", "train", "llm"),
        "A third-party LLM through its API. Prompts include employee data; training use is unknown.",
    ),
    (("retention", "kept", "delete"), "Everything is kept indefinitely to improve future models."),
]
DEFAULT_ANSWER = "Not documented yet."
# Options preferred for yes/no and choice questions, in order.
PREFERRED_OPTIONS = ("No", "On for everyone")
# Sticky elements would otherwise be stamped into full-page and element screenshots.
CAPTURE_CSS = ".site-header, .dossier, .toc, .answer-bar { position: static !important; } .chat-toggle { display: none !important; }"


def _answer_for(question: str) -> str:
    text = question.lower()
    return next((answer for keys, answer in ANSWERS if any(k in text for k in keys)), DEFAULT_ANSWER)


def _serve_demo(port: int) -> uvicorn.Server:
    from scripts.demo_services import build_demo_services
    from src.api.app import create_app

    app = create_app(build_demo_services(delay_s=0.9))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            return server
        time.sleep(0.1)
    raise RuntimeError("The demo server did not start")


def _launch(playwright: object) -> Browser:
    chromium = playwright.chromium  # type: ignore[attr-defined]
    chrome = shutil.which("google-chrome") or shutil.which("google-chrome-stable") or shutil.which("chromium")
    browser: Browser = chromium.launch(executable_path=chrome) if chrome else chromium.launch()
    return browser


def _answer_round(page: Page) -> None:
    for fieldset in page.locator("fieldset.question").all():
        question = fieldset.locator("legend").inner_text()
        options = fieldset.locator(".choices span")
        if options.count():
            preferred = [o for o in PREFERRED_OPTIONS if options.filter(has_text=o).count()]
            (options.filter(has_text=preferred[0]).first if preferred else options.first).click()
            fieldset.locator("textarea.detail").fill(_answer_for(question))
        else:
            fieldset.locator("textarea").fill(_answer_for(question))


def capture(base_url: str, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    shots: list[Path] = []

    def shot(page: Page, name: str, *, selector: str | None = None, full_page: bool = False) -> None:
        path = out / f"{name}.png"
        if selector or full_page:
            style = page.add_style_tag(content=CAPTURE_CSS)
            if selector:
                page.locator(selector).first.screenshot(path=str(path))
            else:
                page.screenshot(path=str(path), full_page=True)
            style.evaluate("node => node.remove()")
        else:
            page.evaluate("window.scrollTo(0, 0)")
            page.screenshot(path=str(path))
        shots.append(path)

    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport=VIEWPORT)  # type: ignore[arg-type]
        page.goto(base_url + "/")
        page.locator("#status-pill:not([data-state='unknown'])").wait_for()
        shot(page, "01-overview")
        shot(page, "02-calendar", selector="section[aria-labelledby='calendar-title']")

        page.goto(base_url + "/assessment")
        page.set_input_files("#file", str(EXAMPLE))
        page.fill("#description", "Pilot with two logistics customers in Lyon since September 2026.")
        shot(page, "03-upload")

        page.click("#start-button")
        page.wait_for_url("**/assessment/*", timeout=600_000)
        _answer_round(page)
        shot(page, "04-interview", full_page=True)

        while True:
            page.click("#answers-button")
            page.wait_for_load_state("networkidle", timeout=600_000)
            if page.locator("#audit-button").count():
                break
            _answer_round(page)
        shot(page, "05-ready")

        page.click("#audit-button")
        page.wait_for_url("**/report/*")
        page.wait_for_timeout(3500)
        shot(page, "06-progress")

        page.wait_for_selector(".verdict", timeout=1_800_000)
        page.locator("#status-pill:not([data-state='unknown'])").wait_for()
        shot(page, "07-report")
        shot(page, "08-ai-act", selector="#ai-act")
        shot(page, "09-action-plan", selector="#plan")
        for box in page.locator("#checklist input[type=checkbox]").all()[:3]:
            box.check()
        page.wait_for_timeout(300)
        shot(page, "10-checklist", selector="#checklist")

        page.click("#chat-toggle")
        page.locator("[data-suggest]").first.click()
        page.locator(".msg--assistant .sources").wait_for(timeout=600_000)
        page.wait_for_timeout(300)
        shot(page, "11-assistant")
        browser.close()
    return shots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", help="URL of a running application (real models). Default: offline demo")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "screenshots")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    server = None if args.url else _serve_demo(args.port)
    base_url = (args.url or f"http://127.0.0.1:{args.port}").rstrip("/")
    try:
        for path in capture(base_url, args.out):
            print(f"Saved {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")  # noqa: T201 - CLI
    finally:
        if server is not None:
            server.should_exit = True
    return 0


if __name__ == "__main__":
    sys.exit(main())
