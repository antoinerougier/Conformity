"""ASGI entry point of the offline demo (scripted model): ``uvicorn scripts.demo_app:app``.

Lets anyone click through the interface without Ollama. The answers are scripted for the
WorkPulse 360 example (``examples/workpulse360_high_risk_project.pdf``) whatever the upload.
"""

from __future__ import annotations

from scripts.demo_services import build_demo_services
from src.api.app import create_app

app = create_app(build_demo_services(delay_s=0.8))
