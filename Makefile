.PHONY: install lint format typecheck test check eval preprocess index dataset run demo screenshots examples up down

install:      ## Install every dependency group (dev + eval) and the git hooks
	uv sync --all-groups
	uv run pre-commit install

lint:         ## Ruff (lint + format check)
	uv run ruff check .
	uv run ruff format --check .

format:       ## Format the code
	uv run ruff format .
	uv run ruff check --fix .

typecheck:    ## Strict mypy
	uv run mypy src tests scripts

test:         ## Unit and integration tests (no external service)
	uv run pytest --cov=src

check: lint typecheck test  ## Everything the CI runs (except Docker and Ragas)

eval:         ## Ragas evaluation (requires Ollama + Qdrant index)
	uv run pytest tests/eval -m eval -v --no-cov

preprocess:   ## Legal splitting -> data/processed/legal_chunks.jsonl
	uv run python -m src.rag.preprocess_laws --stats

index:        ## Full Qdrant re-indexing
	uv run python -m src.rag.preprocess_laws --index --recreate

dataset:      ## Generate the synthetic Ragas test set
	uv run python -m src.evaluation.dataset --per-source 10

run:          ## Start the web application on http://127.0.0.1:8000
	uv run uvicorn src.api.app:app --reload

demo:         ## Start the web application with the scripted demo model (no Ollama needed)
	uv run uvicorn scripts.demo_app:app --port 8000

screenshots:  ## Re-capture the README screenshots (offline demo)
	uv run python scripts/capture_screenshots.py

examples:     ## Regenerate the example project PDFs
	uv run python scripts/make_example_projects.py

up:           ## Full Docker stack (app + Qdrant + Ollama)
	docker compose up -d --build

down:
	docker compose down
