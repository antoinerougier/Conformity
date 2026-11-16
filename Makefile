.PHONY: install lint format typecheck test check eval preprocess index dataset run up down

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
	uv run mypy src app.py pages tests

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

run:          ## Start the Streamlit UI locally
	uv run streamlit run app.py

up:           ## Full Docker stack (app + Qdrant + Ollama)
	docker compose up -d --build

down:
	docker compose down
