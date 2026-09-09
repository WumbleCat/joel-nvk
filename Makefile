.PHONY: help sync lock test cov lint format typecheck run clean

help:
	@echo "sync       Create/update .venv from uv.lock (installs dev group)"
	@echo "lock       Re-resolve dependencies and update uv.lock"
	@echo "test       Run the test suite"
	@echo "cov        Run the test suite with coverage"
	@echo "lint       Run ruff"
	@echo "format     Format with ruff"
	@echo "typecheck  Run mypy"
	@echo "run        Run the CLI (make run ARGS='--env dev config')"
	@echo "clean      Remove caches and build artefacts"

sync:
	uv sync

lock:
	uv lock

test:
	uv run pytest

cov:
	uv run pytest --cov=joel_nvk --cov-report=term-missing

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

run:
	uv run joel-nvk $(ARGS)

clean:
	uv run python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	uv run python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('build', 'dist', '.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov')]"
