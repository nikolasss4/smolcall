.PHONY: install test lint

install:  ## Python deps, including MLX (Apple Silicon only)
	uv sync --all-extras

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
