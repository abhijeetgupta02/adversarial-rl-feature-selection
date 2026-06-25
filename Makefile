.PHONY: test lint typecheck reproduce-smoke reproduce-results
test:
	uv run pytest
lint:
	uv run ruff check .
typecheck:
	uv run mypy
reproduce-smoke:
	PYTHONPATH=src uv run python -m robustfeatures.run --smoke
reproduce-results:
	PYTHONPATH=src uv run python -m robustfeatures.run
