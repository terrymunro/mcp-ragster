# Agent Instructions

## Running Tests and Linters

This project relies on **uv** for dependency management. Install the locked dependencies and run all checks using the commands below:

```bash
uv sync        # install from uv.lock
uv run ruff check .
uv run pyright
uv run pytest -q
```

If a `justfile` is present you may also use `just` wrappers, but by default run the `uv` commands directly.

## Useful Notes

- Python 3.13 is required and a local `.venv` directory is used by `uv`.
- `tests/conftest.py` sets default environment variables so tests work without real API keys.
- Avoid using `# type: ignore`; prefer accurate typing or `object` as described in `CLAUDE.md`.
