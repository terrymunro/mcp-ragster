[private]
default:
  @just --list


# Start the MCP server
start:
  uv run start-mcp-server


# Run linter
lint:
  uv run ruff check --unsafe-fixes .


# Run formmatter
fmt:
  uv run ruff format

# Run type checker
types:
  uv run pyright

# Run tests
tests:
  uv run pytest

# Run all checks
checks: fmt lint types tests
