# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Development Mode:**

```bash
mcp dev src/ragster/main.py
```

**Direct Execution:**

```bash
python src/ragster/main.py
```

**Install Dependencies:**

```bash
uv sync
```

## Environment Setup

Copy `.env.example` to `.env` and configure all required API keys:

- `VOYAGEAI_API_KEY` (required)
- `PERPLEXITY_API_KEY` (required)
- `JINA_API_KEY` (required)
- Either `FIRECRAWL_API_KEY` or `FIRECRAWL_API_URL` (required)

Ensure `EMBEDDING_DIMENSION` matches your `VOYAGEAI_MODEL_NAME` (config.py handles known models automatically).

## Architecture Overview

**Core Flow:**

1. `load_topic_context` tool orchestrates data gathering from external APIs
2. Concurrently fetches from Jina (search snippets), Perplexity (summary), and Firecrawl (full content)
3. Content is embedded via Voyage AI and stored in Milvus vector database
4. `query_topic_context` tool performs semantic search against stored embeddings

**Key Components:**

- `main.py`: Entry point and MCP tool registration (50 lines)
- `server.py`: FastMCP server and lifespan management
- `tools.py`: Tool implementations with composable functions
- `external_apis.py`: Clients for Jina, Firecrawl, and Perplexity APIs with connection pooling
- `embedding_client.py`: Voyage AI embedding interface with LRU caching
- `milvus_ops.py`: Vector database operations with connection pooling and health checks
- `config.py`: Environment-based configuration with validation
- `models.py`: Pydantic models for request/response data structures
- `exceptions.py`: Custom exception hierarchy

**Performance Optimizations:**

- **Persistent HTTP clients** with connection pooling for all external API calls
- **LRU caching** for embeddings (avoid redundant API calls) and Jina search responses (1-hour TTL)
- **Progressive content processing** - embed and store content as it arrives using asyncio.as_completed()
- **Concurrent Firecrawl processing** with domain-based batching and configurable rate limiting
- **Async-native Milvus operations** with connection pooling to prevent event loop blocking
- **HNSW index warm-up** on startup for optimal search performance
- **Dynamic search parameters** - exploration vs precise modes with different ef values
- **URL deduplication cache** with LRU and 1-hour TTL to avoid redundant crawling
- **Exponential backoff retry** for failed external API calls

**Error Handling:**
All external API calls use custom exceptions (`JinaAPIError`, `PerplexityAPIError`, `FirecrawlError`) that inherit from `MCPError`. MCP tools properly raise exceptions for FastMCP compliance instead of returning error objects.

**Type Safety:**
Uses relative imports with fallback for direct execution. Pydantic models provide validation for all API interfaces and tool arguments.

## Project Philosophy

- Don't treat this project like normal software, treat it like a personal developer tool
- Prioritize performance, efficiency, and LLM, MCP, Agents / Agentic related best practices

## Coding Principles

- Don't make assumptions and claim things without evidence to support your claim. (e.g. "x is now significantly more responsive" if you didn't verify this with a benchmark or test then don't say it definitively)

## Language and Type Considerations

- Use Python 3.13 syntax and types
- **NEVER** use `# type: ignore` - Instead try to understand the type error properly and either adjust the types if necessary or if its not possible, fall back to using `object` if that isn't sufficient only then fall back to `Any`
  - To re-iterate the priority is:
    1. Accurate Types!
    2. Use type `object`
    3. Use type `Any`
  - **ALWAYS** try to maximise the priority
  - **Remember:** The type checker is always smarter than you when it comes to *types*. Trust the type checker! Work with it, not against it.
- Use `uv` to run linters and formatters:
  - `uv run pyright` for type checking
  - `uv run ruff check .` for linting
  - `uv run ruff format .` for formatting
