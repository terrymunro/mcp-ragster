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

1. `research_topic` tool orchestrates data gathering from external APIs **asynchronously** for 1-10 topics per job
2. Returns immediately with a job ID; background tasks handle topic research and indexing
3. Use `get_research_status` to poll job progress and see per-topic status
4. Use `list_research_jobs` to enumerate jobs, filter by status, and paginate
5. Use `cancel_research_job` to stop a running job and preserve completed topic results
6. Content is embedded via Voyage AI and stored in Milvus vector database
7. `query_topic` tool performs semantic search against stored embeddings

**Key Components:**

- `main.py`: Entry point and MCP tool registration (see new async job tools)
- `server.py`: FastMCP server and lifespan management
- `tools.py`: Tool implementations, including async job management tools
- `job_manager.py`: Thread-safe, async job management, state transitions, and job CRUD
- `background_processor.py`: Async orchestration of multi-topic research jobs, cancellation, and progress callbacks
- `job_models.py`: Data models for jobs, progress, responses, and arguments
- `external_apis.py`: Clients for Jina, Firecrawl, and Perplexity APIs with connection pooling
- `embedding_client.py`: Voyage AI embedding interface with LRU caching
- `milvus_ops.py`: Vector database operations with connection pooling and health checks
- `config.py`: Environment-based configuration with validation
- `models.py`: Pydantic models for request/response data structures
- `exceptions.py`: Custom exception hierarchy

**Async Job Management Tools:**

- `research_topic`: Start a new async research job (1-10 topics)
- `get_research_status`: Poll for job status and per-topic progress
- `list_research_jobs`: List jobs, filter by status, paginate
- `cancel_research_job`: Cancel a running job, preserve completed results

**Async Workflow Example:**

1. Start a job with `research_topic` (returns job_id)
2. Poll with `get_research_status` until status is `completed` or `failed`
3. Use `list_research_jobs` to monitor all jobs
4. Use `cancel_research_job` to stop a job if needed
5. Query results with `query_topic` after job completion

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
- **NEVER** use `# type: ignore` (or any variant of disabling type checking) - Instead, try to understand the type error properly and either adjust the types if necessary or if its not possible, fall back to using `object` if that isn't sufficient only then fall back to `Any`
  - To re-iterate: the priority is:
    1. Use Accurate types whenever possible!
    2. When its impossible to know, because its dynamic, or the third-party library does not have stubs, try to use type `object`
    3. When that was not sufficient, and you don't have any choice then finally you can use type `Any` but add a FIXME comment as well
  - **ALWAYS** use the priority system for typing
  - **Remember:** The type checker is always smarter than you when it comes to _types_. Trust the type checker! Work with it, not against it.
  - If you come across a type ignore comment, remove it and fix the issue!
  - if you come across an Any that does not have a FIXME comment, remove the Any and try to set its type properly!
- Use `uv` to run linters and formatters:
  - `uv run pyright` for type checking
  - `uv run ruff check .` for linting
  - `uv run ruff format .` for formatting
