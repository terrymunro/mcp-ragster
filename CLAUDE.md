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
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install "mcp[cli]>=0.2.0"
uv pip install .
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
- `external_apis.py`: Clients for Jina, Firecrawl, and Perplexity APIs
- `embedding_client.py`: Voyage AI embedding interface
- `milvus_ops.py`: Vector database operations
- `config.py`: Environment-based configuration with validation
- `models.py`: Pydantic models for request/response data structures
- `exceptions.py`: Custom exception hierarchy

**Refactoring Benefits:**
- Separated concerns for better maintainability
- Python 3.13 syntax and type annotations throughout
- Testable components with clear boundaries
- 80% reduction in main.py size (269 → 50 lines)

**Error Handling:**
All external API calls use custom exceptions (`JinaAPIError`, `PerplexityAPIError`, `FirecrawlError`) that inherit from `MCPError`. MCP tools properly raise exceptions for FastMCP compliance instead of returning error objects.

**Concurrency:**
The `load_topic_context` tool uses asyncio.gather() to run Perplexity and Firecrawl operations concurrently after initial Jina search completes. All Milvus operations use async wrappers to avoid blocking the event loop.

**Performance Optimizations:**
- Persistent HTTP client with connection pooling for all external API calls
- Async-wrapped Milvus operations to prevent event loop blocking
- Proper MCP error handling for fast failure propagation

**Type Safety:**
Uses relative imports with fallback for direct execution. Pydantic models provide validation for all API interfaces and tool arguments.

## Project Philosophy

- Don't treat this project like normal software, treat it like a personal developer tool
- Prioritize performance, efficiency, and LLM, MCP, Agents / Agentic related best practices

## Coding Principles

- Don't make assumptions and claim things without evidence to support your claim. (e.g. "x is now significantly more responsive" if you didn't verify this with a benchmark or test then don't say it definitively)

## Language and Type Considerations

- Use Python 3.13 syntax and types