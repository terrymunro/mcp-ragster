"""Main entry point for the MCP RAG Context Server."""

import logging

from mcp.server.fastmcp import Context

from .config import settings
from .models import LoadTopicResponse, QueryTopicResponse
from .server import create_mcp_server
from .tools import (
    LoadTopicToolArgs,
    QueryTopicToolArgs,
    load_topic_context,
    query_topic_context,
)

logger = logging.getLogger(__name__)
logger.info("Starting MCP RAG Context Server...")
mcp_server = create_mcp_server()


@mcp_server.tool(
    name="load_topic_context",
    description="Loads information about a given topic from Jina, Firecrawl, and Perplexity, then indexes it into Milvus.",
)
async def load_topic_tool(args: LoadTopicToolArgs, ctx: Context) -> LoadTopicResponse:
    """MCP Tool to load and index topic context."""
    app_ctx = getattr(ctx, 'lifespan', None)
    if app_ctx is None:
        raise RuntimeError("Application context not available")
    return await load_topic_context(args, app_ctx)


@mcp_server.tool(
    name="query_topic_context",
    description="Queries Milvus for context relevant to the given query string.",
)
async def query_topic_tool(
    args: QueryTopicToolArgs, ctx: Context
) -> QueryTopicResponse:
    """MCP Tool to query indexed topic context."""
    app_ctx = getattr(ctx, 'lifespan', None)
    if app_ctx is None:
        raise RuntimeError("Application context not available")
    return await query_topic_context(args, app_ctx)


def main() -> None:
    """Main entry point for the MCP RAG Context Server."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting MCP RAG Context Server...")
    mcp_server.run()


if __name__ == "__main__":
    main()
