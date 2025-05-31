"""Main entry point for the MCP RAG Context Server."""

import logging
from .config import settings

from mcp.server.fastmcp import Context

from .models import LoadTopicResponse, QueryTopicResponse
from .server import create_mcp_server
from .tools import (
    LoadTopicToolArgs,
    QueryTopicToolArgs,
    research_topic,
    query_topic,
)

logger = logging.getLogger(__name__)
logger.info(f"Starting MCP RAG Context Server with settings: {settings}")
mcp_server = create_mcp_server()


@mcp_server.tool(
    name="research_topic",
    description="Researches information about a given topic from Jina, Firecrawl, and Perplexity, then indexes it into Milvus.",
)
async def research_topic_tool(
    args: LoadTopicToolArgs, ctx: Context
) -> LoadTopicResponse:
    """MCP Tool to research and index topic information."""
    app_ctx = ctx.request_context.lifespan_context
    if app_ctx is None:
        raise RuntimeError("Application context not available")
    return await research_topic(args, app_ctx)


@mcp_server.tool(
    name="query_topic",
    description="Queries Milvus for context relevant to the given query string.",
)
async def query_topic_tool(
    args: QueryTopicToolArgs, ctx: Context
) -> QueryTopicResponse:
    """MCP Tool to query indexed topic context."""
    app_ctx = ctx.request_context.lifespan_context
    if app_ctx is None:
        raise RuntimeError("Application context not available")
    return await query_topic(args, app_ctx)


def main() -> None:
    """Main entry point for the MCP RAG Context Server."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting MCP RAG Context Server...")
    mcp_server.run()


if __name__ == "__main__":
    main()
