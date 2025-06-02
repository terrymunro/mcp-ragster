"""Main entry point for the MCP RAG Context Server."""

import logging

from mcp.server.fastmcp import Context

from .config import settings
from .models import LoadTopicResponse, QueryTopicResponse
from .job_models import (
    GetJobStatusArgs,
    JobStatusResponse,
    ListJobsArgs,
    ListJobsResponse,
    CancelJobArgs,
    CancelJobResponse,
    ResearchJobResponse,
)
from .server import create_mcp_server
from .tools import (
    LoadTopicToolArgs,
    QueryTopicToolArgs,
    research_topic,
    query_topic,
    get_research_status,
    list_research_jobs,
    cancel_research_job,
)

logger = logging.getLogger(__name__)
logger.info(f"Starting MCP RAG Context Server with settings: {settings}")
mcp_server = create_mcp_server()


@mcp_server.tool(
    name="research_topic",
    description="Research and index topic information asynchronously with intelligent caching and resource management. Supports 1-10 topics per job with automatic job tracking for multi-topic requests.",
)
async def research_topic_tool(
    args: LoadTopicToolArgs, ctx: Context
) -> ResearchJobResponse | LoadTopicResponse:
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


@mcp_server.tool(
    name="get_research_status",
    description="Get the status and progress of a research job.",
)
async def get_research_status_tool(
    args: GetJobStatusArgs, ctx: Context
) -> JobStatusResponse:
    """MCP Tool to get research job status."""
    app_ctx = ctx.request_context.lifespan_context
    if app_ctx is None:
        raise RuntimeError("Application context not available")
    return await get_research_status(args, app_ctx)


@mcp_server.tool(
    name="list_research_jobs",
    description="List research jobs with optional status filtering and pagination.",
)
async def list_research_jobs_tool(args: ListJobsArgs, ctx: Context) -> ListJobsResponse:
    """MCP Tool to list research jobs."""
    app_ctx = ctx.request_context.lifespan_context
    if app_ctx is None:
        raise RuntimeError("Application context not available")
    return await list_research_jobs(args, app_ctx)


@mcp_server.tool(
    name="cancel_research_job",
    description="Cancel a running research job, preserving any completed topic results.",
)
async def cancel_research_job_tool(
    args: CancelJobArgs, ctx: Context
) -> CancelJobResponse:
    """MCP Tool to cancel a research job."""
    app_ctx = ctx.request_context.lifespan_context
    if app_ctx is None:
        raise RuntimeError("Application context not available")
    return await cancel_research_job(args, app_ctx)


def main() -> None:
    """Main entry point for the MCP RAG Context Server."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting MCP RAG Context Server...")
    mcp_server.run()


if __name__ == "__main__":
    main()
