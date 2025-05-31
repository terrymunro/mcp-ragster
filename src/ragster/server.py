"""MCP server configuration and lifespan management."""

import asyncio
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any
from pydantic import BaseModel
import httpx
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import tomllib

from mcp.server.fastmcp import FastMCP

from .config import settings
from .embedding_client import EmbeddingClient
from .milvus_ops import MilvusOperator
from .client_jina import JinaAPIClient
from .client_perplexity import PerplexityAPIClient
from .client_firecrawl import FirecrawlAPIClient
from .job_manager import JobManager
from .background_processor import BackgroundTaskProcessor
from .resource_manager import MultiTopicResourceManager
from .result_cache import TopicResultCache


logger = logging.getLogger("mcp_rag_server")


def get_package_version() -> str:
    """Get the package version from multiple sources with fallbacks."""
    # Try 1: Get from installed package metadata
    try:
        return version("mcp-ragster")
    except PackageNotFoundError:
        pass

    # Try 2: Read from pyproject.toml
    try:
        # Look for pyproject.toml in the project root
        current_file = Path(__file__)
        project_root = (
            current_file.parent.parent.parent
        )  # src/ragster/server.py -> project root
        pyproject_path = project_root / "pyproject.toml"

        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                pyproject_data = tomllib.load(f)
                return pyproject_data.get("project", {}).get("version", "unknown")
    except Exception as e:
        logger.warning(f"Could not read version from pyproject.toml: {e}")

    # Fallback: Static version
    return "0.4.0"


class AppContext(BaseModel):
    """Application context for lifespan management."""

    model_config = {"arbitrary_types_allowed": True}

    embedding_client: EmbeddingClient
    milvus_operator: MilvusOperator
    jina_client: JinaAPIClient
    perplexity_client: PerplexityAPIClient
    firecrawl_client: FirecrawlAPIClient
    http_client: httpx.AsyncClient
    job_manager: JobManager
    background_processor: BackgroundTaskProcessor
    resource_manager: MultiTopicResourceManager
    result_cache: TopicResultCache


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context."""
    logger.info("MCP Server lifespan startup: Initializing clients...")

    clients: dict[str, Any] = {}

    try:
        # Create persistent HTTP client with connection pooling first
        clients["http_client"] = httpx.AsyncClient(
            timeout=httpx.Timeout(
                settings.HTTP_TIMEOUT_DEFAULT, connect=settings.HTTP_TIMEOUT_CONNECT
            ),
            limits=httpx.Limits(
                max_keepalive_connections=settings.HTTP_MAX_KEEPALIVE_CONNECTIONS,
                max_connections=settings.HTTP_MAX_CONNECTIONS,
            ),
        )

        clients["embedding_client"] = EmbeddingClient()
        clients["milvus_operator"] = MilvusOperator()
        clients["milvus_operator"].load_collection()
        clients["jina_client"] = JinaAPIClient(http_client=clients["http_client"])
        clients["perplexity_client"] = PerplexityAPIClient(
            http_client=clients["http_client"]
        )
        clients["firecrawl_client"] = FirecrawlAPIClient()

        # Initialize job management components
        clients["job_manager"] = JobManager()
        clients["background_processor"] = BackgroundTaskProcessor(
            clients["job_manager"]
        )
        clients["resource_manager"] = MultiTopicResourceManager()
        clients["result_cache"] = TopicResultCache(
            max_size=100, ttl_hours=settings.JOB_CACHE_TTL_HOURS
        )

        app_ctx = AppContext(
            embedding_client=clients["embedding_client"],
            milvus_operator=clients["milvus_operator"],
            jina_client=clients["jina_client"],
            perplexity_client=clients["perplexity_client"],
            firecrawl_client=clients["firecrawl_client"],
            http_client=clients["http_client"],
            job_manager=clients["job_manager"],
            background_processor=clients["background_processor"],
            resource_manager=clients["resource_manager"],
            result_cache=clients["result_cache"],
        )
        logger.info("All clients initialized successfully.")

        # Start lifecycle services
        await app_ctx.job_manager.start()
        await app_ctx.result_cache.start()

        # Perform index warm-up if enabled and data exists
        if settings.ENABLE_INDEX_WARMUP:
            await _perform_index_warmup(app_ctx)

        yield app_ctx

    except Exception as e:
        logger.critical(
            f"Failed to initialize clients during MCP lifespan startup: {e}",
            exc_info=True,
        )
        raise RuntimeError(f"Lifespan initialization failed: {e}") from e

    finally:
        logger.info("MCP Server lifespan shutdown: Cleaning up clients...")

        # Shutdown background processor first
        if "background_processor" in clients:
            try:
                await clients["background_processor"].shutdown()
            except Exception as e:
                logger.error(f"Error shutting down background processor: {e}")

        # Shutdown job manager
        if "job_manager" in clients:
            try:
                await clients["job_manager"].stop()
            except Exception as e:
                logger.error(f"Error shutting down job manager: {e}")

        # Shutdown result cache
        if "result_cache" in clients:
            try:
                await clients["result_cache"].stop()
            except Exception as e:
                logger.error(f"Error shutting down result cache: {e}")

        await _cleanup_clients(clients)
        logger.info("Client cleanup complete.")


async def _perform_index_warmup(app_ctx: AppContext) -> None:
    """Perform smart index warm-up using stored data."""
    try:
        has_data = app_ctx.milvus_operator.has_data()
        if not has_data:
            logger.info("Index warm-up skipped: No data in collection")
            return

        logger.info("Starting index warm-up...")
        stored_topics = app_ctx.milvus_operator.get_stored_topics(limit=5)

        if not stored_topics:
            logger.info("Index warm-up skipped: No topics found")
            return

        # Warm up with stored topic variations
        warmup_queries: list[str] = []
        for topic in stored_topics:
            warmup_queries.extend(
                [topic, f"overview of {topic}", f"examples of {topic}"]
            )

        # Limit to 10 warm-up queries
        warmup_queries = warmup_queries[:10]

        from .embedding_client import VoyageInputType

        voyage_query_type: VoyageInputType = "query"

        for i, query in enumerate(warmup_queries):
            try:
                query_vector = await app_ctx.embedding_client.embed_texts(
                    query, input_type=voyage_query_type
                )
                app_ctx.milvus_operator.query_data(query_vector[0], top_k=3)
                logger.debug(
                    f"Warm-up query {i + 1}/{len(warmup_queries)}: {query[:30]}..."
                )
            except Exception as e:
                logger.warning(f"Warm-up query failed: {e}")

        logger.info(f"Index warm-up completed with {len(warmup_queries)} queries")

    except Exception as e:
        logger.error(f"Index warm-up failed: {e}")


async def _cleanup_clients(clients: dict[str, Any]) -> None:
    """Clean up all initialized clients."""
    cleanup_order = [
        "http_client",
        "embedding_client",
        "jina_client",
        "perplexity_client",
        "firecrawl_client",
        "milvus_operator",
        "job_manager",
        "background_processor",
    ]

    for client_name in cleanup_order:
        if client_name not in clients:
            continue

        client = clients[client_name]
        try:
            if client_name == "http_client" and not client.is_closed:
                await client.aclose()
            elif hasattr(client, "close_voyage_client"):
                await client.close_voyage_client()
            elif hasattr(client, "close"):
                if asyncio.iscoroutinefunction(client.close):
                    await client.close()
                else:
                    client.close()
        except Exception as e:
            logger.error(f"Error closing {client_name}: {e}")


def create_mcp_server() -> FastMCP:
    """Create and configure the MCP server."""
    return FastMCP(
        name="Ragster",
        title="Ragster the RAG Context Server",
        description="Provides tools to load and query topic-related context for RAG applications using external services.",
        version=get_package_version(),
        lifespan=app_lifespan,
    )
