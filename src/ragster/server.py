"""MCP server configuration and lifespan management."""

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from pydantic import BaseModel
import httpx

from mcp.server.fastmcp import FastMCP, Context

from .config import settings
from .embedding_client import EmbeddingClient
from .milvus_ops import MilvusOperator
from .external_apis import ExternalAPIClient

logger = logging.getLogger("mcp_rag_server")


class AppContext(BaseModel):
    """Application context for lifespan management."""
    model_config = {"arbitrary_types_allowed": True}
    
    embedding_client: EmbeddingClient
    milvus_operator: MilvusOperator
    external_api_client: ExternalAPIClient
    http_client: httpx.AsyncClient


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context."""
    logger.info("MCP Server lifespan startup: Initializing clients...")
    
    clients: dict[str, any] = {}
    
    try:
        # Create persistent HTTP client with connection pooling first
        clients['http_client'] = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        
        clients['embedding_client'] = EmbeddingClient()
        clients['milvus_operator'] = MilvusOperator()
        await clients['milvus_operator'].connect_and_load()
        clients['external_api_client'] = ExternalAPIClient(
            http_client=clients['http_client']
        )
        
        app_ctx = AppContext(
            embedding_client=clients['embedding_client'],
            milvus_operator=clients['milvus_operator'],
            external_api_client=clients['external_api_client'],
            http_client=clients['http_client']
        )
        logger.info("All clients initialized successfully.")
        
        # Perform index warm-up if enabled and data exists
        if settings.ENABLE_INDEX_WARMUP:
            await _perform_index_warmup(app_ctx)
        
        yield app_ctx
        
    except Exception as e:
        logger.critical(f"Failed to initialize clients during MCP lifespan startup: {e}", exc_info=True)
        raise RuntimeError(f"Lifespan initialization failed: {e}") from e
        
    finally:
        logger.info("MCP Server lifespan shutdown: Cleaning up clients...")
        await _cleanup_clients(clients)
        logger.info("Client cleanup complete.")


async def _perform_index_warmup(app_ctx: AppContext) -> None:
    """Perform smart index warm-up using stored data."""
    try:
        has_data = await app_ctx.milvus_operator.has_data()
        if not has_data:
            logger.info("Index warm-up skipped: No data in collection")
            return
        
        logger.info("Starting index warm-up...")
        stored_topics = await app_ctx.milvus_operator.get_stored_topics(limit=5)
        
        if not stored_topics:
            logger.info("Index warm-up skipped: No topics found")
            return
        
        # Warm up with stored topic variations
        warmup_queries = []
        for topic in stored_topics:
            warmup_queries.extend([
                topic,
                f"overview of {topic}",
                f"examples of {topic}"
            ])
        
        # Limit to 10 warm-up queries
        warmup_queries = warmup_queries[:10]
        
        from .embedding_client import VoyageInputType
        voyage_query_type = settings.VOYAGEAI_INPUT_TYPE_QUERY
        
        for i, query in enumerate(warmup_queries):
            try:
                embedding = await app_ctx.embedding_client.embed_texts(
                    query, input_type=voyage_query_type
                )
                await app_ctx.milvus_operator.query_data(embedding, top_k=3)
                logger.debug(f"Warm-up query {i+1}/{len(warmup_queries)}: {query[:30]}...")
            except Exception as e:
                logger.warning(f"Warm-up query failed: {e}")
        
        logger.info(f"Index warm-up completed with {len(warmup_queries)} queries")
        
    except Exception as e:
        logger.error(f"Index warm-up failed: {e}")


async def _cleanup_clients(clients: dict[str, any]) -> None:
    """Clean up all initialized clients."""
    cleanup_order = ['http_client', 'embedding_client', 'external_api_client', 'milvus_operator']
    
    for client_name in cleanup_order:
        if client_name not in clients:
            continue
            
        client = clients[client_name]
        try:
            if client_name == 'http_client' and not client.is_closed:
                await client.aclose()
            elif hasattr(client, 'close_voyage_client'):
                await client.close_voyage_client()
            elif hasattr(client, 'close'):
                await client.close()
        except Exception as e:
            logger.error(f"Error closing {client_name}: {e}")


def create_mcp_server() -> FastMCP:
    """Create and configure the MCP server."""
    server = FastMCP(
        name="RAGContextServer",
        title="RAG Context Server",
        description="Provides tools to load and query topic-related context for RAG applications using external services.",
        version="0.4.0"
    )
    
    server.lifespan = app_lifespan
    return server