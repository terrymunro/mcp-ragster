#!/usr/bin/env python3
"""Test script for query_topic_context functionality."""

import asyncio
import logging
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ragster.tools import QueryTopicToolArgs, query_topic_context
from ragster.server import AppContext
from ragster.config import settings
from ragster.embedding_client import EmbeddingClient
from ragster.milvus_ops import MilvusOperator
from ragster.client_jina import JinaAPIClient
from ragster.client_perplexity import PerplexityAPIClient
from ragster.client_firecrawl import FirecrawlAPIClient
import httpx

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_query():
    """Test the query functionality directly."""
    try:
        # Create app context manually
        logger.info("Creating app context...")
        
        # Create HTTP client first
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                settings.HTTP_TIMEOUT_DEFAULT, 
                connect=settings.HTTP_TIMEOUT_CONNECT
            ),
            limits=httpx.Limits(
                max_keepalive_connections=settings.HTTP_MAX_KEEPALIVE_CONNECTIONS,
                max_connections=settings.HTTP_MAX_CONNECTIONS,
            ),
        )
        
        # Initialize clients
        embedding_client = EmbeddingClient()
        milvus_operator = MilvusOperator()
        milvus_operator.load_collection()
        jina_client = JinaAPIClient(http_client=http_client)
        perplexity_client = PerplexityAPIClient(http_client=http_client)
        firecrawl_client = FirecrawlAPIClient()
        
        app_context = AppContext(
            embedding_client=embedding_client,
            milvus_operator=milvus_operator,
            jina_client=jina_client,
            perplexity_client=perplexity_client,
            firecrawl_client=firecrawl_client,
            http_client=http_client,
        )
        
        # Test query
        query_args = QueryTopicToolArgs(
            query="How to use Databricks Asset Bundles with Python packages and wheels",
            top_k=3
        )
        
        logger.info(f"Executing query: {query_args.query}")
        result = await query_topic_context(query_args, app_context)
        
        logger.info("Query result:")
        logger.info(f"  Query: {result.query}")
        logger.info(f"  Results count: {len(result.results)}")
        logger.info(f"  Message: {result.message}")
        
        for i, doc in enumerate(result.results):
            logger.info(f"  Result {i+1}:")
            logger.info(f"    Source: {doc.source_type} - {doc.source_identifier}")
            logger.info(f"    Topic: {doc.topic}")
            logger.info(f"    Distance: {doc.distance}")
            logger.info(f"    Text (first 200 chars): {doc.text_content[:200]}...")
            
        # Check if Milvus has any data
        has_data = app_context.milvus_operator.has_data()
        logger.info(f"Milvus has data: {has_data}")
        
        if has_data:
            topics = app_context.milvus_operator.get_stored_topics(limit=5)
            logger.info(f"Stored topics: {topics}")
        
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
    finally:
        if 'http_client' in locals():
            try:
                await http_client.aclose()
            except Exception as e:
                logger.error(f"Error closing HTTP client: {e}")
        if 'milvus_operator' in locals():
            try:
                milvus_operator.close()
            except Exception as e:
                logger.error(f"Error closing Milvus operator: {e}")

if __name__ == "__main__":
    asyncio.run(test_query()) 