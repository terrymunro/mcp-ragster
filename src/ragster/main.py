import asyncio
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator # For AsyncIterator type hint
from typing import List, Dict, Any, Optional # For type hints
from pydantic import BaseModel, Field # For tool argument and return type hints if complex
import httpx

from mcp.server.fastmcp import FastMCP, Context

# Relative imports for package structure
if __package__ or "." in __name__:
    from .config import settings
    from .models import LoadTopicResponse as SDKLoadTopicResponse, QueryTopicResponse as SDKQueryTopicResponse, DocumentFragment as SDKDocumentFragment
    from .embedding_client import EmbeddingClient, VoyageInputType
    from .milvus_ops import MilvusOperator
    from .external_apis import ExternalAPIClient
    from .exceptions import MCPError
else: # For potential direct execution IF sys.path is manipulated by runner
    from ragster.config import settings
    from ragster.models import LoadTopicResponse as SDKLoadTopicResponse, QueryTopicResponse as SDKQueryTopicResponse, DocumentFragment as SDKDocumentFragment
    from ragster.embedding_client import EmbeddingClient, VoyageInputType
    from ragster.milvus_ops import MilvusOperator
    from ragster.external_apis import ExternalAPIClient
    from ragster.exceptions import MCPError

logger = logging.getLogger("mcp_rag_server") # Main logger for the app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Define Application Context for lifespan management
class AppContext(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    
    embedding_client: EmbeddingClient
    milvus_operator: MilvusOperator
    external_api_client: ExternalAPIClient
    http_client: httpx.AsyncClient

# Create an MCP server instance
mcp_server = FastMCP(
    name="RAGContextServer",
    title="RAG Context Server",
    description="Provides tools to load and query topic-related context for RAG applications using external services.",
    version="0.4.0" # Match pyproject.toml
)

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context."""
    logger.info("MCP Server lifespan startup: Initializing clients...")
    
    # Initialize clients
    # These will raise exceptions from their constructors if critical setup fails (e.g. API key missing in config was not caught)
    try:
        # Create persistent HTTP client with connection pooling first
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        
        embedding_cli = EmbeddingClient()
        milvus_op = MilvusOperator()
        await milvus_op.connect_and_load() # Explicit connect and load for Milvus
        external_apis_cli = ExternalAPIClient(http_client=http_client)
        
        app_ctx = AppContext(
            embedding_client=embedding_cli,
            milvus_operator=milvus_op,
            external_api_client=external_apis_cli,
            http_client=http_client
        )
        logger.info("All clients initialized successfully.")
        yield app_ctx
    except Exception as e:
        logger.critical(f"Failed to initialize clients during MCP lifespan startup: {e}", exc_info=True)
        # Re-raise to prevent server from starting in a broken state
        # FastMCP might handle this or FastAPI underneath will.
        raise RuntimeError(f"Lifespan initialization failed: {e}") from e
    finally:
        logger.info("MCP Server lifespan shutdown: Cleaning up clients...")
        if 'http_client' in locals() and not http_client.is_closed:
            await http_client.aclose()
        if 'embedding_cli' in locals() and hasattr(embedding_cli, 'close_voyage_client'):
            await embedding_cli.close_voyage_client()
        if 'milvus_op' in locals() and hasattr(milvus_op, 'close'):
            await milvus_op.close() # If you implement an async close
        if 'external_apis_cli' in locals() and hasattr(external_apis_cli, 'close'):
            await external_apis_cli.close()
        logger.info("Client cleanup complete.")

# Assign lifespan to the MCP server
mcp_server.lifespan = app_lifespan


# Define MCP Tools
# We use Pydantic models for tool arguments if they are complex,
# or simple type hints for basic arguments.
# MCP SDK's FastMCP uses function signature for tool schema.

class LoadTopicToolArgs(BaseModel):
    topic: str = Field(..., min_length=1, description="The topic to load information about.")

# The return type of the tool should be a Pydantic model or a type that MCP can serialize.
# Let's use our existing SDKLoadTopicResponse.

@mcp_server.tool(
    name="load_topic_context", # MCP tool names usually follow snake_case or similar
    description="Loads information about a given topic from Jina, Firecrawl, and Perplexity, then indexes it into Milvus."
)
async def load_topic_tool(args: LoadTopicToolArgs, ctx: Context) -> SDKLoadTopicResponse:
    """
    MCP Tool to load and index topic context.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context
    topic = args.topic
    
    response_errors: List[str] = []
    processed_urls_count = 0
    perplexity_queried_successfully = False
    jina_results_count = 0
    voyage_input_doc_type: VoyageInputType = settings.VOYAGEAI_INPUT_TYPE_DOCUMENT

    logger.info(f"[Tool: load_topic_context] Processing topic: {topic}")

    jina_results: List[Dict[str, Any]] = []
    try:
        jina_results = await app_ctx.external_api_client.search_jina(topic, num_results=settings.JINA_SEARCH_LIMIT)
        jina_results_count = len(jina_results)
        logger.info(f"Jina found {jina_results_count} results for '{topic}'.")
        
        jina_snippets_to_embed = [res['snippet'] for res in jina_results if res.get('snippet')]
        # Filter metadata list to match snippets that were actually found and will be embedded
        jina_metadata = [{'text': res['snippet'], 'url': res.get('url', f"jina_snippet_for_{topic}")}
                         for res in jina_results if res.get('snippet')]

        if jina_snippets_to_embed: # Ensure we only proceed if there's something to embed
            snippet_embeddings = await app_ctx.embedding_client.embed_texts(jina_snippets_to_embed, input_type=voyage_input_doc_type)
            data_to_insert = [{
                settings.MILVUS_TEXT_FIELD_NAME: meta['text'],
                settings.MILVUS_TOPIC_FIELD_NAME: topic,
                settings.MILVUS_SOURCE_TYPE_FIELD_NAME: "jina_snippet",
                settings.MILVUS_SOURCE_IDENTIFIER_FIELD_NAME: meta['url'],
                settings.MILVUS_INDEX_FIELD_NAME: snippet_embeddings[i]
            } for i, meta in enumerate(jina_metadata)] # metadata and snippet_embeddings should align
            await app_ctx.milvus_operator.insert_data(data_to_insert)
    except MCPError as e:
        logger.error(f"Error in Jina processing for '{topic}': {e.message}", exc_info=True) # Log full exc_info for MCPError
        response_errors.append(f"Jina processing: {e.message}")
    except Exception as e:
        logger.error(f"Unexpected error in Jina processing for '{topic}': {e}", exc_info=True)
        response_errors.append(f"Jina processing: Unexpected error - {str(e)[:100]}")

    # Define async sub-tasks for Perplexity and Firecrawl
    async def perplexity_sub_task():
        nonlocal perplexity_queried_successfully # Allow modification of outer scope var
        try:
            summary = await app_ctx.external_api_client.query_perplexity(topic)
            embedding = await app_ctx.embedding_client.embed_texts(summary, input_type=voyage_input_doc_type)
            await app_ctx.milvus_operator.insert_data([{
                settings.MILVUS_INDEX_FIELD_NAME: embedding, settings.MILVUS_TEXT_FIELD_NAME: summary,
                settings.MILVUS_TOPIC_FIELD_NAME: topic, settings.MILVUS_SOURCE_TYPE_FIELD_NAME: "perplexity",
                settings.MILVUS_SOURCE_IDENTIFIER_FIELD_NAME: f"perplexity_summary_for_{topic}"
            }])
            perplexity_queried_successfully = True
            return "Perplexity task completed."
        except MCPError as e:
            logger.error(f"Error in Perplexity sub-task for '{topic}': {e.message}", exc_info=True)
            response_errors.append(f"Perplexity task: {e.message}")
            return f"Perplexity task failed: {e.message}" # Return error string for gather result
        except Exception as e:
            logger.error(f"Unexpected error in Perplexity sub-task for '{topic}': {e}", exc_info=True)
            err_msg = f"Perplexity task: Unexpected error - {str(e)[:100]}"
            response_errors.append(err_msg)
            return err_msg


    async def firecrawl_sub_task(url: str):
        nonlocal processed_urls_count # Allow modification
        try:
            crawled = await app_ctx.external_api_client.crawl_url_firecrawl(url)
            embedding = await app_ctx.embedding_client.embed_texts(crawled["content"], input_type=voyage_input_doc_type)
            await app_ctx.milvus_operator.insert_data([{
                settings.MILVUS_INDEX_FIELD_NAME: embedding, settings.MILVUS_TEXT_FIELD_NAME: crawled["content"],
                settings.MILVUS_TOPIC_FIELD_NAME: topic, settings.MILVUS_SOURCE_TYPE_FIELD_NAME: "firecrawl",
                settings.MILVUS_SOURCE_IDENTIFIER_FIELD_NAME: url
            }])
            processed_urls_count += 1
            return f"Firecrawl task for {url} completed."
        except MCPError as e:
            logger.error(f"Error in Firecrawl sub-task for URL '{url}': {e.message}", exc_info=True)
            response_errors.append(f"Firecrawl ({url}): {e.message}")
            return f"Firecrawl task for {url} failed: {e.message}"
        except Exception as e:
            logger.error(f"Unexpected error in Firecrawl sub-task for URL '{url}': {e}", exc_info=True)
            err_msg = f"Firecrawl ({url}): Unexpected error - {str(e)[:100]}"
            response_errors.append(err_msg)
            return err_msg

    concurrent_tasks = [perplexity_sub_task()]
    if jina_results:
        concurrent_tasks.extend([firecrawl_sub_task(res['url']) for res in jina_results if res.get('url')])
    
    if concurrent_tasks:
        task_results = await asyncio.gather(*concurrent_tasks, return_exceptions=False) # Errors handled within tasks and appended to response_errors
        logger.info(f"Concurrent task results for topic '{topic}': {task_results}")


    final_message = f"Topic '{topic}' processing initiated."
    if response_errors:
        final_message += " Some operations encountered errors or are pending."
    else:
        final_message += " All operations initiated successfully."
    
    return SDKLoadTopicResponse(
        message=final_message, topic=topic, urls_processed=processed_urls_count,
        perplexity_queried=perplexity_queried_successfully, jina_results_found=jina_results_count,
        errors=response_errors
    )

class QueryTopicToolArgs(BaseModel):
    query: str = Field(..., min_length=1, description="The query to search for relevant context.")
    top_k: Optional[int] = Field(default_factory=lambda: settings.MILVUS_SEARCH_LIMIT, gt=0, description="Number of results to return.")


@mcp_server.tool(
    name="query_topic_context",
    description="Queries Milvus for context relevant to the given query string."
)
async def query_topic_tool(args: QueryTopicToolArgs, ctx: Context) -> SDKQueryTopicResponse:
    """
    MCP Tool to query indexed topic context.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context
    query_text = args.query
    top_k = args.top_k
    voyage_input_query_type: VoyageInputType = settings.VOYAGEAI_INPUT_TYPE_QUERY
    
    logger.info(f"[Tool: query_topic_context] Processing query: {query_text[:100]}...")

    try:
        query_embedding = await app_ctx.embedding_client.embed_texts(query_text, input_type=voyage_input_query_type)
        milvus_results = await app_ctx.milvus_operator.query_data(query_embedding, top_k, None)
        
        results_for_response = [SDKDocumentFragment(**res) for res in milvus_results]
        
        return SDKQueryTopicResponse(
            query=query_text, results=results_for_response, 
            message=f"Found {len(results_for_response)} relevant documents."
        )
    except MCPError as e:
        logger.error(f"Error processing query '{query_text[:100]}...': {e.message}", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Unexpected error processing query '{query_text[:100]}...': {e}", exc_info=True)
        raise MCPError(f"Query processing failed: {str(e)[:100]}")


# To run this server directly using `python src/mcp_app.py`
def main():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting MCP RAG Context Server...")
    # The mcp SDK provides mcp.run() or mcp dev for execution.
    # For direct python execution with FastMCP:
    mcp_server.run() 

if __name__ == "__main__":
    main()
