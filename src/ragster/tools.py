"""MCP tool implementations for RAG context management."""

import asyncio
import logging
from pydantic import BaseModel, Field

from .config import settings
from .models import (
    LoadTopicResponse,
    QueryTopicResponse, 
    DocumentFragment
)
from .embedding_client import VoyageInputType
from .exceptions import MCPError

logger = logging.getLogger(__name__)


class LoadTopicToolArgs(BaseModel):
    topic: str = Field(..., min_length=1, description="The topic to load information about.")


class QueryTopicToolArgs(BaseModel):
    query: str = Field(..., min_length=1, description="The query to search for relevant context.")
    top_k: int | None = Field(
        default_factory=lambda: settings.MILVUS_SEARCH_LIMIT, 
        gt=0, 
        description="Number of results to return."
    )


class TopicProcessor:
    """Handles topic processing operations."""
    
    def __init__(self, app_context):
        self.app_ctx = app_context
        self.response_errors: list[str] = []
        self.processed_urls_count = 0
        self.perplexity_queried_successfully = False
        self.jina_results_count = 0
        self.voyage_input_doc_type: VoyageInputType = settings.VOYAGEAI_INPUT_TYPE_DOCUMENT

    async def process_jina_search(self, topic: str) -> list[dict[str, any]]:
        """Process Jina search and embed snippets."""
        try:
            jina_results = await self.app_ctx.external_api_client.search_jina(
                topic, num_results=settings.JINA_SEARCH_LIMIT
            )
            self.jina_results_count = len(jina_results)
            logger.info(f"Jina found {self.jina_results_count} results for '{topic}'.")
            
            await self._embed_jina_snippets(topic, jina_results)
            return jina_results
            
        except MCPError as e:
            logger.error(f"Error in Jina processing for '{topic}': {e.message}", exc_info=True)
            self.response_errors.append(f"Jina processing: {e.message}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in Jina processing for '{topic}': {e}", exc_info=True)
            self.response_errors.append(f"Jina processing: Unexpected error - {str(e)[:100]}")
            return []

    async def _embed_jina_snippets(self, topic: str, jina_results: list[dict[str, any]]) -> None:
        """Embed and store Jina snippets."""
        jina_snippets_to_embed = [res['snippet'] for res in jina_results if res.get('snippet')]
        jina_metadata = [
            {'text': res['snippet'], 'url': res.get('url', f"jina_snippet_for_{topic}")}
            for res in jina_results if res.get('snippet')
        ]

        if jina_snippets_to_embed:
            snippet_embeddings = await self.app_ctx.embedding_client.embed_texts(
                jina_snippets_to_embed, input_type=self.voyage_input_doc_type
            )
            data_to_insert = [{
                settings.MILVUS_TEXT_FIELD_NAME: meta['text'],
                settings.MILVUS_TOPIC_FIELD_NAME: topic,
                settings.MILVUS_SOURCE_TYPE_FIELD_NAME: "jina_snippet",
                settings.MILVUS_SOURCE_IDENTIFIER_FIELD_NAME: meta['url'],
                settings.MILVUS_INDEX_FIELD_NAME: snippet_embeddings[i]
            } for i, meta in enumerate(jina_metadata)]
            
            await self.app_ctx.milvus_operator.insert_data(data_to_insert)

    async def perplexity_sub_task(self, topic: str) -> str:
        """Process Perplexity summary."""
        try:
            summary = await self.app_ctx.external_api_client.query_perplexity(topic)
            embedding = await self.app_ctx.embedding_client.embed_texts(
                summary, input_type=self.voyage_input_doc_type
            )
            await self.app_ctx.milvus_operator.insert_data([{
                settings.MILVUS_INDEX_FIELD_NAME: embedding, 
                settings.MILVUS_TEXT_FIELD_NAME: summary,
                settings.MILVUS_TOPIC_FIELD_NAME: topic, 
                settings.MILVUS_SOURCE_TYPE_FIELD_NAME: "perplexity",
                settings.MILVUS_SOURCE_IDENTIFIER_FIELD_NAME: f"perplexity_summary_for_{topic}"
            }])
            self.perplexity_queried_successfully = True
            return "Perplexity task completed."
        except MCPError as e:
            logger.error(f"Error in Perplexity sub-task for '{topic}': {e.message}", exc_info=True)
            self.response_errors.append(f"Perplexity task: {e.message}")
            return f"Perplexity task failed: {e.message}"
        except Exception as e:
            logger.error(f"Unexpected error in Perplexity sub-task for '{topic}': {e}", exc_info=True)
            err_msg = f"Perplexity task: Unexpected error - {str(e)[:100]}"
            self.response_errors.append(err_msg)
            return err_msg

    async def firecrawl_sub_task(self, topic: str, url: str) -> str:
        """Process Firecrawl URL."""
        try:
            crawled = await self.app_ctx.external_api_client.crawl_url_firecrawl(url)
            embedding = await self.app_ctx.embedding_client.embed_texts(
                crawled["content"], input_type=self.voyage_input_doc_type
            )
            await self.app_ctx.milvus_operator.insert_data([{
                settings.MILVUS_INDEX_FIELD_NAME: embedding, 
                settings.MILVUS_TEXT_FIELD_NAME: crawled["content"],
                settings.MILVUS_TOPIC_FIELD_NAME: topic, 
                settings.MILVUS_SOURCE_TYPE_FIELD_NAME: "firecrawl",
                settings.MILVUS_SOURCE_IDENTIFIER_FIELD_NAME: url
            }])
            self.processed_urls_count += 1
            return f"Firecrawl task for {url} completed."
        except MCPError as e:
            logger.error(f"Error in Firecrawl sub-task for URL '{url}': {e.message}", exc_info=True)
            self.response_errors.append(f"Firecrawl ({url}): {e.message}")
            return f"Firecrawl task for {url} failed: {e.message}"
        except Exception as e:
            logger.error(f"Unexpected error in Firecrawl sub-task for URL '{url}': {e}", exc_info=True)
            err_msg = f"Firecrawl ({url}): Unexpected error - {str(e)[:100]}"
            self.response_errors.append(err_msg)
            return err_msg

    async def process_concurrent_tasks(self, topic: str, jina_results: list[dict[str, any]]) -> None:
        """Process Perplexity and Firecrawl tasks concurrently."""
        concurrent_tasks = [self.perplexity_sub_task(topic)]
        
        if jina_results:
            concurrent_tasks.extend([
                self.firecrawl_sub_task(topic, res['url']) 
                for res in jina_results if res.get('url')
            ])
        
        if concurrent_tasks:
            task_results = await asyncio.gather(*concurrent_tasks, return_exceptions=False)
            logger.info(f"Concurrent task results for topic '{topic}': {task_results}")

    def build_response(self, topic: str) -> LoadTopicResponse:
        """Build the final response."""
        final_message = f"Topic '{topic}' processing initiated."
        if self.response_errors:
            final_message += " Some operations encountered errors or are pending."
        else:
            final_message += " All operations initiated successfully."
        
        return LoadTopicResponse(
            message=final_message, 
            topic=topic, 
            urls_processed=self.processed_urls_count,
            perplexity_queried=self.perplexity_queried_successfully, 
            jina_results_found=self.jina_results_count,
            errors=self.response_errors
        )


async def load_topic_context(args: LoadTopicToolArgs, app_context) -> LoadTopicResponse:
    """MCP Tool to load and index topic context."""
    topic = args.topic
    logger.info(f"[Tool: load_topic_context] Processing topic: {topic}")

    processor = TopicProcessor(app_context)
    
    # Process Jina search and embed snippets
    jina_results = await processor.process_jina_search(topic)
    
    # Process concurrent tasks (Perplexity + Firecrawl)
    await processor.process_concurrent_tasks(topic, jina_results)
    
    return processor.build_response(topic)


async def query_topic_context(args: QueryTopicToolArgs, app_context) -> QueryTopicResponse:
    """MCP Tool to query indexed topic context."""
    query_text = args.query
    top_k = args.top_k
    voyage_input_query_type: VoyageInputType = settings.VOYAGEAI_INPUT_TYPE_QUERY
    
    logger.info(f"[Tool: query_topic_context] Processing query: {query_text[:100]}...")

    try:
        query_embedding = await app_context.embedding_client.embed_texts(
            query_text, input_type=voyage_input_query_type
        )
        milvus_results = await app_context.milvus_operator.query_data(
            query_embedding, top_k, None
        )
        
        results_for_response = [DocumentFragment(**res) for res in milvus_results]
        
        return QueryTopicResponse(
            query=query_text, 
            results=results_for_response, 
            message=f"Found {len(results_for_response)} relevant documents."
        )
    except MCPError as e:
        logger.error(f"Error processing query '{query_text[:100]}...': {e.message}", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Unexpected error processing query '{query_text[:100]}...': {e}", exc_info=True)
        raise MCPError(f"Query processing failed: {str(e)[:100]}")