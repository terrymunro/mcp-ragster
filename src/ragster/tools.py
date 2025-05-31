"""MCP tool implementations for RAG context management."""

import asyncio
import logging
from typing import Any, cast
from pydantic import BaseModel, Field, field_validator, model_validator

from .config import settings
from .models import LoadTopicResponse, QueryTopicResponse, DocumentFragment
from .embedding_client import VoyageInputType, is_voyage_input_type
from .exceptions import MCPError
from .server import AppContext

logger = logging.getLogger(__name__)


class LoadTopicToolArgs(BaseModel):
    # Multi-topic support with backward compatibility
    topics: list[str] = Field(
        default_factory=list,
        min_length=1,
        max_length=settings.MAX_TOPICS_PER_JOB,
        description="List of topics to research (1-10 topics). For backward compatibility, if 'topic' is provided instead, it will be converted to a single-item list.",
    )

    # Backward compatibility: single topic field
    topic: str | None = Field(
        default=None,
        min_length=1,
        description="Single topic to research (deprecated, use 'topics' instead).",
    )

    @field_validator("topics")
    @classmethod
    def validate_topics(cls, v: list[str]) -> list[str]:
        """Validate topics list."""
        if not v:
            return v  # Will be handled in model_validator

        # Strip whitespace and filter empty strings
        cleaned_topics = [topic.strip() for topic in v if topic.strip()]

        if not cleaned_topics:
            raise ValueError("All topics are empty or contain only whitespace")

        if len(cleaned_topics) > settings.MAX_TOPICS_PER_JOB:
            raise ValueError(
                f"Too many topics: {len(cleaned_topics)} > {settings.MAX_TOPICS_PER_JOB}"
            )

        return cleaned_topics

    @model_validator(mode="after")
    def ensure_topics_provided(self) -> "LoadTopicToolArgs":
        """Ensure either topics or topic is provided, with backward compatibility."""
        if not self.topics and not self.topic:
            raise ValueError("Either 'topics' list or 'topic' string must be provided")

        # Backward compatibility: convert single topic to topics list
        if self.topic and not self.topics:
            self.topics = [self.topic.strip()]

        # If both are provided, topics takes precedence
        if self.topics and self.topic:
            # Clear the deprecated field to avoid confusion
            self.topic = None

        return self


class QueryTopicToolArgs(BaseModel):
    query: str = Field(
        ..., min_length=1, description="The query to search for relevant context."
    )
    top_k: int | None = Field(
        default=settings.MILVUS_SEARCH_LIMIT,
        gt=0,
        description="Number of results to return.",
    )
    search_mode: str = Field(
        default="precise",
        description="Search mode: 'precise' for focused results, 'exploration' for broader semantic search.",
    )


class TopicProcessor:
    """Handles topic processing operations."""

    app_ctx: AppContext
    response_errors: list[str]
    processed_urls_count: int
    perplexity_queried_successfully: bool
    jina_results_count: int
    voyage_input_doc_type: VoyageInputType
    firecrawl_semaphore: asyncio.Semaphore

    def __init__(self, app_context: AppContext):
        self.app_ctx = app_context
        self.response_errors: list[str] = []
        self.processed_urls_count = 0
        self.perplexity_queried_successfully = False
        self.jina_results_count = 0
        self.voyage_input_doc_type: VoyageInputType = (
            settings.VOYAGEAI_INPUT_TYPE_DOCUMENT
            if is_voyage_input_type(settings.VOYAGEAI_INPUT_TYPE_DOCUMENT)
            else "document"
        )
        self.firecrawl_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_FIRECRAWL)

    async def process_jina_search(self, topic: str) -> list[dict[str, Any]]:
        """Process Jina search and embed snippets."""
        try:
            jina_results = await self.app_ctx.jina_client.search(topic)
            self.jina_results_count = len(jina_results)
            logger.info(f"Jina found {self.jina_results_count} results for '{topic}'.")

            await self._embed_jina_snippets(topic, jina_results)
            return jina_results

        except MCPError as e:
            logger.error(
                f"Error in Jina processing for '{topic}': {e.message}", exc_info=True
            )
            self.response_errors.append(f"Jina processing: {e.message}")
            return []
        except Exception as e:
            logger.error(
                f"Unexpected error in Jina processing for '{topic}': {e}", exc_info=True
            )
            self.response_errors.append(
                f"Jina processing: Unexpected error - {str(e)[:100]}"
            )
            return []

    async def _embed_jina_snippets(
        self, topic: str, jina_results: list[dict[str, Any]]
    ) -> None:
        """Embed and store Jina snippets."""
        jina_snippets_to_embed = [
            res["snippet"] for res in jina_results if res.get("snippet")
        ]
        jina_metadata = [
            {"text": res["snippet"], "url": res.get("url", f"jina_snippet_for_{topic}")}
            for res in jina_results
            if res.get("snippet")
        ]

        if jina_snippets_to_embed:
            snippet_embeddings = await self.app_ctx.embedding_client.embed_texts(
                jina_snippets_to_embed, input_type=self.voyage_input_doc_type
            )
            data_to_insert = [
                {
                    settings.MILVUS_TEXT_FIELD_NAME: meta["text"],
                    settings.MILVUS_TOPIC_FIELD_NAME: topic,
                    settings.MILVUS_SOURCE_TYPE_FIELD_NAME: "jina_snippet",
                    settings.MILVUS_SOURCE_IDENTIFIER_FIELD_NAME: meta["url"],
                    settings.MILVUS_INDEX_FIELD_NAME: snippet_embeddings[i],
                }
                for i, meta in enumerate(jina_metadata)
            ]

            self.app_ctx.milvus_operator.insert_data(data_to_insert)

    async def perplexity_sub_task(self, topic: str) -> str:
        """Process Perplexity summary."""
        try:
            summary = await self.app_ctx.perplexity_client.query(topic)
            embedding = await self.app_ctx.embedding_client.embed_texts(
                summary, input_type=self.voyage_input_doc_type
            )
            self.app_ctx.milvus_operator.insert_data(
                [
                    {
                        settings.MILVUS_INDEX_FIELD_NAME: embedding[0],
                        settings.MILVUS_TEXT_FIELD_NAME: summary,
                        settings.MILVUS_TOPIC_FIELD_NAME: topic,
                        settings.MILVUS_SOURCE_TYPE_FIELD_NAME: "perplexity",
                        settings.MILVUS_SOURCE_IDENTIFIER_FIELD_NAME: f"perplexity_summary_for_{topic}",
                    }
                ]
            )
            self.perplexity_queried_successfully = True
            return "Perplexity task completed."
        except MCPError as e:
            logger.error(
                f"Error in Perplexity sub-task for '{topic}': {e.message}",
                exc_info=True,
            )
            self.response_errors.append(f"Perplexity task: {e.message}")
            return f"Perplexity task failed: {e.message}"
        except Exception as e:
            logger.error(
                f"Unexpected error in Perplexity sub-task for '{topic}': {e}",
                exc_info=True,
            )
            err_msg = f"Perplexity task: Unexpected error - {str(e)[:100]}"
            self.response_errors.append(err_msg)
            return err_msg

    async def firecrawl_sub_task(self, topic: str, url: str) -> str:
        """Process Firecrawl URL with semaphore-based concurrency control."""
        async with self.firecrawl_semaphore:
            try:
                crawled = await self.app_ctx.firecrawl_client.crawl_url(url)
                embedding = await self.app_ctx.embedding_client.embed_texts(
                    crawled["content"], input_type=self.voyage_input_doc_type
                )
                self.app_ctx.milvus_operator.insert_data(
                    [
                        {
                            settings.MILVUS_INDEX_FIELD_NAME: embedding[0],
                            settings.MILVUS_TEXT_FIELD_NAME: crawled["content"],
                            settings.MILVUS_TOPIC_FIELD_NAME: topic,
                            settings.MILVUS_SOURCE_TYPE_FIELD_NAME: "firecrawl",
                            settings.MILVUS_SOURCE_IDENTIFIER_FIELD_NAME: url,
                        }
                    ]
                )
                self.processed_urls_count += 1
                return f"Firecrawl task for {url} completed."
            except MCPError as e:
                logger.error(
                    f"Error in Firecrawl sub-task for URL '{url}': {e.message}",
                    exc_info=True,
                )
                self.response_errors.append(f"Firecrawl ({url}): {e.message}")
                return f"Firecrawl task for {url} failed: {e.message}"
            except Exception as e:
                logger.error(
                    f"Unexpected error in Firecrawl sub-task for URL '{url}': {e}",
                    exc_info=True,
                )
                err_msg = f"Firecrawl ({url}): Unexpected error - {str(e)[:100]}"
                self.response_errors.append(err_msg)
                return err_msg

    async def process_concurrent_tasks(
        self, topic: str, jina_results: list[dict[str, Any]]
    ) -> None:
        """Process Perplexity and Firecrawl tasks with progressive result handling."""
        # Create task mapping for metadata tracking
        task_metadata = {}

        # Add Perplexity task
        perplexity_task = asyncio.create_task(self.perplexity_sub_task(topic))
        task_metadata[perplexity_task] = ("perplexity", None)

        # Add Firecrawl tasks
        if jina_results:
            for res in jina_results:
                if url := res.get("url"):
                    firecrawl_task = asyncio.create_task(
                        self.firecrawl_sub_task(topic, url)
                    )
                    task_metadata[firecrawl_task] = ("firecrawl", url)

        if not task_metadata:
            return

        # Wait for all tasks to complete
        try:
            results = await asyncio.gather(
                *task_metadata.keys(), return_exceptions=True
            )
            for task, result in zip(task_metadata.keys(), results):
                task_type, url = task_metadata[task]
                if isinstance(result, Exception):
                    logger.error(f"Task failed ({task_type}, {url}): {str(result)}")
                else:
                    logger.info(f"Task completed ({task_type}): {result}")
        except Exception as e:
            logger.error(f"Error in concurrent task processing: {e}")

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
            errors=self.response_errors,
        )


async def research_topic(
    args: LoadTopicToolArgs, app_context: AppContext
) -> LoadTopicResponse:
    """MCP Tool for performing research and indexing that information for querying."""
    topics = args.topics

    # For backward compatibility and current implementation, process the first topic
    # TODO: This will be updated in later tasks to handle multiple topics properly
    topic = topics[0] if topics else ""

    if len(topics) > 1:
        logger.info(
            f"[Tool: research_topic] Multi-topic support: Processing {len(topics)} topics sequentially"
        )
        logger.info(
            f"[Tool: research_topic] Note: Currently processing only first topic '{topic}'. Full multi-topic processing will be implemented in background tasks."
        )
    else:
        logger.info(f"[Tool: research_topic] Processing single topic: {topic}")

    processor = TopicProcessor(app_context)
    jina_results = await processor.process_jina_search(topic)

    # Process concurrent tasks (Perplexity + Firecrawl)
    await processor.process_concurrent_tasks(topic, jina_results)

    return processor.build_response(topic)


async def query_topic(
    args: QueryTopicToolArgs, app_context: AppContext
) -> QueryTopicResponse:
    """MCP Tool for querying indexed topic context."""
    query_text = args.query
    top_k = args.top_k
    search_mode = args.search_mode
    voyage_input_query_type: VoyageInputType = (
        settings.VOYAGEAI_INPUT_TYPE_QUERY
        if is_voyage_input_type(settings.VOYAGEAI_INPUT_TYPE_QUERY)
        else "query"
    )

    # Determine search ef based on mode
    search_ef = (
        settings.MILVUS_SEARCH_EF_EXPLORATION
        if search_mode == "exploration"
        else settings.MILVUS_SEARCH_EF
    )

    logger.info(
        f"[Tool: query_topic] Processing query: {query_text[:100]}... (mode: {search_mode}, ef: {search_ef})"
    )

    try:
        query_vector = await app_context.embedding_client.embed_texts(
            query_text, input_type=voyage_input_query_type
        )
        # Ensure we have a single vector for querying
        if (
            isinstance(query_embedding, list)
            and len(query_embedding) > 0
            and isinstance(query_embedding[0], list)
        ):
            query_vector = cast(list[float], query_embedding[0])
        elif isinstance(query_embedding, list) and all(
            isinstance(x, float) for x in query_embedding
        ):
            query_vector = cast(list[float], query_embedding)
        else:
            raise MCPError(f"Unexpected embedding type: {type(query_embedding)}")

        milvus_results = await app_context.milvus_operator.query_data(
            query_vector, top_k or settings.MILVUS_SEARCH_LIMIT, None, search_ef
        )

        results_for_response = [DocumentFragment(**res) for res in milvus_results]

        return QueryTopicResponse(
            query=query_text,
            results=results_for_response,
            message=f"Found {len(results_for_response)} relevant context fragments",
        )
    except MCPError as e:
        logger.error(
            f"Error processing query '{query_text[:100]}...': {e.message}",
            exc_info=True,
        )
        raise e
    except Exception as e:
        full_error = str(e)
        logger.error(
            f"Unexpected error processing query '{query_text[:100]}...': {full_error}",
            exc_info=True,
        )
        # Return the full error message for debugging
        raise MCPError(f"Query processing failed: {full_error}")
