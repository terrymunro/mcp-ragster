"""MCP tool implementations for RAG context management."""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional
from datetime import datetime, timedelta
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
    progress_callback: Optional[Callable[[str, str], Awaitable[None]]]

    def __init__(
        self,
        app_context: AppContext,
        progress_callback: Optional[Callable[[str, str], Awaitable[None]]] = None,
    ):
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
        self.progress_callback = progress_callback

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


async def research_topic(args: LoadTopicToolArgs, app_context: AppContext):
    """Research and index topic information asynchronously with intelligent caching and resource management.

    For backward compatibility, single topics return LoadTopicResponse.
    For multi-topic jobs, returns ResearchJobResponse with job tracking.
    """
    from .job_models import ResearchJobResponse

    topics = args.topics
    logger.info(
        f"[Tool: research_topic] Starting research job for {len(topics)} topics: {topics}"
    )

    # Check cache for any existing results
    cache_hits, cache_misses = await app_context.result_cache.check_cache_hits(topics)

    if cache_hits:
        logger.info(f"Cache hits for topics: {cache_hits}")

    # For single topic with cache hit, return immediately
    if len(topics) == 1 and topics[0] in cache_hits:
        cached_result = await app_context.result_cache.get(topics[0])
        if cached_result:
            logger.info(f"Returning cached result for topic: {topics[0]}")
            return cached_result

    # Check global concurrency limits
    active_jobs = await app_context.background_processor.get_active_job_count()
    if active_jobs >= settings.MAX_CONCURRENT_RESEARCH_JOBS:
        return ResearchJobResponse(
            job_id="queued",
            status="pending",
            topics=topics,
            message=f"Job queued due to concurrency limit. {active_jobs} jobs currently active.",
            created_at=datetime.utcnow(),
            estimated_completion_time=None,
        )

    # For single topic (backward compatibility), process synchronously if no cache
    if len(topics) == 1 and not cache_hits:
        logger.info(f"Processing single topic synchronously: {topics[0]}")

        # Allocate resources for single topic
        await app_context.resource_manager.allocate_resources_for_topics(1)

        try:
            processor = TopicProcessor(app_context)
            topic = topics[0]

            # Process the topic stages
            jina_results = await processor.process_jina_search(topic)
            await processor.process_concurrent_tasks(topic, jina_results)

            result = processor.build_response(topic)

            # Cache the result
            await app_context.result_cache.put(topic, result)

            return result

        finally:
            await app_context.resource_manager.release_resources_for_topics(1)

    # Multi-topic processing - use asynchronous job system
    logger.info(
        f"Creating async research job for {len(topics)} topics: {topics} (cache_misses: {cache_misses})"
    )

    # Create the research job
    job = await app_context.job_manager.create_job(topics)

    # Get processing strategy suggestion
    strategy = await app_context.resource_manager.suggest_processing_strategy(
        len(cache_misses)
    )
    logger.info(
        f"Using processing strategy: {strategy} for {len(cache_misses)} uncached topics"
    )

    # Setup progress callback
    async def progress_callback(
        job_id_param: str,
        topic: str,
        stage: str,
        status: str,
        extra_data: dict[str, Any],
    ) -> None:
        """Progress callback for job updates."""
        try:
            await app_context.job_manager.update_topic_progress(
                job_id_param, topic, stage, status, **extra_data
            )
            logger.debug(
                f"Updated progress: job={job_id_param}, topic={topic}, "
                f"stage={stage}, status={status}"
            )
        except Exception as e:
            logger.error(f"Error updating progress: {e}")

    # Start background processing
    await app_context.background_processor.start_research_job(
        job.job_id, topics, app_context, progress_callback
    )

    # Return job info immediately
    estimated_time = None
    if len(cache_misses) > 0:
        # Rough estimation: 30 seconds per topic for processing
        estimated_seconds = len(cache_misses) * 30
        if strategy == "sequential":
            estimated_seconds *= 1.5  # Sequential takes longer
        estimated_time = datetime.utcnow() + timedelta(seconds=estimated_seconds)

    return ResearchJobResponse(
        job_id=job.job_id,
        status=job.status.value,
        topics=job.topics,
        message=f"Research job created for {len(topics)} topics ({len(cache_hits)} cached, {len(cache_misses)} to process). Use get_research_status to monitor progress.",
        created_at=job.created_at,
        estimated_completion_time=estimated_time,
    )


async def query_topic(
    args: QueryTopicToolArgs, app_context: AppContext
) -> QueryTopicResponse:
    """Query indexed topic context using Milvus vector search."""
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
        # Embed the query
        from .embedding_client import VoyageInputType

        milvus_results = await app_context.milvus_operator.query_data(
            query_vector, top_k or settings.MILVUS_SEARCH_LIMIT, None, search_ef
        voyage_query_type: VoyageInputType = "query"
        query_embedding = await app_context.embedding_client.embed_texts(
            args.query, input_type=voyage_query_type
        )

        # Determine search parameters based on search mode
        search_ef = None
        if settings.MILVUS_INDEX_TYPE == "HNSW":
            search_ef = (
                settings.MILVUS_SEARCH_EF_EXPLORATION
                if args.search_mode == "exploration"
                else settings.MILVUS_SEARCH_EF
            )

        # Perform vector search
        search_results = app_context.milvus_operator.query_data(
            query_embedding[0],
            args.top_k or settings.MILVUS_SEARCH_LIMIT,
            search_ef=search_ef,
        )

        # Convert results to response format
        results = [
            DocumentFragment(
                id=result.get("id"),
                text_content=result.get(settings.MILVUS_TEXT_FIELD_NAME, ""),
                source_type=result.get(settings.MILVUS_SOURCE_TYPE_FIELD_NAME, ""),
                source_identifier=result.get(
                    settings.MILVUS_SOURCE_IDENTIFIER_FIELD_NAME, ""
                ),
                topic=result.get(settings.MILVUS_TOPIC_FIELD_NAME, ""),
                distance=result.get("distance"),
            )
            for result in search_results
        ]

        return QueryTopicResponse(
            query=args.query,
            results=results,
            message=f"Found {len(results)} relevant results for query '{args.query}'.",
        )

    except Exception as e:
        logger.error(f"Error in query_topic: {e}", exc_info=True)
        return QueryTopicResponse(
            query=args.query,
            results=[],
            message=f"Error querying topic: {str(e)[:100]}",
        )


# New MCP tools for job management


async def get_research_status(args, app_context: AppContext):
    """Get the status of a research job."""
    from .job_models import JobStatusResponse

    job = await app_context.job_manager.get_job(args.job_id)
    if not job:
        raise MCPError(f"Job {args.job_id} not found")

    # Convert TopicProgress to dict format for response
    topic_progress_dict = {}
    for topic, progress in job.progress.items():
        topic_progress_dict[topic] = {
            "jina_status": progress.jina_status,
            "perplexity_status": progress.perplexity_status,
            "firecrawl_status": progress.firecrawl_status,
            "urls_found": progress.urls_found,
            "urls_processed": progress.urls_processed,
            "completion_percentage": progress.get_completion_percentage(),
            "errors": progress.errors,
        }

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        topics=job.topics,
        overall_progress=job.get_overall_progress(),
        topic_progress=topic_progress_dict,
        estimated_completion_time=job.get_estimated_completion_time(),
        results=job.results,
        error=job.error,
    )


async def list_research_jobs(args, app_context: AppContext):
    """List research jobs with optional filtering."""

    # The job manager already returns a properly formatted ListJobsResponse
    return await app_context.job_manager.list_jobs(
        status_filter=args.status_filter, limit=args.limit, offset=args.offset
    )


async def cancel_research_job(args, app_context: AppContext):
    """Cancel a research job."""
    from .job_models import CancelJobResponse

    job = await app_context.job_manager.get_job(args.job_id)
    if not job:
        raise MCPError(f"Job {args.job_id} not found")

    # Cancel the background task if it's running
    if hasattr(app_context, "background_processor"):
        cancelled = await app_context.background_processor.cancel_job(args.job_id)
        if not cancelled:
            return CancelJobResponse(
                job_id=args.job_id,
                status=job.status.value,
                message=f"Job {args.job_id} was not running or could not be cancelled",
                preserved_results=None,
            )

    # Get the final job state
    updated_job = await app_context.job_manager.get_job(args.job_id)

    # Preserve any completed topic results
    preserved_results = {}
    if updated_job and updated_job.progress:
        for topic, progress in updated_job.progress.items():
            if progress.is_completed():
                # Create a basic LoadTopicResponse for completed topics
                # In practice, you might want to retrieve the actual results
                from .models import LoadTopicResponse

                preserved_results[topic] = LoadTopicResponse(
                    message=f"Topic '{topic}' completed before cancellation",
                    topic=topic,
                    urls_processed=progress.urls_processed,
                    perplexity_queried=progress.perplexity_status == "completed",
                    jina_results_found=progress.urls_found,
                    errors=progress.errors,
                )

    return CancelJobResponse(
        job_id=args.job_id,
        status=updated_job.status.value if updated_job else "unknown",
        message=f"Job {args.job_id} cancelled successfully",
        preserved_results=preserved_results if preserved_results else None,
    )
