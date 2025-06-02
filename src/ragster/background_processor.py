"""Background task infrastructure for asynchronous research operations."""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

from .config import settings
from .job_manager import JobManager
from .job_models import JobStatus
from .models import LoadTopicResponse

logger = logging.getLogger(__name__)


class BackgroundTaskProcessor:
    """Handles background processing of research jobs."""

    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()

    async def start_research_job(
        self,
        job_id: str,
        topics: list[str],
        app_context: Any,
        progress_callback: Optional[
            Callable[[str, str, str, str, dict[str, Any]], Awaitable[None]]
        ] = None,
    ) -> None:
        """Start a research job in the background."""
        if job_id in self._active_tasks:
            logger.warning(f"Job {job_id} is already running")
            return

        # Create and start the background task
        task = asyncio.create_task(
            self._process_research_job(job_id, topics, app_context, progress_callback)
        )
        self._active_tasks[job_id] = task

        # Add cleanup callback
        task.add_done_callback(lambda t: self._cleanup_task(job_id))

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job. Idempotent: returns True if already cancelled."""
        if job_id not in self._active_tasks:
            return False

        # Check if job is already cancelled
        job = await self.job_manager.get_job(job_id)
        if job and job.status == JobStatus.CANCELLED:
            return True

        task = self._active_tasks[job_id]
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Job {job_id} cancelled successfully")
            except Exception as e:
                logger.error(f"Error during job {job_id} cancellation: {e}")

            # Update job status
            job = await self.job_manager.get_job(job_id)
            if job and job.status != JobStatus.CANCELLED:
                await self.job_manager.update_job_status(job_id, JobStatus.CANCELLED)

        return True

    async def get_active_job_count(self) -> int:
        """Get the number of currently active jobs."""
        # Clean up completed tasks
        await self._cleanup_completed_tasks()
        return len(self._active_tasks)

    async def shutdown(self) -> None:
        """Gracefully shutdown the processor, cancelling all active tasks."""
        logger.info("Shutting down background task processor...")
        self._shutdown_event.set()

        # Cancel all active tasks
        if self._active_tasks:
            logger.info(f"Cancelling {len(self._active_tasks)} active tasks")
            for job_id in list(self._active_tasks.keys()):
                await self.cancel_job(job_id)

        # Wait for all tasks to complete
        remaining_tasks = [
            task for task in self._active_tasks.values() if not task.done()
        ]
        if remaining_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*remaining_tasks, return_exceptions=True),
                    timeout=settings.JOB_SHUTDOWN_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.warning("Some tasks did not complete within shutdown timeout")

        logger.info("Background task processor shutdown complete")

    def _cleanup_task(self, job_id: str) -> None:
        """Clean up a completed task."""
        if job_id in self._active_tasks:
            del self._active_tasks[job_id]

    async def _cleanup_completed_tasks(self) -> None:
        """Clean up completed tasks from the active tasks dict."""
        completed_jobs = [
            job_id for job_id, task in self._active_tasks.items() if task.done()
        ]
        for job_id in completed_jobs:
            del self._active_tasks[job_id]

    async def _process_research_job(
        self,
        job_id: str,
        topics: list[str],
        app_context: Any,
        progress_callback: Optional[
            Callable[[str, str, str, str, dict[str, Any]], Awaitable[None]]
        ] = None,
    ) -> None:
        """Process a research job in the background."""
        try:
            # Update job to running status
            await self.job_manager.update_job_status(job_id, JobStatus.RUNNING)

            logger.info(f"Starting research job {job_id} with topics: {topics}")

            # Process topics based on strategy
            if len(topics) <= 2:
                # Sequential processing for small jobs
                await self._process_topics_sequential(
                    job_id, topics, app_context, progress_callback
                )
            else:
                # Parallel processing for larger jobs
                await self._process_topics_parallel(
                    job_id, topics, app_context, progress_callback
                )

            # After processing, check if any topic failed
            job = await self.job_manager.get_job(job_id)
            any_failed = any(
                progress.has_errors() for progress in job.progress.values()
            )
            if any_failed:
                # Gather all errors from topic progress
                all_errors = []
                for progress in job.progress.values():
                    all_errors.extend(progress.errors)
                error_summary = (
                    "; ".join(all_errors)
                    if all_errors
                    else "One or more topics failed."
                )
                await self.job_manager.update_job_status(
                    job_id, JobStatus.FAILED, error=error_summary
                )
                logger.info(f"Research job {job_id} failed due to topic errors")
            else:
                # Mark job as completed
                await self.job_manager.update_job_status(job_id, JobStatus.COMPLETED)
                logger.info(f"Research job {job_id} completed successfully")

        except asyncio.CancelledError:
            logger.info(f"Research job {job_id} was cancelled")
            await self.job_manager.update_job_status(job_id, JobStatus.CANCELLED)
            raise
        except Exception as e:
            logger.error(f"Research job {job_id} failed: {e}", exc_info=True)
            await self.job_manager.update_job_status(
                job_id, JobStatus.FAILED, error=str(e)
            )

    async def _process_topics_sequential(
        self,
        job_id: str,
        topics: list[str],
        app_context: Any,
        progress_callback: Optional[
            Callable[[str, str, str, str, dict[str, Any]], Awaitable[None]]
        ] = None,
    ) -> None:
        """Process topics sequentially (for API rate limit compliance)."""
        logger.info(f"Processing {len(topics)} topics sequentially for job {job_id}")

        topic_results = {}

        for i, topic in enumerate(topics):
            if self._shutdown_event.is_set():
                raise asyncio.CancelledError("Shutdown requested")

            logger.info(f"Processing topic {i + 1}/{len(topics)}: {topic}")

            try:
                # Create a progress callback wrapper for this specific topic
                topic_progress_callback: Optional[
                    Callable[[str, str, dict[str, Any]], Awaitable[None]]
                ] = None
                if progress_callback:

                    async def topic_callback_wrapper(
                        stage: str, status: str, extra_data: dict[str, Any]
                    ) -> None:
                        assert progress_callback is not None
                        return await progress_callback(
                            job_id, topic, stage, status, extra_data
                        )

                    topic_progress_callback = topic_callback_wrapper

                # Process the topic
                result = await self._process_single_topic(
                    topic, app_context, topic_progress_callback
                )
                topic_results[topic] = result

                logger.info(f"Topic '{topic}' processed successfully")

            except Exception as e:
                logger.error(f"Error processing topic '{topic}': {e}", exc_info=True)
                # Update topic progress with error (always record in job progress)
                await self.job_manager.update_topic_progress(
                    job_id, topic, "all", "failed", error=str(e)
                )
                # Also call progress_callback if provided
                if progress_callback:
                    await progress_callback(
                        job_id, topic, "all", "failed", {"error": str(e)}
                    )
                # Continue with other topics
                continue

        # Store combined results
        await self._store_job_results(job_id, topic_results)

    async def _process_topics_parallel(
        self,
        job_id: str,
        topics: list[str],
        app_context: Any,
        progress_callback: Optional[
            Callable[[str, str, str, str, dict[str, Any]], Awaitable[None]]
        ] = None,
    ) -> None:
        """Process topics in parallel with intelligent throttling."""
        logger.info(f"Processing {len(topics)} topics in parallel for job {job_id}")

        # Create semaphore for parallel processing
        max_concurrent = min(settings.MAX_CONCURRENT_TOPICS_PER_JOB, len(topics))
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_topic_with_semaphore(
            topic: str,
        ) -> tuple[str, Optional[LoadTopicResponse]]:
            async with semaphore:
                try:
                    # Create a progress callback wrapper for this specific topic
                    topic_progress_callback: Optional[
                        Callable[[str, str, dict[str, Any]], Awaitable[None]]
                    ] = None
                    if progress_callback:

                        async def topic_callback_wrapper(
                            stage: str, status: str, extra_data: dict[str, Any]
                        ) -> None:
                            assert progress_callback is not None
                            return await progress_callback(
                                job_id, topic, stage, status, extra_data
                            )

                        topic_progress_callback = topic_callback_wrapper

                    result = await self._process_single_topic(
                        topic, app_context, topic_progress_callback
                    )
                    return topic, result
                except Exception as e:
                    logger.error(
                        f"Error processing topic '{topic}': {e}", exc_info=True
                    )
                    if progress_callback:
                        await progress_callback(
                            job_id, topic, "all", "failed", {"error": str(e)}
                        )
                    return topic, None

        # Create tasks for all topics
        tasks = [
            asyncio.create_task(process_topic_with_semaphore(topic)) for topic in topics
        ]

        # Wait for all tasks to complete
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}", exc_info=True)
            raise

        # Process results
        topic_results = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                continue

            if isinstance(result, tuple) and len(result) == 2:
                topic, topic_result = result
                if topic_result:
                    topic_results[topic] = topic_result
            else:
                logger.error(f"Unexpected result format: {result}")

        # Store combined results
        await self._store_job_results(job_id, topic_results)

    async def _process_single_topic(
        self,
        topic: str,
        app_context: Any,
        progress_callback: Optional[
            Callable[[str, str, dict[str, Any]], Awaitable[None]]
        ] = None,
    ) -> LoadTopicResponse:
        """Process a single topic using the existing TopicProcessor."""
        from .tools import TopicProcessor

        processor = TopicProcessor(app_context)

        try:
            # Jina search phase
            if progress_callback:
                await progress_callback("jina", "running", {})

            jina_results = await processor.process_jina_search(topic)

            if progress_callback:
                await progress_callback(
                    "jina", "completed", {"urls_found": len(jina_results)}
                )

            # Perplexity and Firecrawl phases
            if progress_callback:
                await progress_callback("perplexity", "running", {})
                await progress_callback("firecrawl", "running", {})

            await processor.process_concurrent_tasks(topic, jina_results)

            if progress_callback:
                await progress_callback("perplexity", "completed", {})
                await progress_callback(
                    "firecrawl",
                    "completed",
                    {"urls_processed": processor.processed_urls_count},
                )

            return processor.build_response(topic)

        except Exception as e:
            logger.error(f"Error processing topic '{topic}': {e}", exc_info=True)
            if progress_callback:
                await progress_callback("all", "failed", {"error": str(e)})
            raise

    async def _store_job_results(
        self, job_id: str, topic_results: dict[str, LoadTopicResponse]
    ) -> None:
        """Store the combined results from all topics."""
        job = await self.job_manager.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found when storing results")
            return

        # Create combined response
        if topic_results:
            # For now, store the first successful result as the primary result
            # In the future, this could be enhanced to create a MultiTopicResponse
            first_result = next(iter(topic_results.values()))
            job.results = first_result
            # Note: Job results are stored directly in the job object
        logger.info(f"Stored results for job {job_id}")
