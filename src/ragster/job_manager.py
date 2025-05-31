"""Job management system for asynchronous research operations."""

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional

from .job_models import (
    JobStatus,
    ResearchJob,
    ListJobsResponse,
    JobStatusResponse,
    CancelJobResponse,
)
from .config import settings
from .exceptions import MCPError

logger = logging.getLogger(__name__)


class JobManagerError(MCPError):
    """Base exception for JobManager operations."""

    pass


class JobNotFoundError(JobManagerError):
    """Raised when a job is not found."""

    pass


class JobStateTransitionError(JobManagerError):
    """Raised when an invalid job state transition is attempted."""

    pass


class JobManager:
    """Thread-safe manager for research jobs with lifecycle management."""

    def __init__(self):
        """Initialize the job manager with configuration."""
        self._jobs: Dict[str, ResearchJob] = {}
        self._lock = asyncio.Lock()
        self._running_tasks: Dict[str, asyncio.Task] = {}

        # Job retention settings
        self._retention_hours = settings.JOB_RETENTION_HOURS
        self._max_jobs = settings.MAX_STORED_JOBS

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval_seconds = 3600  # 1 hour

        logger.info(
            "JobManager initialized with retention_hours=%d, max_jobs=%d",
            self._retention_hours,
            self._max_jobs,
        )

    async def start(self) -> None:
        """Start the job manager and cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("JobManager cleanup task started")

    async def stop(self) -> None:
        """Stop the job manager and cleanup resources."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("JobManager cleanup task stopped")

        # Cancel any running tasks
        async with self._lock:
            for job_id, task in self._running_tasks.items():
                if not task.done():
                    task.cancel()
                    logger.info("Cancelled running task for job %s", job_id)
            self._running_tasks.clear()

    async def create_job(self, topics: List[str]) -> ResearchJob:
        """Create a new research job with the given topics."""
        if not topics:
            raise JobManagerError("Cannot create job with empty topics list")

        if len(topics) > settings.MAX_TOPICS_PER_JOB:
            raise JobManagerError(
                f"Too many topics: {len(topics)} > {settings.MAX_TOPICS_PER_JOB}"
            )

        job_id = str(uuid.uuid4())
        job = ResearchJob(
            job_id=job_id,
            topics=topics,
            status=JobStatus.PENDING,
            created_at=datetime.now(UTC),
        )

        async with self._lock:
            # Check storage limits
            await self._enforce_storage_limits()
            self._jobs[job_id] = job

        logger.info(
            "Created research job %s with %d topics: %s", job_id, len(topics), topics
        )
        return job

    async def get_job(self, job_id: str) -> ResearchJob:
        """Get a job by ID."""
        async with self._lock:
            job = self._jobs.get(job_id)

        if job is None:
            raise JobNotFoundError(f"Job not found: {job_id}")

        return job

    async def update_job_status(
        self, job_id: str, status: JobStatus, error: Optional[str] = None
    ) -> ResearchJob:
        """Update job status with validation."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(f"Job not found: {job_id}")

            # Validate state transition
            if not self._is_valid_transition(job.status, status):
                raise JobStateTransitionError(
                    f"Invalid transition from {job.status} to {status} for job {job_id}"
                )

            old_status = job.status
            job.status = status

            # Update timestamps
            if status == JobStatus.RUNNING and job.started_at is None:
                job.started_at = datetime.now(UTC)
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                job.completed_at = datetime.now(UTC)

            # Set error if provided
            if error:
                job.error = error

            logger.info("Job %s status changed: %s -> %s", job_id, old_status, status)
            return job

    async def update_topic_progress(
        self, job_id: str, topic: str, stage: str, status: str, **kwargs: Any
    ) -> ResearchJob:
        """Update progress for a specific topic and stage."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(f"Job not found: {job_id}")

            job.update_topic_progress(topic, stage, status, **kwargs)

            # Auto-complete job if all topics are done
            if job.is_complete() and job.status == JobStatus.RUNNING:
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now(UTC)
                logger.info("Job %s auto-completed - all topics finished", job_id)

            return job

    async def list_jobs(
        self, status_filter: Optional[str] = None, limit: int = 10, offset: int = 0
    ) -> ListJobsResponse:
        """List jobs with optional filtering and pagination."""
        async with self._lock:
            jobs = list(self._jobs.values())

        # Filter by status if provided
        if status_filter:
            jobs = [job for job in jobs if job.status == status_filter]

        # Sort by creation time (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        # Apply pagination
        total_count = len(jobs)
        paginated_jobs = jobs[offset : offset + limit]

        # Convert to response format
        job_responses = []
        for job in paginated_jobs:
            topic_progress = {}
            for topic, progress in job.progress.items():
                topic_progress[topic] = {
                    "jina_status": progress.jina_status,
                    "perplexity_status": progress.perplexity_status,
                    "firecrawl_status": progress.firecrawl_status,
                    "urls_found": progress.urls_found,
                    "urls_processed": progress.urls_processed,
                    "completion_percentage": progress.get_completion_percentage(),
                    "errors": progress.errors,
                }

            job_responses.append(
                JobStatusResponse(
                    job_id=job.job_id,
                    status=job.status,
                    topics=job.topics,
                    overall_progress=job.get_overall_progress(),
                    topic_progress=topic_progress,
                    estimated_completion_time=job.get_estimated_completion_time(),
                    results=job.results,
                    error=job.error,
                )
            )

        return ListJobsResponse(
            jobs=job_responses,
            total_count=total_count,
            has_more=(offset + limit) < total_count,
        )

    async def cancel_job(
        self, job_id: str, preserve_partial: bool = True
    ) -> CancelJobResponse:
        """Cancel a job and optionally preserve partial results."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(f"Job not found: {job_id}")

            if job.status in (
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ):
                raise JobStateTransitionError(
                    f"Cannot cancel job in status {job.status}"
                )

            # Cancel running task if exists
            task = self._running_tasks.get(job_id)
            if task and not task.done():
                task.cancel()
                logger.info("Cancelled running task for job %s", job_id)

            # Update job status
            old_status = job.status
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now(UTC)

            # Preserve partial results if requested
            preserved_results = None
            if preserve_partial and job.results:
                preserved_results = {"partial_results": job.results}

            logger.info(
                "Job %s cancelled (was %s), preserve_partial=%s",
                job_id,
                old_status,
                preserve_partial,
            )

            return CancelJobResponse(
                job_id=job_id,
                status=job.status.value,
                message=f"Job cancelled from {old_status} status",
                preserved_results=preserved_results,
            )

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job from storage."""
        async with self._lock:
            if job_id in self._jobs:
                # Ensure job is not running
                task = self._running_tasks.get(job_id)
                if task and not task.done():
                    task.cancel()
                    self._running_tasks.pop(job_id, None)

                del self._jobs[job_id]
                logger.info("Deleted job %s", job_id)
                return True
            return False

    async def register_task(self, job_id: str, task: asyncio.Task) -> None:
        """Register a running task for a job."""
        async with self._lock:
            self._running_tasks[job_id] = task

        # Set up callback to clean up when task completes
        def cleanup_task(task_result: asyncio.Task) -> None:
            asyncio.create_task(self._cleanup_completed_task(job_id))

        task.add_done_callback(cleanup_task)

    async def _cleanup_completed_task(self, job_id: str) -> None:
        """Remove completed task from tracking."""
        async with self._lock:
            self._running_tasks.pop(job_id, None)

    async def get_stats(self) -> Dict[str, Any]:
        """Get job manager statistics."""
        async with self._lock:
            status_counts = {}
            for job in self._jobs.values():
                status_counts[job.status] = status_counts.get(job.status, 0) + 1

            return {
                "total_jobs": len(self._jobs),
                "running_tasks": len(self._running_tasks),
                "status_counts": status_counts,
                "retention_hours": self._retention_hours,
                "max_jobs": self._max_jobs,
            }

    def _is_valid_transition(
        self, from_status: JobStatus, to_status: JobStatus
    ) -> bool:
        """Check if a status transition is valid."""
        valid_transitions = {
            JobStatus.PENDING: {JobStatus.RUNNING, JobStatus.CANCELLED},
            JobStatus.RUNNING: {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            },
            JobStatus.COMPLETED: set(),  # Terminal state
            JobStatus.FAILED: set(),  # Terminal state
            JobStatus.CANCELLED: set(),  # Terminal state
        }

        return to_status in valid_transitions.get(from_status, set())

    async def _enforce_storage_limits(self) -> None:
        """Enforce job storage limits by removing old jobs."""
        if len(self._jobs) < self._max_jobs:
            return

        # Sort jobs by creation time (oldest first)
        jobs_by_age = sorted(self._jobs.values(), key=lambda j: j.created_at)

        # Remove oldest jobs until under limit
        jobs_to_remove = len(self._jobs) - self._max_jobs + 1
        for job in jobs_by_age[:jobs_to_remove]:
            # Don't remove running jobs
            if job.status != JobStatus.RUNNING:
                del self._jobs[job.job_id]
                logger.info("Removed old job %s to enforce storage limits", job.job_id)

    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of expired jobs."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval_seconds)
                await self._cleanup_expired_jobs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in periodic cleanup: %s", e)

    async def _cleanup_expired_jobs(self) -> None:
        """Remove jobs that have exceeded retention time."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=self._retention_hours)
        removed_count = 0

        async with self._lock:
            jobs_to_remove = []
            for job_id, job in self._jobs.items():
                # Only remove completed/failed/cancelled jobs past retention
                if (
                    job.status
                    in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
                    and job.created_at < cutoff_time
                ):
                    jobs_to_remove.append(job_id)

            for job_id in jobs_to_remove:
                del self._jobs[job_id]
                removed_count += 1

        if removed_count > 0:
            logger.info("Cleanup removed %d expired jobs", removed_count)
