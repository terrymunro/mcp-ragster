"""Unit tests for job management system."""

import asyncio
import pytest
from datetime import datetime, UTC, timedelta
from unittest.mock import patch

from ragster.job_manager import (
    JobManager,
    JobNotFoundError,
    JobStateTransitionError,
    JobManagerError,
)
from ragster.job_models import JobStatus
from ragster.config import settings


class TestJobManager:
    """Test suite for JobManager class."""

    @pytest.fixture
    async def job_manager(self):
        """Create a fresh JobManager instance for each test."""
        manager = JobManager()
        await manager.start()
        yield manager
        await manager.stop()

    @pytest.fixture
    def sample_topics(self):
        """Sample topics for testing."""
        return ["artificial intelligence", "machine learning", "deep learning"]

    async def test_create_job_success(self, job_manager, sample_topics):
        """Test successful job creation."""
        job = await job_manager.create_job(sample_topics)

        assert job.job_id is not None
        assert len(job.job_id) == 36  # UUID4 length
        assert job.topics == sample_topics
        assert job.status == JobStatus.PENDING
        assert job.created_at is not None
        assert job.started_at is None
        assert job.completed_at is None
        assert len(job.progress) == len(sample_topics)

        # Verify topic progress initialization
        for topic in sample_topics:
            assert topic in job.progress
            progress = job.progress[topic]
            assert progress.topic == topic
            assert progress.jina_status == "pending"
            assert progress.perplexity_status == "pending"
            assert progress.firecrawl_status == "pending"

    async def test_create_job_empty_topics(self, job_manager):
        """Test job creation with empty topics list fails."""
        with pytest.raises(
            JobManagerError, match="Cannot create job with empty topics list"
        ):
            await job_manager.create_job([])

    async def test_create_job_too_many_topics(self, job_manager):
        """Test job creation with too many topics fails."""
        max_topics = settings.MAX_TOPICS_PER_JOB
        too_many_topics = [f"topic_{i}" for i in range(max_topics + 1)]

        with pytest.raises(
            JobManagerError, match=f"Too many topics: {max_topics + 1} > {max_topics}"
        ):
            await job_manager.create_job(too_many_topics)

    async def test_get_job_success(self, job_manager, sample_topics):
        """Test successful job retrieval."""
        created_job = await job_manager.create_job(sample_topics)
        retrieved_job = await job_manager.get_job(created_job.job_id)

        assert retrieved_job.job_id == created_job.job_id
        assert retrieved_job.topics == created_job.topics
        assert retrieved_job.status == created_job.status

    async def test_get_job_not_found(self, job_manager):
        """Test job retrieval with invalid ID fails."""
        with pytest.raises(JobNotFoundError, match="Job not found: invalid-id"):
            await job_manager.get_job("invalid-id")

    async def test_update_job_status_success(self, job_manager, sample_topics):
        """Test successful job status updates."""
        job = await job_manager.create_job(sample_topics)

        # Test PENDING -> RUNNING transition
        updated_job = await job_manager.update_job_status(job.job_id, JobStatus.RUNNING)
        assert updated_job.status == JobStatus.RUNNING
        assert updated_job.started_at is not None
        assert updated_job.completed_at is None

        # Test RUNNING -> COMPLETED transition
        updated_job = await job_manager.update_job_status(
            job.job_id, JobStatus.COMPLETED
        )
        assert updated_job.status == JobStatus.COMPLETED
        assert updated_job.completed_at is not None

    async def test_update_job_status_with_error(self, job_manager, sample_topics):
        """Test job status update with error message."""
        job = await job_manager.create_job(sample_topics)
        error_msg = "Test error message"

        # First transition to RUNNING, then to FAILED
        await job_manager.update_job_status(job.job_id, JobStatus.RUNNING)
        updated_job = await job_manager.update_job_status(
            job.job_id, JobStatus.FAILED, error=error_msg
        )
        assert updated_job.status == JobStatus.FAILED
        assert updated_job.error == error_msg
        assert updated_job.completed_at is not None

    async def test_update_job_status_invalid_transition(
        self, job_manager, sample_topics
    ):
        """Test invalid job status transitions fail."""
        job = await job_manager.create_job(sample_topics)

        # Complete the job first
        await job_manager.update_job_status(job.job_id, JobStatus.RUNNING)
        await job_manager.update_job_status(job.job_id, JobStatus.COMPLETED)

        # Try to transition from COMPLETED back to RUNNING (invalid)
        with pytest.raises(JobStateTransitionError):
            await job_manager.update_job_status(job.job_id, JobStatus.RUNNING)

    async def test_update_job_status_not_found(self, job_manager):
        """Test job status update with invalid ID fails."""
        with pytest.raises(JobNotFoundError, match="Job not found: invalid-id"):
            await job_manager.update_job_status("invalid-id", JobStatus.RUNNING)

    async def test_update_topic_progress(self, job_manager, sample_topics):
        """Test topic progress updates."""
        job = await job_manager.create_job(sample_topics)
        topic = sample_topics[0]

        # Update Jina progress
        updated_job = await job_manager.update_topic_progress(
            job.job_id, topic, "jina", "completed", urls_found=5
        )
        progress = updated_job.progress[topic]
        assert progress.jina_status == "completed"
        assert progress.urls_found == 5

        # Update Perplexity progress
        await job_manager.update_topic_progress(
            job.job_id, topic, "perplexity", "completed"
        )
        updated_job = await job_manager.get_job(job.job_id)
        assert updated_job.progress[topic].perplexity_status == "completed"

        # Update Firecrawl progress
        await job_manager.update_topic_progress(
            job.job_id, topic, "firecrawl", "running", urls_processed=3
        )
        updated_job = await job_manager.get_job(job.job_id)
        assert updated_job.progress[topic].firecrawl_status == "running"
        assert updated_job.progress[topic].urls_processed == 3

    async def test_update_topic_progress_with_error(self, job_manager, sample_topics):
        """Test topic progress update with error."""
        job = await job_manager.create_job(sample_topics)
        topic = sample_topics[0]
        error_msg = "Jina API error"

        await job_manager.update_topic_progress(
            job.job_id, topic, "jina", "failed", error=error_msg
        )

        updated_job = await job_manager.get_job(job.job_id)
        progress = updated_job.progress[topic]
        assert progress.jina_status == "failed"
        assert error_msg in progress.errors

    async def test_auto_complete_job(self, job_manager, sample_topics):
        """Test automatic job completion when all topics are done."""
        job = await job_manager.create_job(sample_topics)
        await job_manager.update_job_status(job.job_id, JobStatus.RUNNING)

        # Complete all topics
        for topic in sample_topics:
            await job_manager.update_topic_progress(
                job.job_id, topic, "jina", "completed"
            )
            await job_manager.update_topic_progress(
                job.job_id, topic, "perplexity", "completed"
            )
            await job_manager.update_topic_progress(
                job.job_id, topic, "firecrawl", "completed"
            )

        # Job should auto-complete
        updated_job = await job_manager.get_job(job.job_id)
        assert updated_job.status == JobStatus.COMPLETED
        assert updated_job.completed_at is not None

    async def test_list_jobs_empty(self, job_manager):
        """Test listing jobs when none exist."""
        response = await job_manager.list_jobs()
        assert response.jobs == []
        assert response.total_count == 0
        assert response.has_more is False

    async def test_list_jobs_with_jobs(self, job_manager, sample_topics):
        """Test listing jobs with existing jobs."""
        # Create multiple jobs
        job1 = await job_manager.create_job(sample_topics[:1])
        job2 = await job_manager.create_job(sample_topics[:2])
        await job_manager.update_job_status(job1.job_id, JobStatus.RUNNING)

        response = await job_manager.list_jobs()
        assert len(response.jobs) == 2
        assert response.total_count == 2
        assert response.has_more is False

        # Check jobs are sorted by creation time (newest first)
        assert response.jobs[0].job_id == job2.job_id
        assert response.jobs[1].job_id == job1.job_id

    async def test_list_jobs_with_status_filter(self, job_manager, sample_topics):
        """Test listing jobs with status filter."""
        # Create jobs with different statuses
        job1 = await job_manager.create_job(sample_topics[:1])
        job2 = await job_manager.create_job(sample_topics[:2])
        await job_manager.update_job_status(job1.job_id, JobStatus.RUNNING)

        # Filter by PENDING status
        response = await job_manager.list_jobs(status_filter="pending")
        assert len(response.jobs) == 1
        assert response.jobs[0].job_id == job2.job_id
        assert response.total_count == 1

        # Filter by RUNNING status
        response = await job_manager.list_jobs(status_filter="running")
        assert len(response.jobs) == 1
        assert response.jobs[0].job_id == job1.job_id
        assert response.total_count == 1

    async def test_list_jobs_pagination(self, job_manager, sample_topics):
        """Test job listing pagination."""
        # Create multiple jobs
        jobs = []
        for i in range(5):
            job = await job_manager.create_job([f"topic_{i}"])
            jobs.append(job)

        # Test first page
        response = await job_manager.list_jobs(limit=2, offset=0)
        assert len(response.jobs) == 2
        assert response.total_count == 5
        assert response.has_more is True

        # Test second page
        response = await job_manager.list_jobs(limit=2, offset=2)
        assert len(response.jobs) == 2
        assert response.total_count == 5
        assert response.has_more is True

        # Test last page
        response = await job_manager.list_jobs(limit=2, offset=4)
        assert len(response.jobs) == 1
        assert response.total_count == 5
        assert response.has_more is False

    async def test_cancel_job_success(self, job_manager, sample_topics):
        """Test successful job cancellation."""
        job = await job_manager.create_job(sample_topics)
        await job_manager.update_job_status(job.job_id, JobStatus.RUNNING)

        # Partially complete one topic
        await job_manager.update_topic_progress(
            job.job_id, sample_topics[0], "jina", "completed"
        )

        response = await job_manager.cancel_job(job.job_id, preserve_partial=True)
        assert response.job_id == job.job_id
        assert response.status == "cancelled"
        assert "cancelled" in response.message.lower()

        # Check job status was updated
        updated_job = await job_manager.get_job(job.job_id)
        assert updated_job.status == JobStatus.CANCELLED

    async def test_cancel_job_not_found(self, job_manager):
        """Test cancelling non-existent job fails."""
        with pytest.raises(JobNotFoundError, match="Job not found: invalid-id"):
            await job_manager.cancel_job("invalid-id")

    async def test_cancel_job_already_completed(self, job_manager, sample_topics):
        """Test cancelling already completed job fails."""
        job = await job_manager.create_job(sample_topics)
        await job_manager.update_job_status(job.job_id, JobStatus.RUNNING)
        await job_manager.update_job_status(job.job_id, JobStatus.COMPLETED)

        with pytest.raises(JobStateTransitionError):
            await job_manager.cancel_job(job.job_id)

    async def test_delete_job_success(self, job_manager, sample_topics):
        """Test successful job deletion."""
        job = await job_manager.create_job(sample_topics)

        result = await job_manager.delete_job(job.job_id)
        assert result is True

        # Job should no longer exist
        with pytest.raises(JobNotFoundError):
            await job_manager.get_job(job.job_id)

    async def test_delete_job_not_found(self, job_manager):
        """Test deleting non-existent job returns False."""
        result = await job_manager.delete_job("invalid-id")
        assert result is False

    async def test_register_task(self, job_manager, sample_topics):
        """Test task registration and cleanup."""
        job = await job_manager.create_job(sample_topics)

        # Create a mock task
        async def dummy_task():
            await asyncio.sleep(0.1)
            return "done"

        task = asyncio.create_task(dummy_task())
        await job_manager.register_task(job.job_id, task)

        # Wait for task completion
        result = await task
        assert result == "done"

        # Give some time for cleanup
        await asyncio.sleep(0.05)

    async def test_get_stats(self, job_manager, sample_topics):
        """Test job statistics retrieval."""
        # Create jobs with different statuses
        job1 = await job_manager.create_job(sample_topics[:1])
        job2 = await job_manager.create_job(sample_topics[:2])
        await job_manager.update_job_status(job1.job_id, JobStatus.RUNNING)
        await job_manager.update_job_status(job2.job_id, JobStatus.RUNNING)
        await job_manager.update_job_status(job2.job_id, JobStatus.COMPLETED)

        stats = await job_manager.get_stats()

        assert stats["total_jobs"] == 2
        assert stats["running_tasks"] == 0
        # Check status counts
        status_counts = stats["status_counts"]
        assert status_counts.get(JobStatus.RUNNING, 0) == 1
        assert status_counts.get(JobStatus.COMPLETED, 0) == 1

    @patch("ragster.job_manager.settings")
    async def test_storage_limits_enforcement(self, mock_settings, sample_topics):
        """Test storage limits are enforced."""
        mock_settings.MAX_STORED_JOBS = 2
        mock_settings.JOB_RETENTION_HOURS = 24
        mock_settings.MAX_TOPICS_PER_JOB = 10

        manager = JobManager()
        await manager.start()

        try:
            # Create max number of jobs
            job1 = await manager.create_job(sample_topics[:1])
            job2 = await manager.create_job(sample_topics[:2])

            # Creating a third job should trigger cleanup
            job3 = await manager.create_job(sample_topics[:1])

            # The oldest job should be removed
            with pytest.raises(JobNotFoundError):
                await manager.get_job(job1.job_id)

            # Other jobs should still exist
            await manager.get_job(job2.job_id)
            await manager.get_job(job3.job_id)
        finally:
            await manager.stop()

    async def test_cleanup_expired_jobs(self, job_manager, sample_topics):
        """Test cleanup of expired jobs."""
        # Create a job and complete it to make it eligible for cleanup
        job = await job_manager.create_job(sample_topics)
        await job_manager.update_job_status(job.job_id, JobStatus.RUNNING)
        await job_manager.update_job_status(job.job_id, JobStatus.COMPLETED)

        # Manually modify the creation time to make it expired
        async with job_manager._lock:
            stored_job = job_manager._jobs[job.job_id]
            stored_job.created_at = datetime.now(UTC) - timedelta(
                hours=settings.JOB_RETENTION_HOURS + 1
            )

        # Trigger cleanup
        await job_manager._cleanup_expired_jobs()

        # Job should be removed
        with pytest.raises(JobNotFoundError):
            await job_manager.get_job(job.job_id)

    async def test_concurrent_job_operations(self, job_manager, sample_topics):
        """Test concurrent job operations don't cause race conditions."""

        async def create_and_update_job(topic_suffix):
            job = await job_manager.create_job([f"topic_{topic_suffix}"])
            await job_manager.update_job_status(job.job_id, JobStatus.RUNNING)
            await job_manager.update_topic_progress(
                job.job_id, f"topic_{topic_suffix}", "jina", "completed"
            )
            return job.job_id

        # Run multiple concurrent operations
        tasks = [create_and_update_job(i) for i in range(10)]
        job_ids = await asyncio.gather(*tasks)

        # All jobs should exist and be in correct state
        for job_id in job_ids:
            job = await job_manager.get_job(job_id)
            assert job.status == JobStatus.RUNNING
            # Check that at least one topic has jina completed
            topic_progresses = list(job.progress.values())
            assert any(p.jina_status == "completed" for p in topic_progresses)

    async def test_state_transition_validation(self, job_manager, sample_topics):
        """Test comprehensive state transition validation."""
        job = await job_manager.create_job(sample_topics)

        # Valid transitions from PENDING
        await job_manager.update_job_status(job.job_id, JobStatus.RUNNING)

        # Reset to test other transitions
        job = await job_manager.create_job(sample_topics)
        await job_manager.update_job_status(job.job_id, JobStatus.RUNNING)
        await job_manager.update_job_status(job.job_id, JobStatus.FAILED)

        # Invalid transition from FAILED back to RUNNING
        with pytest.raises(JobStateTransitionError):
            await job_manager.update_job_status(job.job_id, JobStatus.RUNNING)

        # Test CANCELLED state
        job = await job_manager.create_job(sample_topics)
        await job_manager.update_job_status(job.job_id, JobStatus.CANCELLED)

        # Invalid transition from CANCELLED
        with pytest.raises(JobStateTransitionError):
            await job_manager.update_job_status(job.job_id, JobStatus.RUNNING)
