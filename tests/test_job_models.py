"""Unit tests for job models."""

import pytest
from datetime import datetime, UTC, timedelta

from ragster.job_models import (
    JobStatus,
    TopicProgress,
    JobMetrics,
    ResearchJob,
    ResearchJobResponse,
    JobStatusResponse,
    MultiTopicResponse,
    GetJobStatusArgs,
    ListJobsArgs,
    CancelJobArgs,
    ListJobsResponse,
    CancelJobResponse,
)
from ragster.models import LoadTopicResponse


class TestTopicProgress:
    """Test suite for TopicProgress dataclass."""

    def test_topic_progress_initialization(self):
        """Test TopicProgress initialization with defaults."""
        progress = TopicProgress(topic="test topic")

        assert progress.topic == "test topic"
        assert progress.jina_status == "pending"
        assert progress.perplexity_status == "pending"
        assert progress.firecrawl_status == "pending"
        assert progress.urls_found == 0
        assert progress.urls_processed == 0
        assert progress.errors == []

    def test_get_completion_percentage_all_pending(self):
        """Test completion percentage calculation when all stages are pending."""
        progress = TopicProgress(topic="test")
        assert progress.get_completion_percentage() == 0.0

    def test_get_completion_percentage_one_completed(self):
        """Test completion percentage calculation with one stage completed."""
        progress = TopicProgress(topic="test", jina_status="completed")
        assert progress.get_completion_percentage() == pytest.approx(1.0 / 3.0)

    def test_get_completion_percentage_two_completed(self):
        """Test completion percentage calculation with two stages completed."""
        progress = TopicProgress(
            topic="test", jina_status="completed", perplexity_status="completed"
        )
        assert progress.get_completion_percentage() == pytest.approx(2.0 / 3.0)

    def test_get_completion_percentage_all_completed(self):
        """Test completion percentage calculation when all stages are completed."""
        progress = TopicProgress(
            topic="test",
            jina_status="completed",
            perplexity_status="completed",
            firecrawl_status="completed",
        )
        assert progress.get_completion_percentage() == 1.0

    def test_get_completion_percentage_firecrawl_partial(self):
        """Test completion percentage with partial Firecrawl progress."""
        progress = TopicProgress(
            topic="test",
            jina_status="completed",
            perplexity_status="completed",
            firecrawl_status="running",
            urls_found=10,
            urls_processed=5,
        )
        # 1 + 1 + 0.5 = 2.5 out of 3
        assert progress.get_completion_percentage() == pytest.approx(2.5 / 3.0)

    def test_is_completed_false(self):
        """Test is_completed returns False when not all stages are done."""
        progress = TopicProgress(
            topic="test",
            jina_status="completed",
            perplexity_status="completed",
            firecrawl_status="running",
        )
        assert progress.is_completed() is False

    def test_is_completed_true(self):
        """Test is_completed returns True when all stages are done."""
        progress = TopicProgress(
            topic="test",
            jina_status="completed",
            perplexity_status="completed",
            firecrawl_status="completed",
        )
        assert progress.is_completed() is True

    def test_has_errors_false(self):
        """Test has_errors returns False when no errors."""
        progress = TopicProgress(topic="test")
        assert progress.has_errors() is False

    def test_has_errors_true(self):
        """Test has_errors returns True when errors exist."""
        progress = TopicProgress(topic="test", errors=["Some error"])
        assert progress.has_errors() is True


class TestJobMetrics:
    """Test suite for JobMetrics dataclass."""

    def test_job_metrics_initialization(self):
        """Test JobMetrics initialization with defaults."""
        metrics = JobMetrics()

        assert metrics.total_api_calls == 0
        assert metrics.jina_api_calls == 0
        assert metrics.perplexity_api_calls == 0
        assert metrics.firecrawl_api_calls == 0
        assert metrics.total_processing_time_seconds == 0.0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0

    def test_get_cache_hit_rate_no_requests(self):
        """Test cache hit rate calculation with no requests."""
        metrics = JobMetrics()
        assert metrics.get_cache_hit_rate() == 0.0

    def test_get_cache_hit_rate_all_hits(self):
        """Test cache hit rate calculation with all hits."""
        metrics = JobMetrics(cache_hits=10, cache_misses=0)
        assert metrics.get_cache_hit_rate() == 1.0

    def test_get_cache_hit_rate_all_misses(self):
        """Test cache hit rate calculation with all misses."""
        metrics = JobMetrics(cache_hits=0, cache_misses=10)
        assert metrics.get_cache_hit_rate() == 0.0

    def test_get_cache_hit_rate_mixed(self):
        """Test cache hit rate calculation with mixed results."""
        metrics = JobMetrics(cache_hits=7, cache_misses=3)
        assert metrics.get_cache_hit_rate() == 0.7


class TestResearchJob:
    """Test suite for ResearchJob dataclass."""

    def test_research_job_initialization(self):
        """Test ResearchJob initialization."""
        topics = ["AI", "ML", "DL"]
        job = ResearchJob(
            job_id="test-id",
            topics=topics,
            status=JobStatus.PENDING,
            created_at=datetime.now(UTC),
        )

        assert job.job_id == "test-id"
        assert job.topics == topics
        assert job.status == JobStatus.PENDING
        assert job.started_at is None
        assert job.completed_at is None
        assert job.results is None
        assert job.error is None
        assert isinstance(job.metrics, JobMetrics)

        # Check progress initialization
        assert len(job.progress) == 3
        for topic in topics:
            assert topic in job.progress
            assert isinstance(job.progress[topic], TopicProgress)
            assert job.progress[topic].topic == topic

    def test_get_overall_progress_empty(self):
        """Test overall progress calculation with no progress."""
        job = ResearchJob(
            job_id="test",
            topics=[],
            status=JobStatus.PENDING,
            created_at=datetime.now(UTC),
        )
        assert job.get_overall_progress() == 0.0

    def test_get_overall_progress_calculation(self):
        """Test overall progress calculation."""
        topics = ["AI", "ML"]
        job = ResearchJob(
            job_id="test",
            topics=topics,
            status=JobStatus.RUNNING,
            created_at=datetime.now(UTC),
        )

        # Set different completion levels for topics
        job.progress["AI"].jina_status = "completed"  # 1/3 completion
        job.progress["ML"].jina_status = "completed"
        job.progress["ML"].perplexity_status = "completed"  # 2/3 completion

        # Average: (1/3 + 2/3) / 2 = 0.5
        assert job.get_overall_progress() == pytest.approx(0.5)

    def test_get_estimated_completion_time_not_running(self):
        """Test estimated completion time for non-running job."""
        job = ResearchJob(
            job_id="test",
            topics=["AI"],
            status=JobStatus.PENDING,
            created_at=datetime.now(UTC),
        )
        assert job.get_estimated_completion_time() is None

    def test_get_estimated_completion_time_no_start_time(self):
        """Test estimated completion time without start time."""
        job = ResearchJob(
            job_id="test",
            topics=["AI"],
            status=JobStatus.RUNNING,
            created_at=datetime.now(UTC),
        )
        assert job.get_estimated_completion_time() is None

    def test_get_estimated_completion_time_no_progress(self):
        """Test estimated completion time with no progress."""
        now = datetime.now(UTC)
        job = ResearchJob(
            job_id="test",
            topics=["AI"],
            status=JobStatus.RUNNING,
            created_at=now,
            started_at=now,
        )
        assert job.get_estimated_completion_time() is None

    def test_get_estimated_completion_time_with_progress(self):
        """Test estimated completion time calculation with progress."""
        now = datetime.now(UTC)
        start_time = now - timedelta(minutes=10)  # Started 10 minutes ago

        job = ResearchJob(
            job_id="test",
            topics=["AI", "ML"],
            status=JobStatus.RUNNING,
            created_at=start_time,
            started_at=start_time,
        )

        # Complete one topic (50% progress)
        job.progress["AI"].jina_status = "completed"
        job.progress["AI"].perplexity_status = "completed"
        job.progress["AI"].firecrawl_status = "completed"

        # Mock datetime.utcnow to return our 'now'
        from unittest.mock import patch

        with patch("ragster.job_models.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value = now

            completion_time = job.get_estimated_completion_time()

            # Should estimate completion in about 10 more minutes (50% done, 50% remaining)
            assert completion_time is not None
            estimated_remaining = (completion_time - now).total_seconds()
            assert 500 <= estimated_remaining <= 700  # Allow some tolerance

    def test_get_completed_topics(self):
        """Test getting list of completed topics."""
        job = ResearchJob(
            job_id="test",
            topics=["AI", "ML", "DL"],
            status=JobStatus.RUNNING,
            created_at=datetime.now(UTC),
        )

        # Complete one topic
        job.progress["AI"].jina_status = "completed"
        job.progress["AI"].perplexity_status = "completed"
        job.progress["AI"].firecrawl_status = "completed"

        # Partially complete another
        job.progress["ML"].jina_status = "completed"

        completed = job.get_completed_topics()
        assert completed == ["AI"]

    def test_get_failed_topics(self):
        """Test getting list of failed topics."""
        job = ResearchJob(
            job_id="test",
            topics=["AI", "ML", "DL"],
            status=JobStatus.RUNNING,
            created_at=datetime.now(UTC),
        )

        # Add errors to some topics
        job.progress["AI"].errors.append("Jina API error")
        job.progress["DL"].errors.append("Firecrawl timeout")

        failed = job.get_failed_topics()
        assert set(failed) == {"AI", "DL"}

    def test_is_complete_false(self):
        """Test is_complete returns False when not all topics are done."""
        job = ResearchJob(
            job_id="test",
            topics=["AI", "ML"],
            status=JobStatus.RUNNING,
            created_at=datetime.now(UTC),
        )

        # Complete only one topic
        job.progress["AI"].jina_status = "completed"
        job.progress["AI"].perplexity_status = "completed"
        job.progress["AI"].firecrawl_status = "completed"

        assert job.is_complete() is False

    def test_is_complete_true(self):
        """Test is_complete returns True when all topics are done."""
        job = ResearchJob(
            job_id="test",
            topics=["AI", "ML"],
            status=JobStatus.RUNNING,
            created_at=datetime.now(UTC),
        )

        # Complete all topics
        for topic in ["AI", "ML"]:
            job.progress[topic].jina_status = "completed"
            job.progress[topic].perplexity_status = "completed"
            job.progress[topic].firecrawl_status = "completed"

        assert job.is_complete() is True

    def test_update_topic_progress_new_topic(self):
        """Test updating progress for a new topic."""
        job = ResearchJob(
            job_id="test",
            topics=["AI"],
            status=JobStatus.RUNNING,
            created_at=datetime.now(UTC),
        )

        # Update progress for new topic
        job.update_topic_progress("NEW", "jina", "completed", urls_found=5)

        assert "NEW" in job.progress
        progress = job.progress["NEW"]
        assert progress.topic == "NEW"
        assert progress.jina_status == "completed"
        assert progress.urls_found == 5

    def test_update_topic_progress_jina_stage(self):
        """Test updating Jina stage progress."""
        job = ResearchJob(
            job_id="test",
            topics=["AI"],
            status=JobStatus.RUNNING,
            created_at=datetime.now(UTC),
        )

        job.update_topic_progress("AI", "jina", "completed", urls_found=10)

        progress = job.progress["AI"]
        assert progress.jina_status == "completed"
        assert progress.urls_found == 10

    def test_update_topic_progress_perplexity_stage(self):
        """Test updating Perplexity stage progress."""
        job = ResearchJob(
            job_id="test",
            topics=["AI"],
            status=JobStatus.RUNNING,
            created_at=datetime.now(UTC),
        )

        job.update_topic_progress("AI", "perplexity", "completed")

        progress = job.progress["AI"]
        assert progress.perplexity_status == "completed"

    def test_update_topic_progress_firecrawl_stage(self):
        """Test updating Firecrawl stage progress."""
        job = ResearchJob(
            job_id="test",
            topics=["AI"],
            status=JobStatus.RUNNING,
            created_at=datetime.now(UTC),
        )

        job.update_topic_progress("AI", "firecrawl", "running", urls_processed=5)

        progress = job.progress["AI"]
        assert progress.firecrawl_status == "running"
        assert progress.urls_processed == 5

    def test_update_topic_progress_with_error(self):
        """Test updating topic progress with error."""
        job = ResearchJob(
            job_id="test",
            topics=["AI"],
            status=JobStatus.RUNNING,
            created_at=datetime.now(UTC),
        )

        error_msg = "API timeout"
        job.update_topic_progress("AI", "jina", "failed", error=error_msg)

        progress = job.progress["AI"]
        assert progress.jina_status == "failed"
        assert error_msg in progress.errors


class TestResponseModels:
    """Test suite for Pydantic response models."""

    def test_research_job_response(self):
        """Test ResearchJobResponse model."""
        now = datetime.now(UTC)
        response = ResearchJobResponse(
            job_id="test-id",
            status="pending",
            topics=["AI", "ML"],
            message="Job created successfully",
            created_at=now,
            estimated_completion_time=now + timedelta(minutes=30),
        )

        assert response.job_id == "test-id"
        assert response.status == "pending"
        assert response.topics == ["AI", "ML"]
        assert response.message == "Job created successfully"
        assert response.created_at == now
        assert response.estimated_completion_time == now + timedelta(minutes=30)

    def test_job_status_response(self):
        """Test JobStatusResponse model."""
        now = datetime.now(UTC)
        topic_progress = {
            "AI": {
                "jina_status": "completed",
                "perplexity_status": "running",
                "firecrawl_status": "pending",
            }
        }

        response = JobStatusResponse(
            job_id="test-id",
            status="running",
            topics=["AI"],
            overall_progress=0.33,
            topic_progress=topic_progress,
            estimated_completion_time=now + timedelta(minutes=20),
            results=None,
            error=None,
        )

        assert response.job_id == "test-id"
        assert response.status == "running"
        assert response.topics == ["AI"]
        assert response.overall_progress == 0.33
        assert response.topic_progress == topic_progress
        assert response.estimated_completion_time == now + timedelta(minutes=20)

    def test_multi_topic_response(self):
        """Test MultiTopicResponse model."""
        # Use a real LoadTopicResponse for type correctness
        real_result = LoadTopicResponse(
            message="Success", topic="AI", urls_processed=10
        )
        topic_results = {"AI": real_result}

        response = MultiTopicResponse(
            message="Research completed for 1 out of 2 topics",
            topics=["AI", "ML"],
            total_urls_processed=25,
            successful_topics=["AI"],
            failed_topics=["ML"],
            topic_results=topic_results,
            overall_errors=["ML research failed due to API timeout"],
        )

        assert response.message == "Research completed for 1 out of 2 topics"
        assert response.topics == ["AI", "ML"]
        assert response.total_urls_processed == 25
        assert response.successful_topics == ["AI"]
        assert response.failed_topics == ["ML"]
        assert response.topic_results == topic_results
        assert response.overall_errors == ["ML research failed due to API timeout"]

    def test_get_job_status_args(self):
        """Test GetJobStatusArgs model."""
        args = GetJobStatusArgs(job_id="test-id-123")
        assert args.job_id == "test-id-123"

    def test_list_jobs_args_defaults(self):
        """Test ListJobsArgs model with defaults."""
        args = ListJobsArgs(status_filter=None, limit=10, offset=0)
        assert args.status_filter is None
        assert args.limit == 10
        assert args.offset == 0

    def test_list_jobs_args_with_values(self):
        """Test ListJobsArgs model with custom values."""
        args = ListJobsArgs(status_filter="running", limit=20, offset=5)
        assert args.status_filter == "running"
        assert args.limit == 20
        assert args.offset == 5

    def test_list_jobs_args_validation(self):
        """Test ListJobsArgs validation."""
        # Test valid status filter
        args = ListJobsArgs(status_filter="pending", limit=10, offset=0)
        assert args.status_filter == "pending"

        # Test invalid status filter should raise
        with pytest.raises(ValueError):
            ListJobsArgs(status_filter="invalid_status", limit=10, offset=0)

    def test_cancel_job_args(self):
        """Test CancelJobArgs model."""
        args = CancelJobArgs(job_id="test-id-456")
        assert args.job_id == "test-id-456"

    def test_list_jobs_response(self):
        """Test ListJobsResponse model."""
        # Use a real JobStatusResponse for type correctness
        now = datetime.now(UTC)
        job_status = JobStatusResponse(
            job_id="test-id",
            status="running",
            topics=["AI"],
            overall_progress=0.5,
            topic_progress={
                "AI": {
                    "jina_status": "completed",
                    "perplexity_status": "running",
                    "firecrawl_status": "pending",
                }
            },
            estimated_completion_time=now,
            results=None,
            error=None,
        )
        response = ListJobsResponse(jobs=[job_status], total_count=1, has_more=False)

        assert response.jobs == [job_status]
        assert response.total_count == 1
        assert response.has_more is False

    def test_cancel_job_response(self):
        """Test CancelJobResponse model."""
        # Use a real LoadTopicResponse for type correctness
        real_result = LoadTopicResponse(
            message="Preserved", topic="AI", urls_processed=10
        )
        preserved_results = {"AI": real_result}

        response = CancelJobResponse(
            job_id="test-id",
            status="cancelled",
            message="Job cancelled successfully",
            preserved_results=preserved_results,
        )

        assert response.job_id == "test-id"
        assert response.status == "cancelled"
        assert response.message == "Job cancelled successfully"
        assert response.preserved_results == preserved_results
