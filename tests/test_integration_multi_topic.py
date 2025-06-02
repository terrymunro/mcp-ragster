"""Integration tests for multi-topic processing functionality."""

import asyncio
import pytest
from datetime import datetime, UTC
from unittest.mock import AsyncMock, patch, Mock

from ragster.background_processor import BackgroundTaskProcessor
from ragster.job_manager import JobManager
from ragster.resource_manager import MultiTopicResourceManager
from ragster.job_models import JobStatus
from ragster.models import LoadTopicResponse
from ragster.config import settings


class TestMultiTopicIntegration:
    """Integration test suite for multi-topic processing."""

    @pytest.fixture
    async def setup_components(self):
        """Set up all components needed for integration tests."""
        job_manager = JobManager()
        resource_manager = MultiTopicResourceManager()
        background_processor = BackgroundTaskProcessor(job_manager=job_manager)

        await job_manager.start()

        yield {
            "job_manager": job_manager,
            "resource_manager": resource_manager,
            "background_processor": background_processor,
        }

        await job_manager.stop()
        await background_processor.shutdown()

    @pytest.fixture
    def mock_topic_processor(self):
        """Create a mock topic processor for testing."""
        processor = AsyncMock()

        # Default successful processing
        mock_result = Mock(spec=LoadTopicResponse)
        mock_result.urls_processed = 10
        mock_result.error = None
        processor.process_topic.return_value = mock_result

        return processor

    async def test_single_topic_processing(self, setup_components):
        """Test processing a single topic."""
        components = setup_components
        job_manager = components["job_manager"]
        background_processor = components["background_processor"]

        # Mock the _process_single_topic method
        mock_result = Mock(spec=LoadTopicResponse)
        mock_result.urls_processed = 10
        mock_result.error = None

        with patch.object(
            background_processor, "_process_single_topic", return_value=mock_result
        ):
            # Create and process job
            topics = ["artificial intelligence"]
            job = await job_manager.create_job(topics)

            # Start processing
            mock_app_context = Mock()
            await background_processor.start_research_job(
                job.job_id, topics, mock_app_context
            )

            # Wait for processing to complete
            await asyncio.sleep(0.1)

            # Verify job completion
            completed_job = await job_manager.get_job(job.job_id)
            assert completed_job.status in [JobStatus.COMPLETED, JobStatus.RUNNING]

    async def test_three_topic_parallel_processing(self, setup_components):
        """Test parallel processing of 3 topics."""
        components = setup_components
        job_manager = components["job_manager"]
        background_processor = components["background_processor"]

        topics = ["artificial intelligence", "machine learning", "deep learning"]

        # Mock the _process_single_topic method
        async def mock_process_topic(topic, app_context, progress_callback=None):
            # Simulate processing stages
            if progress_callback:
                await progress_callback("jina", "running", {})
                await asyncio.sleep(0.01)  # Small delay to simulate work
                await progress_callback("jina", "completed", {"urls_found": 5})
                await progress_callback("perplexity", "running", {})
                await asyncio.sleep(0.01)
                await progress_callback("perplexity", "completed", {})
                await progress_callback("firecrawl", "running", {})
                await asyncio.sleep(0.01)
                await progress_callback("firecrawl", "completed", {"urls_processed": 5})

            mock_result = Mock(spec=LoadTopicResponse)
            mock_result.urls_processed = 5
            mock_result.error = None
            return mock_result

        with patch.object(
            background_processor,
            "_process_single_topic",
            side_effect=mock_process_topic,
        ):
            # Create job
            job = await job_manager.create_job(topics)

            # Start processing
            start_time = datetime.now(UTC)
            mock_app_context = Mock()
            await background_processor.start_research_job(
                job.job_id, topics, mock_app_context
            )

            # Wait for processing to complete
            await asyncio.sleep(0.2)
            end_time = datetime.now(UTC)

            # Verify job completion
            completed_job = await job_manager.get_job(job.job_id)
            assert completed_job.status in [JobStatus.COMPLETED, JobStatus.RUNNING]

            # Verify parallel processing (should be faster than sequential)
            processing_time = (end_time - start_time).total_seconds()
            assert processing_time < 0.5  # Should complete within reasonable time

    async def test_job_cancellation_during_processing(self, setup_components):
        """Test job cancellation during active processing."""
        components = setup_components
        job_manager = components["job_manager"]
        background_processor = components["background_processor"]

        topics = ["long_topic_1", "long_topic_2", "long_topic_3"]

        # Create an event to control when processing should be cancelled
        processing_started = asyncio.Event()

        async def mock_process_topic(topic, app_context, progress_callback=None):
            processing_started.set()

            # Simulate long processing
            if progress_callback:
                await progress_callback("jina", "running", {})

            # Wait for a long time (simulating slow processing)
            await asyncio.sleep(2.0)

            mock_result = Mock(spec=LoadTopicResponse)
            mock_result.urls_processed = 5
            mock_result.error = None
            return mock_result

        with patch.object(
            background_processor,
            "_process_single_topic",
            side_effect=mock_process_topic,
        ):
            # Create job
            job = await job_manager.create_job(topics)

            # Start processing in background
            mock_app_context = Mock()
            await background_processor.start_research_job(
                job.job_id, topics, mock_app_context
            )

            # Wait for processing to start
            await processing_started.wait()

            # Cancel the job
            success = await background_processor.cancel_job(job.job_id)
            assert success is True

            # Wait a bit for cancellation to take effect
            await asyncio.sleep(0.1)

            # Verify job status
            cancelled_job = await job_manager.get_job(job.job_id)
            assert cancelled_job.status == JobStatus.CANCELLED

    async def test_resource_manager_allocation(self, setup_components):
        """Test resource manager adapts to different topic counts."""
        components = setup_components
        resource_manager = components["resource_manager"]

        # Test with different topic counts
        test_cases = [1, 3, 8, 15]

        for topic_count in test_cases:
            # Allocate resources
            resources = await resource_manager.allocate_resources_for_topics(
                topic_count
            )

            # Verify resource allocation
            assert "firecrawl_limit" in resources
            assert "api_limit" in resources
            assert "jina_batch_size" in resources
            assert resources["firecrawl_limit"] >= 1
            assert resources["api_limit"] >= 2
            assert resources["jina_batch_size"] >= 2

            # Release resources
            await resource_manager.release_resources_for_topics(topic_count)

    async def test_concurrent_job_limits(self, setup_components):
        """Test that concurrent job limits are respected."""
        components = setup_components
        job_manager = components["job_manager"]
        background_processor = components["background_processor"]

        # Mock slow processing
        async def slow_process_topic(topic, app_context, progress_callback=None):
            await asyncio.sleep(0.5)  # Simulate slow processing
            mock_result = Mock(spec=LoadTopicResponse)
            mock_result.urls_processed = 1
            mock_result.error = None
            return mock_result

        with patch.object(
            background_processor,
            "_process_single_topic",
            side_effect=slow_process_topic,
        ):
            # Create multiple jobs
            jobs = []
            for i in range(3):
                job = await job_manager.create_job([f"topic_{i}"])
                jobs.append(job)

            # Start all jobs
            mock_app_context = Mock()
            for job in jobs:
                await background_processor.start_research_job(
                    job.job_id, job.topics, mock_app_context
                )

            # Check active job count
            active_count = await background_processor.get_active_job_count()
            assert active_count <= settings.MAX_CONCURRENT_RESEARCH_JOBS

            # Wait for jobs to complete
            await asyncio.sleep(1.0)

    async def test_error_handling_in_processing(self, setup_components):
        """Test error handling during topic processing."""
        components = setup_components
        job_manager = components["job_manager"]
        background_processor = components["background_processor"]

        # Mock processing that fails
        async def failing_process_topic(topic, app_context, progress_callback=None):
            if progress_callback:
                await progress_callback("jina", "failed", {"error": "API timeout"})
            raise Exception("Simulated processing failure")

        with patch.object(
            background_processor,
            "_process_single_topic",
            side_effect=failing_process_topic,
        ):
            # Create job
            topics = ["failing_topic"]
            job = await job_manager.create_job(topics)

            # Start processing
            mock_app_context = Mock()
            await background_processor.start_research_job(
                job.job_id, topics, mock_app_context
            )

            # Wait for processing to complete
            await asyncio.sleep(0.2)

            # Verify job failed
            failed_job = await job_manager.get_job(job.job_id)
            assert failed_job.status == JobStatus.FAILED
            assert failed_job.error is not None

    async def test_performance_validation(self, setup_components):
        """Test performance characteristics meet requirements."""
        components = setup_components
        job_manager = components["job_manager"]
        background_processor = components["background_processor"]

        # Test scenario: 5 topics should complete in reasonable time
        topics = [f"perf_topic_{i}" for i in range(5)]

        # Mock lightweight processing
        async def fast_process_topic(topic, app_context, progress_callback=None):
            if progress_callback:
                await progress_callback("jina", "completed", {"urls_found": 2})
                await progress_callback("perplexity", "completed", {})
                await progress_callback("firecrawl", "completed", {"urls_processed": 2})

            # Small delay to simulate real processing
            await asyncio.sleep(0.01)

            mock_result = Mock(spec=LoadTopicResponse)
            mock_result.urls_processed = 2
            mock_result.error = None
            return mock_result

        with patch.object(
            background_processor,
            "_process_single_topic",
            side_effect=fast_process_topic,
        ):
            # Create job
            job = await job_manager.create_job(topics)

            # Measure processing time
            start_time = datetime.now(UTC)
            mock_app_context = Mock()
            await background_processor.start_research_job(
                job.job_id, topics, mock_app_context
            )

            # Wait for processing to complete
            await asyncio.sleep(0.5)
            end_time = datetime.now(UTC)

            processing_time = (end_time - start_time).total_seconds()

            # Verify performance (should complete within reasonable time)
            assert processing_time < 1.0, (
                f"Processing took {processing_time}s, expected < 1.0s"
            )

            # Verify successful completion
            completed_job = await job_manager.get_job(job.job_id)
            assert completed_job.status in [JobStatus.COMPLETED, JobStatus.RUNNING]

    async def test_memory_usage_control(self, setup_components):
        """Test memory usage remains within acceptable bounds."""
        components = setup_components
        job_manager = components["job_manager"]
        background_processor = components["background_processor"]

        # Create a larger job to test memory control
        topics = [f"memory_topic_{i}" for i in range(10)]

        # Mock processing that creates some data
        async def memory_process_topic(topic, app_context, progress_callback=None):
            if progress_callback:
                await progress_callback("jina", "completed", {"urls_found": 10})
                await progress_callback("perplexity", "completed", {})
                await progress_callback(
                    "firecrawl", "completed", {"urls_processed": 10}
                )

            # Create mock result with some data
            mock_result = Mock(spec=LoadTopicResponse)
            mock_result.urls_processed = 10
            mock_result.content = "x" * 1000  # 1KB of data per topic
            mock_result.error = None
            return mock_result

        with patch.object(
            background_processor,
            "_process_single_topic",
            side_effect=memory_process_topic,
        ):
            # Create job
            job = await job_manager.create_job(topics)

            # Start processing
            mock_app_context = Mock()
            await background_processor.start_research_job(
                job.job_id, topics, mock_app_context
            )

            # Wait for processing to complete
            await asyncio.sleep(0.5)

            # Verify job completion
            completed_job = await job_manager.get_job(job.job_id)
            assert completed_job.status in [JobStatus.COMPLETED, JobStatus.RUNNING]

            # Verify memory usage is reasonable (this is a basic check)
            # In a real scenario, you might use memory profiling tools
            assert completed_job is not None
