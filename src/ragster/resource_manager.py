"""Resource management for adaptive API rate limiting and concurrency control."""

import asyncio
import logging
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class ResourceMetrics:
    """Metrics for tracking resource usage and performance."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_request_time: datetime | None = None
    throttle_events: int = 0


class MultiTopicResourceManager:
    """Manages resource allocation and rate limiting for multi-topic research jobs."""

    def __init__(self):
        """Initialize the resource manager with adaptive settings."""

        # Base concurrency limits
        self._base_firecrawl_limit = settings.MAX_CONCURRENT_FIRECRAWL
        self._base_api_limit = 5  # Base limit for API calls

        # Adaptive semaphores
        self._firecrawl_semaphore: asyncio.Semaphore | None = None
        self._jina_semaphore: asyncio.Semaphore | None = None
        self._perplexity_semaphore: asyncio.Semaphore | None = None

        # Resource metrics tracking
        self._metrics: Dict[str, ResourceMetrics] = {
            "firecrawl": ResourceMetrics(),
            "jina": ResourceMetrics(),
            "perplexity": ResourceMetrics(),
        }

        # Adaptive batching settings
        self._jina_batch_size = 5  # Start with conservative batch size
        self._max_jina_batch_size = 10
        self._min_jina_batch_size = 2

        # Load tracking
        self._active_topic_count = 0
        self._lock = asyncio.Lock()

    async def allocate_resources_for_topics(self, topic_count: int) -> Dict[str, Any]:
        """Allocate resources based on the number of topics being processed."""
        async with self._lock:
            self._active_topic_count += topic_count

            # Calculate adaptive limits based on topic count
            firecrawl_limit = max(
                1,
                min(
                    self._base_firecrawl_limit,
                    self._base_firecrawl_limit // max(1, topic_count // 2),
                ),
            )

            api_limit = max(
                2,
                min(
                    self._base_api_limit,
                    self._base_api_limit // max(1, topic_count // 3),
                ),
            )

            # Create or update semaphores
            self._firecrawl_semaphore = asyncio.Semaphore(firecrawl_limit)
            self._jina_semaphore = asyncio.Semaphore(api_limit)
            self._perplexity_semaphore = asyncio.Semaphore(api_limit)

            # Adjust batch sizes based on load
            if topic_count >= 5:
                self._jina_batch_size = max(
                    self._min_jina_batch_size, self._jina_batch_size // 2
                )
            elif topic_count <= 2:
                self._jina_batch_size = min(
                    self._max_jina_batch_size, self._jina_batch_size * 2
                )

            logger.info(
                f"Allocated resources for {topic_count} topics: "
                f"firecrawl={firecrawl_limit}, api={api_limit}, "
                f"jina_batch={self._jina_batch_size}"
            )

            return {
                "firecrawl_limit": firecrawl_limit,
                "api_limit": api_limit,
                "jina_batch_size": self._jina_batch_size,
                "firecrawl_semaphore": self._firecrawl_semaphore,
                "jina_semaphore": self._jina_semaphore,
                "perplexity_semaphore": self._perplexity_semaphore,
            }

    async def release_resources_for_topics(self, topic_count: int) -> None:
        """Release resources when topics complete."""
        async with self._lock:
            self._active_topic_count = max(0, self._active_topic_count - topic_count)
            logger.info(
                f"Released resources for {topic_count} topics. "
                f"Active topics: {self._active_topic_count}"
            )

    async def get_firecrawl_semaphore(self) -> asyncio.Semaphore:
        """Get the current Firecrawl semaphore."""
        if self._firecrawl_semaphore is None:
            await self.allocate_resources_for_topics(1)
        assert self._firecrawl_semaphore is not None
        return self._firecrawl_semaphore

    async def get_jina_semaphore(self) -> asyncio.Semaphore:
        """Get the current Jina API semaphore."""
        if self._jina_semaphore is None:
            await self.allocate_resources_for_topics(1)
        assert self._jina_semaphore is not None
        return self._jina_semaphore

    async def get_perplexity_semaphore(self) -> asyncio.Semaphore:
        """Get the current Perplexity API semaphore."""
        if self._perplexity_semaphore is None:
            await self.allocate_resources_for_topics(1)
        assert self._perplexity_semaphore is not None
        return self._perplexity_semaphore

    async def get_jina_batch_size(self) -> int:
        """Get the current adaptive Jina batch size."""
        return self._jina_batch_size

    async def record_request_metrics(
        self, service: str, success: bool, response_time: float
    ) -> None:
        """Record metrics for a service request."""
        if service not in self._metrics:
            return

        metrics = self._metrics[service]
        metrics.total_requests += 1
        metrics.last_request_time = datetime.utcnow()

        if success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1

        # Update rolling average response time
        if metrics.total_requests == 1:
            metrics.average_response_time = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            metrics.average_response_time = (
                alpha * response_time + (1 - alpha) * metrics.average_response_time
            )

    async def record_throttle_event(self, service: str) -> None:
        """Record a throttling event for adaptive adjustment."""
        if service not in self._metrics:
            return

        metrics = self._metrics[service]
        metrics.throttle_events += 1

        # Adaptive response to throttling
        if service == "jina" and self._jina_batch_size > self._min_jina_batch_size:
            async with self._lock:
                self._jina_batch_size = max(
                    self._min_jina_batch_size, int(self._jina_batch_size * 0.8)
                )
                logger.info(
                    f"Reduced Jina batch size to {self._jina_batch_size} "
                    f"due to throttling"
                )

    async def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource usage statistics."""
        stats = {
            "active_topics": self._active_topic_count,
            "jina_batch_size": self._jina_batch_size,
            "semaphore_limits": {},
            "metrics": {},
        }

        # Add semaphore information
        if self._firecrawl_semaphore:
            stats["semaphore_limits"]["firecrawl"] = self._firecrawl_semaphore._value
        if self._jina_semaphore:
            stats["semaphore_limits"]["jina"] = self._jina_semaphore._value
        if self._perplexity_semaphore:
            stats["semaphore_limits"]["perplexity"] = self._perplexity_semaphore._value

        # Add metrics
        for service, metrics in self._metrics.items():
            stats["metrics"][service] = {
                "total_requests": metrics.total_requests,
                "success_rate": (
                    metrics.successful_requests / metrics.total_requests
                    if metrics.total_requests > 0
                    else 0.0
                ),
                "average_response_time": metrics.average_response_time,
                "throttle_events": metrics.throttle_events,
                "last_request": (
                    metrics.last_request_time.isoformat()
                    if metrics.last_request_time
                    else None
                ),
            }

        return stats

    async def is_overloaded(self) -> bool:
        """Check if the system is currently overloaded."""
        # Simple heuristic: overloaded if too many throttle events recently
        recent_throttles = sum(
            metrics.throttle_events for metrics in self._metrics.values()
        )

        # Also check if we have too many active topics
        return (
            recent_throttles > 5
            or self._active_topic_count > settings.MAX_CONCURRENT_RESEARCH_JOBS * 3
        )

    async def suggest_processing_strategy(self, topic_count: int) -> str:
        """Suggest processing strategy based on current load and topic count."""
        is_overloaded = await self.is_overloaded()

        if is_overloaded or topic_count <= 2:
            return "sequential"
        elif topic_count <= 5:
            return "parallel_conservative"
        else:
            return "parallel_aggressive"
