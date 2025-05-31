"""Result caching system for topic research with TTL-based expiration."""

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, field

from .models import LoadTopicResponse

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached topic result with metadata."""

    topic_hash: str
    topic: str
    result: LoadTopicResponse
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=lambda: datetime.utcnow())

    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        return datetime.utcnow() > self.expires_at

    def update_access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


class TopicResultCache:
    """LRU cache with TTL for topic research results."""

    def __init__(self, max_size: int = 100, ttl_hours: int = 24):
        """Initialize the cache with size and TTL limits."""
        self._max_size = max_size
        self._ttl_hours = ttl_hours
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: list[str] = []  # For LRU tracking
        self._lock = asyncio.Lock()

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval_seconds = 3600  # 1 hour

        logger.info(
            f"TopicResultCache initialized: max_size={max_size}, ttl_hours={ttl_hours}"
        )

    async def start(self) -> None:
        """Start the cache cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("Cache cleanup task started")

    async def stop(self) -> None:
        """Stop the cache cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Cache cleanup task stopped")

    def _generate_topic_hash(self, topic: str) -> str:
        """Generate a hash for a topic to use as cache key."""
        # Normalize topic: lowercase, strip whitespace, remove common words
        normalized = topic.lower().strip()

        # Create hash from normalized topic
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    async def get(self, topic: str) -> Optional[LoadTopicResponse]:
        """Get a cached result for a topic."""
        topic_hash = self._generate_topic_hash(topic)

        async with self._lock:
            entry = self._cache.get(topic_hash)

            if entry is None:
                logger.debug(f"Cache miss for topic: {topic}")
                return None

            if entry.is_expired():
                logger.debug(f"Cache entry expired for topic: {topic}")
                del self._cache[topic_hash]
                if topic_hash in self._access_order:
                    self._access_order.remove(topic_hash)
                return None

            # Update access statistics and LRU order
            entry.update_access()
            self._update_access_order(topic_hash)

            logger.info(
                f"Cache hit for topic: {topic} (access count: {entry.access_count})"
            )
            return entry.result

    async def put(self, topic: str, result: LoadTopicResponse) -> None:
        """Cache a result for a topic."""
        topic_hash = self._generate_topic_hash(topic)

        async with self._lock:
            # Check if we need to evict entries
            if len(self._cache) >= self._max_size and topic_hash not in self._cache:
                await self._evict_lru()

            # Create new cache entry
            expires_at = datetime.utcnow() + timedelta(hours=self._ttl_hours)
            entry = CacheEntry(
                topic_hash=topic_hash,
                topic=topic,
                result=result,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
            )

            self._cache[topic_hash] = entry
            self._update_access_order(topic_hash)

            logger.info(f"Cached result for topic: {topic} (expires: {expires_at})")

    async def invalidate(self, topic: str) -> bool:
        """Invalidate a cached result for a topic."""
        topic_hash = self._generate_topic_hash(topic)

        async with self._lock:
            if topic_hash in self._cache:
                del self._cache[topic_hash]
                if topic_hash in self._access_order:
                    self._access_order.remove(topic_hash)
                logger.info(f"Invalidated cache for topic: {topic}")
                return True
            return False

    async def clear(self) -> None:
        """Clear all cached results."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._access_order.clear()
            logger.info(f"Cleared {count} entries from cache")

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            expired_count = sum(
                1 for entry in self._cache.values() if entry.is_expired()
            )

            total_access_count = sum(
                entry.access_count for entry in self._cache.values()
            )

            # Calculate cache hit rate (approximate)
            hit_rate = 0.0
            if total_access_count > 0:
                # This is an approximation since we don't track misses separately
                hit_rate = min(1.0, total_access_count / max(1, len(self._cache) * 2))

            return {
                "total_entries": len(self._cache),
                "expired_entries": expired_count,
                "max_size": self._max_size,
                "ttl_hours": self._ttl_hours,
                "total_access_count": total_access_count,
                "estimated_hit_rate": hit_rate,
                "cache_utilization": len(self._cache) / self._max_size,
                "oldest_entry": (
                    min(entry.created_at for entry in self._cache.values()).isoformat()
                    if self._cache
                    else None
                ),
                "newest_entry": (
                    max(entry.created_at for entry in self._cache.values()).isoformat()
                    if self._cache
                    else None
                ),
            }

    async def get_cached_topics(self) -> list[str]:
        """Get a list of all currently cached topics."""
        async with self._lock:
            return [
                entry.topic for entry in self._cache.values() if not entry.is_expired()
            ]

    async def check_cache_hits(self, topics: list[str]) -> Tuple[list[str], list[str]]:
        """Check which topics have cache hits and which need processing."""
        cache_hits = []
        cache_misses = []

        for topic in topics:
            cached_result = await self.get(topic)
            if cached_result is not None:
                cache_hits.append(topic)
            else:
                cache_misses.append(topic)

        return cache_hits, cache_misses

    def _update_access_order(self, topic_hash: str) -> None:
        """Update the LRU access order for a topic hash."""
        if topic_hash in self._access_order:
            self._access_order.remove(topic_hash)
        self._access_order.append(topic_hash)

    async def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if not self._access_order:
            return

        lru_hash = self._access_order.pop(0)
        if lru_hash in self._cache:
            topic = self._cache[lru_hash].topic
            del self._cache[lru_hash]
            logger.info(f"Evicted LRU entry for topic: {topic}")

    async def _cleanup_expired_entries(self) -> int:
        """Clean up expired cache entries."""
        expired_hashes = []

        async with self._lock:
            for topic_hash, entry in self._cache.items():
                if entry.is_expired():
                    expired_hashes.append(topic_hash)

            for topic_hash in expired_hashes:
                entry = self._cache[topic_hash]
                del self._cache[topic_hash]
                if topic_hash in self._access_order:
                    self._access_order.remove(topic_hash)
                logger.debug(f"Cleaned up expired entry for topic: {entry.topic}")

        if expired_hashes:
            logger.info(f"Cleaned up {len(expired_hashes)} expired cache entries")

        return len(expired_hashes)

    async def _periodic_cleanup(self) -> None:
        """Periodic task to clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval_seconds)
                await self._cleanup_expired_entries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait a minute before retrying
