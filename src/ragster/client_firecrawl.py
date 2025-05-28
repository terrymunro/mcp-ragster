import asyncio
import logging
import time
from collections import defaultdict
from typing import Any
from urllib.parse import urlparse

from firecrawl import FirecrawlApp

from ragster.config import settings
from ragster.exceptions import FirecrawlError


logger = logging.getLogger(__name__)


class FirecrawlAPIClient:
    """Client for Firecrawl API."""

    def __init__(self):
        logger.info("Initializing FirecrawlAPIClient.")
        self.client: FirecrawlApp | None = None

        try:
            if settings.FIRECRAWL_API_KEY:
                self.client = FirecrawlApp(api_key=settings.FIRECRAWL_API_KEY)
                logger.info("Firecrawl client initialized with API key.")
            elif settings.FIRECRAWL_API_URL:
                self.client = FirecrawlApp(
                    api_key=None, api_url=settings.FIRECRAWL_API_URL
                )
                logger.info(
                    f"Firecrawl client initialized with base URL: {settings.FIRECRAWL_API_URL}"
                )
        except Exception as e:
            logger.critical(
                f"Failed to initialize FirecrawlApp client: {e}", exc_info=True
            )
            raise FirecrawlError(
                f"Failed to initialize FirecrawlApp client: {e}",
                underlying_error=e,
                status_code=500,
            )

    async def crawl_url(self, url: str) -> dict[str, Any]:
        """Scrape a single URL using Firecrawl."""
        if not self.client:
            raise FirecrawlError(
                "Firecrawl client not available. Initialization failed earlier.",
                status_code=500,
            )
        try:
            logger.info(f"Scraping URL with Firecrawl: {url}")
            # Use scrape_url for single URL scraping, not crawl_url
            scrape_result = self.client.scrape_url(
                url=url,
                params={"formats": ["markdown"], "onlyMainContent": True},
            )
            
            # Log the result structure for debugging
            logger.debug(f"Firecrawl scrape result type: {type(scrape_result)}")
            logger.debug(f"Firecrawl scrape result keys: {list(scrape_result.keys()) if isinstance(scrape_result, dict) else 'Not a dict'}")
            
            if not (scrape_result and isinstance(scrape_result, dict)):
                raise FirecrawlError(
                    f"Unexpected Firecrawl result format for {url}: {type(scrape_result)}"
                )

            # Look for content in order of preference directly from the result
            content_key_priority = ["markdown", "content", "html"]
            for key in content_key_priority:
                if key in scrape_result and scrape_result[key]:
                    content = scrape_result[key]
                    if isinstance(content, str) and content.strip():  # Ensure content is non-empty string
                        return {
                            "content": content,
                            "source_url": url,
                            "type": key,
                        }
            
            # If we get here, no usable content was found
            available_content_info = []
            for key in content_key_priority:
                if key in scrape_result:
                    value = scrape_result[key]
                    if isinstance(value, str):
                        available_content_info.append(f"{key}: {len(value)} chars")
                    else:
                        available_content_info.append(f"{key}: {type(value)}")
            
            raise FirecrawlError(
                f"Firecrawl result for {url} has no usable content. Content info: {available_content_info}. All keys: {list(scrape_result.keys())}"
            )
        except Exception as e:
            if isinstance(e, FirecrawlError):
                raise
            raise FirecrawlError(
                f"Error scraping URL {url} with Firecrawl: {e}", underlying_error=e
            )

    async def close(self):
        """Cleanup method for consistency."""
        pass


class FirecrawlBatcher:
    """Manages batched Firecrawl requests with domain grouping and deduplication."""

    def __init__(self, client: FirecrawlAPIClient, max_concurrent: int = 3):
        self.client = client
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # URL deduplication cache with timestamps
        self._url_cache: dict[str, tuple[dict[str, Any], float]] = {}
        self._cache_ttl = settings.FIRECRAWL_CACHE_TTL_SECONDS

        # Retry queue with backoff - (url, attempts, next_retry_time)
        self._retry_queue: list[tuple[str, int, float]] = []
        self._max_retries = settings.FIRECRAWL_MAX_RETRIES
        self._base_backoff = settings.FIRECRAWL_BASE_BACKOFF

        # Metrics tracking
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "successful_crawls": 0,
            "failed_crawls": 0,
            "retries": 0,
            "domains_processed": set(),
        }

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL for grouping."""
        parsed = urlparse(url)
        return parsed.netloc.lower()

    def _is_cached(self, url: str) -> tuple[bool, dict[str, Any] | None]:
        """Check if URL result is cached and not expired."""
        if url in self._url_cache:
            result, timestamp = self._url_cache[url]
            if time.time() - timestamp < self._cache_ttl:
                self.metrics["cache_hits"] += 1
                return True, result
            else:
                # Expired, remove from cache
                del self._url_cache[url]
        return False, None

    async def _crawl_with_retry(self, url: str) -> dict[str, Any] | None:
        """Scrape URL with retry logic."""
        attempts = 0
        while attempts < self._max_retries:
            try:
                async with self.semaphore:
                    result = await self.client.crawl_url(url)
                    self.metrics["successful_crawls"] += 1
                    # Cache the result
                    self._url_cache[url] = (result, time.time())
                    return result
            except FirecrawlError as e:
                attempts += 1
                self.metrics["retries"] += 1
                if attempts >= self._max_retries:
                    self.metrics["failed_crawls"] += 1
                    logger.error(
                        f"Failed to scrape {url} after {attempts} attempts: {e}"
                    )
                    return None
                # Exponential backoff
                backoff = self._base_backoff**attempts
                logger.warning(
                    f"Retrying {url} after {backoff}s (attempt {attempts}/{self._max_retries})"
                )
                await asyncio.sleep(backoff)
        return None

    async def crawl_urls(
        self, urls: list[str]
    ) -> list[tuple[str, dict[str, Any] | None]]:
        """Scrape multiple URLs with domain-based grouping and caching."""
        # Group URLs by domain
        domain_groups = defaultdict(list)
        results: list[tuple[str, dict[str, Any] | None]] = []

        for url in urls:
            self.metrics["total_requests"] += 1

            # Check cache first
            is_cached, cached_result = self._is_cached(url)
            if is_cached:
                results.append((url, cached_result))
                continue

            # Group by domain for sequential processing
            domain = self._get_domain(url)
            domain_groups[domain].append(url)
            self.metrics["domains_processed"].add(domain)

        # Process each domain group
        crawl_tasks = []
        for domain, domain_urls in domain_groups.items():
            # Create a task for each domain group
            crawl_tasks.append(self._process_domain_urls(domain_urls))

        # Process all domain groups concurrently
        if crawl_tasks:
            domain_results = await asyncio.gather(*crawl_tasks)
            for domain_result in domain_results:
                results.extend(domain_result)

        return results

    async def _process_domain_urls(
        self, urls: list[str]
    ) -> list[tuple[str, dict[str, Any] | None]]:
        """Process URLs from the same domain sequentially to avoid rate limiting."""
        results = []
        for i, url in enumerate(urls):
            if i > 0:
                # Small delay between same-domain requests
                await asyncio.sleep(settings.FIRECRAWL_SAME_DOMAIN_DELAY)
            result = await self._crawl_with_retry(url)
            results.append((url, result))
        return results

    def get_metrics(self) -> dict[str, Any]:
        """Return current metrics."""
        return {
            **self.metrics,
            "cache_size": len(self._url_cache),
            "domains_processed": len(self.metrics["domains_processed"]),
            "cache_hit_rate": self.metrics["cache_hits"]
            / max(1, self.metrics["total_requests"]),
        }
