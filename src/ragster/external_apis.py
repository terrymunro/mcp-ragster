import asyncio
import logging
import time
from collections import defaultdict
from typing import Any
from urllib.parse import urlparse, quote

import httpx
from firecrawl import FirecrawlApp
import redis.asyncio as redis
import pickle

from .config import settings
from .exceptions import (
    APICallError,
    FirecrawlError,
    JinaAPIError,
    PerplexityAPIError,
)

logger = logging.getLogger(__name__)


class ExternalAPIClient:
    def __init__(self, http_client: httpx.AsyncClient | None = None):
        logger.info("Attempting to initialize ExternalAPIClient.")
        self.http_client = http_client
        self.firecrawl_client: FirecrawlApp | None = None

        # Initialize Redis for Jina search cache
        self._redis = redis.from_url(settings.REDIS_URI, decode_responses=False)
        self._jina_cache_ttl = settings.JINA_CACHE_TTL_HOURS * 3600
        logger.info(
            f"Jina cache using Redis at {settings.REDIS_URI} with TTL {self._jina_cache_ttl}s"
        )

        # API key/URL presence for Firecrawl is guaranteed by config.py
        try:
            if settings.FIRECRAWL_API_KEY:
                self.firecrawl_client = FirecrawlApp(api_key=settings.FIRECRAWL_API_KEY)
                logger.info("Firecrawl client initialized with API key.")
            elif settings.FIRECRAWL_API_URL:
                self.firecrawl_client = FirecrawlApp(
                    api_key=None, api_url=settings.FIRECRAWL_API_URL
                )
                logger.info(
                    f"Firecrawl client initialized with base URL: {settings.FIRECRAWL_API_URL}"
                )
            # No 'else' needed as config.py would have raised error
        except Exception as e:
            logger.critical(
                f"Failed to initialize FirecrawlApp client: {e}", exc_info=True
            )
            raise FirecrawlError(
                f"Failed to initialize FirecrawlApp client: {e}",
                underlying_error=e,
                status_code=500,
            )

    async def crawl_url_firecrawl(self, url: str) -> dict[str, Any]:
        if not self.firecrawl_client:
            raise FirecrawlError(
                "Firecrawl client not available. Initialization failed earlier.",
                status_code=500,
            )
        try:
            logger.info(f"Crawling URL with Firecrawl: {url}")
            crawl_result = self.firecrawl_client.crawl_url(
                url=url,
                params={"pageOptions": {"onlyMainContent": True, "includeHtml": False}},
            )
            if not (crawl_result and isinstance(crawl_result, dict)):
                raise FirecrawlError(
                    f"Unexpected Firecrawl result format for {url}: {type(crawl_result)}"
                )

            content_key_priority = ["markdown", "content", "data"]
            for key in content_key_priority:
                if key in crawl_result and crawl_result[key]:
                    return {
                        "content": crawl_result[key],
                        "source_url": url,
                        "type": key,
                    }
            raise FirecrawlError(
                f"Firecrawl result for {url} lacks expected content fields. Result: {str(crawl_result)[:200]}"
            )
        except Exception as e:
            if isinstance(e, FirecrawlError):
                raise
            raise FirecrawlError(
                f"Error crawling URL {url} with Firecrawl: {e}", underlying_error=e
            )

    async def check_fact_perplexity(self, fact: str) -> str:
        headers = {
            "Authorization": f"Bearer {settings.PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "model": "sonar-pro",
            "messages": [
                {"role": "system", "content": """
You are a professional fact-checker with extensive research capabilities. Your task is to evaluate claims or articles for factual accuracy. Focus on identifying false, misleading, or unsubstantiated claims.

## Evaluation Process
For each piece of content, you will:
1. Identify specific claims that can be verified
2. Research each claim thoroughly using the most reliable sources available
3. Determine if each claim is:
   - TRUE: Factually accurate and supported by credible evidence
   - FALSE: Contradicted by credible evidence
   - MISLEADING: Contains some truth but presents information in a way that could lead to incorrect conclusions
   - UNVERIFIABLE: Cannot be conclusively verified with available information
4. For claims rated as FALSE or MISLEADING, explain why and provide corrections

## Rating Criteria
- TRUE: Claim is supported by multiple credible sources with no significant contradicting evidence
- FALSE: Claim is contradicted by clear evidence from credible sources
- MISLEADING: Claim contains factual elements but is presented in a way that omits crucial context or leads to incorrect conclusions
- UNVERIFIABLE: Claim cannot be conclusively verified or refuted with available evidence

## Guidelines
- Remain politically neutral and focus solely on factual accuracy
- Do not use political leaning as a factor in your evaluation
- Prioritize official data, peer-reviewed research, and reports from credible institutions
- Cite specific, reliable sources for your determinations
- Consider the context and intended meaning of statements
- Distinguish between factual claims and opinions
- Pay attention to dates, numbers, and specific details
- Be precise and thorough in your explanations

## Response Format
Respond in JSON format with the following structure:
```json
{
    "overall_rating": "MOSTLY_TRUE|MIXED|MOSTLY_FALSE",
    "summary": "Brief summary of your overall findings",
    "claims": [
        {
            "claim": "The specific claim extracted from the text",
            "rating": "TRUE|FALSE|MISLEADING|UNVERIFIABLE",
            "explanation": "Your explanation with supporting evidence",
            "sources": ["Source 1", "Source 2"]
        },
        // Additional claims...
    ]
}
```

## Criteria for Overall Rating
- MOSTLY_TRUE: Most claims are true, with minor inaccuracies that don't affect the main message
- MIXED: The content contains a roughly equal mix of true and false/misleading claims
- MOSTLY_FALSE: Most claims are false or misleading, significantly distorting the facts

Ensure your evaluation is thorough, fair, and focused solely on factual accuracy. Do not allow personal bias to influence your assessment. Be especially rigorous with claims that sound implausible or extraordinary.
"""},
                {"role": "user", "content": f"Fact check the following text and identify any false or misleading claims:\n\n{fact}"},
            ]
        }
        try:
            if self.http_client:
                response = await self.http_client.post(
                    settings.PERPLEXITY_CHAT_API_URL, json=payload, headers=headers
                )
            else:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        settings.PERPLEXITY_CHAT_API_URL, json=payload, headers=headers
                    )

            if response.status_code != 200:
                raise APICallError(
                    "Perplexity", response.status_code, response.text[:500]
                )
            response_data = response.json()
            choices = response_data.get("choices")
            if (
                choices
                and isinstance(choices, list)
                and len(choices) > 0
                and choices[0].get("message")
            ):  # Check len(choices) > 0
                content = choices[0]["message"].get("content")
                if content and isinstance(content, str):
                    return content.strip()  # Ensure content is string
            raise PerplexityAPIError(
                f"Perplexity response format unexpected or empty content: {str(response_data)[:200]}"
            )
        except httpx.HTTPStatusError as e:
            raise APICallError(
                "Perplexity", e.response.status_code, e.response.text[:200]
            ) from e
        except httpx.RequestError as e:
            raise PerplexityAPIError(
                f"Network request to Perplexity failed: {e}", underlying_error=e
            )
        except Exception as e:
            if isinstance(e, (PerplexityAPIError, APICallError)):
                raise
            raise PerplexityAPIError(
                f"Unexpected error querying Perplexity: {e}", underlying_error=e
            )
      

    async def query_perplexity(self, topic: str) -> str:
        headers = {
            "Authorization": f"Bearer {settings.PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "model": "sonar-pro",
            "messages": [
                {"role": "system", "content": """You are an expert research assistant with access to real-time web search capabilities. Your task is to conduct comprehensive, in-depth research on any given topic.

## Research Methodology
For each topic, you will:
1. **Comprehensive Coverage**: Research all major aspects, subtopics, and dimensions of the subject
2. **Current Information**: Prioritize recent developments, trends, and up-to-date information
3. **Multiple Perspectives**: Include different viewpoints, approaches, and schools of thought
4. **Authoritative Sources**: Cite credible sources, academic papers, official reports, and expert opinions
5. **Context & Background**: Provide historical context and foundational knowledge
6. **Practical Applications**: Include real-world applications, use cases, and examples
7. **Future Outlook**: Discuss trends, predictions, and emerging developments

## Research Structure
Organize your research with clear sections:
- **Overview**: Brief introduction and key concepts
- **Current State**: Latest developments and current understanding
- **Key Players/Organizations**: Important figures, companies, institutions
- **Technical Details**: In-depth explanations of mechanisms, processes, or methodologies
- **Applications & Use Cases**: Real-world implementations and examples
- **Challenges & Limitations**: Known issues, controversies, or obstacles
- **Future Directions**: Emerging trends and potential developments
- **Sources**: Key references and citations

## Research Quality Standards
- Use multiple credible sources for each major claim
- Include quantitative data, statistics, and metrics where available
- Distinguish between established facts and emerging theories
- Note any conflicting information or debates in the field
- Provide specific examples and case studies
- Include relevant technical specifications or parameters

## Output Format
Present findings in a well-structured, comprehensive report that balances depth with readability. Use clear headings, bullet points, and logical flow. Aim for thoroughness while maintaining clarity."""},
                {"role": "user", "content": f"Conduct comprehensive research on: {topic}"},
            ],
            "max_tokens": 4000,
            "return_related_questions": False,
            "search_recency_filter": "year",
            "search_domain_filter": ["-reddit.com", "-quora.com"],
        }
        # TBD: Run the related queries as well but limited by depth
        try:
            if self.http_client:
                response = await self.http_client.post(
                    settings.PERPLEXITY_CHAT_API_URL, json=payload, headers=headers
                )
            else:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        settings.PERPLEXITY_CHAT_API_URL, json=payload, headers=headers
                    )

            if response.status_code != 200:
                raise APICallError(
                    "Perplexity", response.status_code, response.text[:500]
                )
            response_data = response.json()
            choices = response_data.get("choices")
            if (
                choices
                and isinstance(choices, list)
                and len(choices) > 0
                and choices[0].get("message")
            ):  # Check len(choices) > 0
                content = choices[0]["message"].get("content")
                if content and isinstance(content, str):
                    return content.strip()  # Ensure content is string
            raise PerplexityAPIError(
                f"Perplexity response format unexpected or empty content: {str(response_data)[:200]}"
            )
        except httpx.HTTPStatusError as e:
            raise APICallError(
                "Perplexity", e.response.status_code, e.response.text[:200]
            ) from e
        except httpx.RequestError as e:
            raise PerplexityAPIError(
                f"Network request to Perplexity failed: {e}", underlying_error=e
            )
        except Exception as e:
            if isinstance(e, (PerplexityAPIError, APICallError)):
                raise
            raise PerplexityAPIError(
                f"Unexpected error querying Perplexity: {e}", underlying_error=e
            )

    def _get_jina_cache_key(self, topic: str) -> str:
        normalized_topic = topic.lower().strip()
        return f"jina:{hash(normalized_topic)}"

    async def _jina_cache_get(self, cache_key: str) -> list[dict[str, Any]] | None:
        val = await self._redis.get(cache_key)
        if val is not None:
            try:
                return pickle.loads(val)
            except Exception as e:
                logger.warning(f"Failed to unpickle Jina cache from Redis: {e}")
                return None
        return None

    async def _jina_cache_set(
        self, cache_key: str, results: list[dict[str, Any]]
    ) -> None:
        try:
            await self._redis.set(
                cache_key, pickle.dumps(results), ex=self._jina_cache_ttl
            )
        except Exception as e:
            logger.warning(f"Failed to set Jina cache in Redis: {e}")

    async def search_jina(
        self, topic: str
    ) -> list[dict[str, Any]]:
        cache_key = self._get_jina_cache_key(topic)
        cached_results = await self._jina_cache_get(cache_key)
        if cached_results:
            logger.debug(f"Jina cache hit for topic: {topic}")
            return cached_results

        headers = {
            "Authorization": f"Bearer {settings.JINA_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Respond-With": "no-content",
        }
        search_url = settings.JINA_SEARCH_API_URL
        params = {"q": topic, "limit": num_results}
        logger.debug(
            f"Sending Jina search request to {search_url} with params {params}"
        )
        try:
            if self.http_client:
                response = await self.http_client.get(
                    search_url, headers=headers, params=params
                )
            else:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(
                        search_url, headers=headers, params=params
                    )

            logger.debug(f"Jina response status: {response.status_code}, content-type: {response.headers.get('content-type', 'unknown')}")
            if response.status_code != 200:
                raise APICallError("Jina", response.status_code, response.text[:500])
            
            # Better JSON parsing with error handling
            try:
                response_data = response.json()
            except Exception as json_error:
                # Log the raw response for debugging
                raw_text = response.text
                logger.error(
                    f"Jina JSON parsing failed for topic '{topic}'. "
                    f"Error: {json_error}. "
                    f"Raw response (first 500 chars): {raw_text[:500]}"
                )
                raise JinaAPIError(
                    f"Failed to parse Jina response as JSON: {json_error}. "
                    f"Raw response: {raw_text[:200]}"
                )
            
            # Handle the new API response format
            if response_data.get("code") == 200 and "data" in response_data:
                data_source = response_data["data"]
                if isinstance(data_source, list):
                    results = [
                        {
                            "url": i["url"],
                            "title": i["title"],
                            "snippet": i.get("description", ""),
                        }
                        for i in data_source
                        if i.get("url") and i.get("title")
                    ]

                    # Cache the results
                    await self._jina_cache_set(cache_key, results)
                    logger.debug(f"Jina results cached for topic: {topic}")
                    return results
            
            raise JinaAPIError(
                f"Jina Search response format unexpected: {str(response_data)[:200]}"
            )
        except httpx.HTTPStatusError as e:
            raise APICallError(
                "Jina", e.response.status_code, e.response.text[:200]
            ) from e
        except httpx.RequestError as e:
            raise JinaAPIError(
                f"Network request to Jina failed: {e}", underlying_error=e
            )
        except Exception as e:
            if isinstance(e, (JinaAPIError, APICallError)):
                raise
            raise JinaAPIError(
                f"Unexpected error querying Jina: {e}", underlying_error=e
            )

    async def close(self):
        """Placeholder for any explicit cleanup if needed."""
        # Firecrawl client doesn't have an explicit close method in the SDK from what's seen.
        # httpx clients for Jina/Perplexity are created per-call.
        logger.info(
            "ExternalAPIClient close called (currently no specific resources to release)."
        )


class FirecrawlBatcher:
    """Manages batched Firecrawl requests with domain grouping and deduplication."""

    def __init__(self, client: ExternalAPIClient, max_concurrent: int = 3):
        self.client = client
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # URL deduplication cache with timestamps
        self._url_cache: dict[str, tuple[dict[str, Any], float]] = {}
        self._cache_ttl = 3600  # 1 hour TTL

        # Retry queue with backoff - (url, attempts, next_retry_time)
        self._retry_queue: list[tuple[str, int, float]] = []
        self._max_retries = 3
        self._base_backoff = 2.0

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
        """Crawl URL with retry logic."""
        attempts = 0
        while attempts < self._max_retries:
            try:
                async with self.semaphore:
                    result = await self.client.crawl_url_firecrawl(url)
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
                        f"Failed to crawl {url} after {attempts} attempts: {e}"
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
        """Crawl multiple URLs with domain-based grouping and caching."""
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
                await asyncio.sleep(0.5)
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
