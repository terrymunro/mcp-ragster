from ragster.config import settings
from ragster.exceptions import APICallError, JinaAPIError
from ragster.external_apis import logger


import httpx
import redis.asyncio as redis


import pickle
from typing import Any


class JinaAPIClient:
    """Client for Jina AI search API with Redis caching."""

    def __init__(self, http_client: httpx.AsyncClient | None = None):
        logger.info("Initializing JinaAPIClient.")
        self.http_client = http_client

        # Initialize Redis for caching
        self._redis = redis.from_url(settings.REDIS_URI, decode_responses=False)
        self._cache_ttl = settings.JINA_CACHE_TTL_HOURS * 3600
        logger.info(
            f"Jina cache using Redis at {settings.REDIS_URI} with TTL {self._cache_ttl}s"
        )

    def _get_cache_key(self, topic: str) -> str:
        normalized_topic = topic.lower().strip()
        return f"jina:{hash(normalized_topic)}"

    async def _cache_get(self, cache_key: str) -> list[dict[str, Any]] | None:
        val = await self._redis.get(cache_key)
        if val is not None:
            try:
                return pickle.loads(val)
            except Exception as e:
                logger.warning(f"Failed to unpickle Jina cache from Redis: {e}")
                return None
        return None

    async def _cache_set(
        self, cache_key: str, results: list[dict[str, Any]]
    ) -> None:
        try:
            await self._redis.set(
                cache_key, pickle.dumps(results), ex=self._cache_ttl
            )
        except Exception as e:
            logger.warning(f"Failed to set Jina cache in Redis: {e}")

    async def search(self, topic: str) -> list[dict[str, Any]]:
        """Search for content using Jina AI API with caching."""
        cache_key = self._get_cache_key(topic)
        cached_results = await self._cache_get(cache_key)
        if cached_results:
            logger.debug(f"Jina cache hit for topic: {topic}")
            return cached_results

        headers = {
            "Authorization": f"Bearer {settings.JINA_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Respond-With": "no-content",
        }

        payload = {
            "q": topic
        }

        try:
            logger.debug(f"Sending Jina search request to {settings.JINA_SEARCH_API_URL} with payload: {payload}")
            if self.http_client:
                response = await self.http_client.post(settings.JINA_SEARCH_API_URL, json=payload, headers=headers)
            else:
                async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_JINA) as client:
                    response = await client.post(settings.JINA_SEARCH_API_URL, json=payload, headers=headers)

            logger.debug(f"Jina response status: {response.status_code}, content-type: {response.headers.get('content-type', 'unknown')}")
            if response.status_code != 200:
                raise APICallError("Jina", response.status_code, response.text[:500])

            try:
                response_data = response.json()
            except Exception as json_error:
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

                    await self._cache_set(cache_key, results)
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
        """Close Redis connection."""
        if hasattr(self, '_redis'):
            await self._redis.close()