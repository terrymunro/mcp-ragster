import hashlib
import logging
from typing import Literal, TypeGuard

import httpx

from .config import settings
from .exceptions import APICallError, EmbeddingServiceError


logger = logging.getLogger(__name__)
VoyageInputType = Literal["document", "query"]


def is_voyage_input_type(input_type: str) -> TypeGuard[VoyageInputType]:
    return input_type in ("document", "query")


class EmbeddingClient:
    _voyage_client: httpx.AsyncClient | None = None  # Initialize as None

    def __init__(self):
        logger.info("Attempting to initialize EmbeddingClient for Voyage AI.")
        # VOYAGEAI_API_KEY presence is guaranteed by config.py
        try:
            # Create the client during initialization of this instance
            # This will be called once by the lifespan manager
            self._voyage_client = httpx.AsyncClient(timeout=30.0)
            logger.info("Voyage AI embedding client httpx.AsyncClient created.")

            # Initialize LRU cache for embeddings
            self._embedding_cache = {}
            self._cache_size = settings.EMBEDDING_CACHE_SIZE
            logger.info(f"Embedding cache initialized with size {self._cache_size}")
        except Exception as e:
            logger.critical(
                f"Failed to initialize httpx.AsyncClient for Voyage AI: {e}",
                exc_info=True,
            )
            raise EmbeddingServiceError(
                f"Failed to initialize Voyage AI client infrastructure: {e}",
                underlying_error=e,
            )

    def get_embedding_dimension(self) -> int:
        return settings.EMBEDDING_DIMENSION

    def _get_cache_key(self, text: str, input_type: VoyageInputType) -> str:
        """Generate cache key for embedding."""
        content = f"{text}:{input_type}:{settings.VOYAGEAI_MODEL_NAME}"
        return hashlib.md5(content.encode()).hexdigest()

    def _cache_get(self, cache_key: str) -> list[float] | None:
        """Get embedding from cache."""
        return self._embedding_cache.get(cache_key)

    def _cache_set(self, cache_key: str, embedding: list[float]) -> None:
        """Set embedding in cache with LRU eviction."""
        if len(self._embedding_cache) >= self._cache_size:
            # Remove oldest item (FIFO approximation of LRU)
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
        self._embedding_cache[cache_key] = embedding

    async def embed_texts(
        self, texts: str | list[str], input_type: VoyageInputType
    ) -> list[float] | list[list[float]]:
        if not self._voyage_client or self._voyage_client.is_closed:
            logger.error(
                "Voyage AI client is not available (not initialized or closed)."
            )
            raise EmbeddingServiceError(
                "Voyage AI client is not available (not initialized or closed)."
            )

        input_texts_list = texts if isinstance(texts, list) else [texts]
        if not input_texts_list:
            logger.warning("embed_texts called with empty input list.")
            return []

        # Check cache for single text queries
        if isinstance(texts, str):
            cache_key = self._get_cache_key(texts, input_type)
            cached_embedding = self._cache_get(cache_key)
            if cached_embedding:
                logger.debug(f"Cache hit for embedding: {texts[:50]}...")
                return cached_embedding

        # Check cache for multiple texts
        cached_results = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(input_texts_list):
            cache_key = self._get_cache_key(text, input_type)
            cached_embedding = self._cache_get(cache_key)
            if cached_embedding:
                cached_results.append((i, cached_embedding))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # If all texts are cached
        if not uncached_texts:
            logger.debug(f"All {len(input_texts_list)} embeddings found in cache")
            embeddings: list[list[float]] = [[] for _ in range(len(input_texts_list))]
            for i, embedding in cached_results:
                embeddings[i] = embedding
            return embeddings[0] if isinstance(texts, str) else embeddings

        # Make API call for uncached texts
        headers = {
            "Authorization": f"Bearer {settings.VOYAGEAI_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "input": uncached_texts,
            "model": settings.VOYAGEAI_MODEL_NAME,
            "input_type": input_type,
        }

        cache_hits = len(input_texts_list) - len(uncached_texts)
        logger.info(
            f"Embedding {len(uncached_texts)} text(s) via Voyage AI ({cache_hits} cache hits): model={settings.VOYAGEAI_MODEL_NAME}, type={input_type}."
        )

        try:
            response = await self._voyage_client.post(
                settings.VOYAGEAI_EMBEDDING_API_URL, json=payload, headers=headers
            )

            if response.status_code != 200:
                error_detail = (
                    response.text[:500] if response.text else "No additional detail."
                )
                logger.error(
                    f"Voyage AI API error: HTTP {response.status_code} - {error_detail}"
                )
                raise APICallError(
                    service_name="Voyage AI",
                    http_status=response.status_code,
                    detail=error_detail,
                )

            response_data = response.json()
            if "data" not in response_data or not isinstance(
                response_data["data"], list
            ):
                raise EmbeddingServiceError(
                    f"Unexpected response format from Voyage AI: 'data' field missing/invalid. Response: {str(response_data)[:200]}"
                )

            new_embeddings = [item["embedding"] for item in response_data["data"]]
            if len(new_embeddings) != len(uncached_texts):
                raise EmbeddingServiceError(
                    f"Embeddings count mismatch. Expected {len(uncached_texts)}, got {len(new_embeddings)}."
                )

            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                cache_key = self._get_cache_key(text, input_type)
                self._cache_set(cache_key, embedding)

            # Combine cached and new results
            if isinstance(texts, str):
                return new_embeddings[0]

            all_embeddings: list[list[float]] = [
                [] for _ in range(len(input_texts_list))
            ]
            for i, embedding in cached_results:
                all_embeddings[i] = embedding
            for i, embedding in zip(uncached_indices, new_embeddings):
                all_embeddings[i] = embedding

            return all_embeddings

        except httpx.HTTPStatusError as e:
            raise APICallError(
                service_name="Voyage AI",
                http_status=e.response.status_code,
                detail=e.response.text[:200],
            ) from e
        except httpx.RequestError as e:
            raise EmbeddingServiceError(
                f"Network request to Voyage AI failed: {e}", underlying_error=e
            )
        except Exception as e:
            if isinstance(e, (EmbeddingServiceError, APICallError)):
                raise
            raise EmbeddingServiceError(
                f"Unexpected error during embedding: {e}", underlying_error=e
            )

    async def close_voyage_client(self):
        if self._voyage_client and not self._voyage_client.is_closed:
            try:
                await self._voyage_client.aclose()
                logger.info("Voyage AI httpx client closed.")
            except Exception as e:
                logger.error(f"Error closing Voyage AI client: {e}")
        self._voyage_client = None  # Ensure it's marked as unusable
