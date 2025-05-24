import httpx
import logging
from typing import List, Union, Optional, Literal # Keep Optional for now, can refine later

if __package__:
    from .config import settings
    from .exceptions import EmbeddingServiceError, APICallError
else: 
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    from ragster.config import settings
    from ragster.exceptions import EmbeddingServiceError, APICallError


logger = logging.getLogger(__name__)
VoyageInputType = Literal["document", "query"]


class EmbeddingClient:
    _voyage_client: Optional[httpx.AsyncClient] = None # Initialize as None

    def __init__(self):
        logger.info("Attempting to initialize EmbeddingClient for Voyage AI.")
        # VOYAGEAI_API_KEY presence is guaranteed by config.py
        try:
            # Create the client during initialization of this instance
            # This will be called once by the lifespan manager
            self._voyage_client = httpx.AsyncClient(timeout=30.0)
            logger.info("Voyage AI embedding client httpx.AsyncClient created.")
        except Exception as e:
            logger.critical(f"Failed to initialize httpx.AsyncClient for Voyage AI: {e}", exc_info=True)
            raise EmbeddingServiceError(f"Failed to initialize Voyage AI client infrastructure: {e}", underlying_error=e)

    def get_embedding_dimension(self) -> int:
        return settings.EMBEDDING_DIMENSION

    async def embed_texts(
        self, texts: Union[str, List[str]], input_type: VoyageInputType
    ) -> Union[List[float], List[List[float]]]:
        if not self._voyage_client or self._voyage_client.is_closed:
            # This might happen if close_voyage_client was called and then this method is used,
            # or if initialization failed to set it.
            logger.error("Voyage AI client is not available (not initialized or closed).")
            raise EmbeddingServiceError("Voyage AI client is not available (not initialized or closed).")
        
        input_texts_list = texts if isinstance(texts, list) else [texts]
        if not input_texts_list: 
            logger.warning("embed_texts called with empty input list.")
            return [] 

        headers = {"Authorization": f"Bearer {settings.VOYAGEAI_API_KEY}", "Content-Type": "application/json", "Accept": "application/json"}
        payload = {"input": input_texts_list, "model": settings.VOYAGEAI_MODEL_NAME, "input_type": input_type}
        
        logger.info(f"Embedding {len(input_texts_list)} text(s) via Voyage AI: model={settings.VOYAGEAI_MODEL_NAME}, type={input_type}.")
        try:
            response = await self._voyage_client.post(settings.VOYAGEAI_EMBEDDING_API_URL, json=payload, headers=headers)
            
            if response.status_code != 200:
                error_detail = response.text[:500] if response.text else "No additional detail."
                logger.error(f"Voyage AI API error: HTTP {response.status_code} - {error_detail}")
                raise APICallError(service_name="Voyage AI", http_status=response.status_code, detail=error_detail)

            response_data = response.json()
            if "data" not in response_data or not isinstance(response_data["data"], list):
                raise EmbeddingServiceError(f"Unexpected response format from Voyage AI: 'data' field missing/invalid. Response: {str(response_data)[:200]}")

            embeddings = [item['embedding'] for item in response_data["data"]]
            if (not embeddings and input_texts_list) or (len(embeddings) != len(input_texts_list)):
                raise EmbeddingServiceError(f"Embeddings count mismatch or empty result. Expected {len(input_texts_list)}, got {len(embeddings)}.")
            
            return embeddings[0] if isinstance(texts, str) else embeddings
            
        except httpx.HTTPStatusError as e:
             raise APICallError(service_name="Voyage AI", http_status=e.response.status_code, detail=e.response.text[:200]) from e
        except httpx.RequestError as e: 
            raise EmbeddingServiceError(f"Network request to Voyage AI failed: {e}", underlying_error=e)
        except Exception as e: 
            if isinstance(e, (EmbeddingServiceError, APICallError)): raise # Re-raise specific known errors
            raise EmbeddingServiceError(f"Unexpected error during embedding: {e}", underlying_error=e)

    async def close_voyage_client(self):
        if self._voyage_client and not self._voyage_client.is_closed:
            try:
                await self._voyage_client.aclose()
                logger.info("Voyage AI httpx client closed.")
            except Exception as e: 
                logger.error(f"Error closing Voyage AI client: {e}")
        self._voyage_client = None # Ensure it's marked as unusable
