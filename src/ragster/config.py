import os
import logging

from dotenv import load_dotenv

from .exceptions import ConfigurationError


load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

KNOWN_VOYAGE_MODEL_DIMS = {
    "voyage-3-large": 2048,
    "voyage-3.5": 2048,
    "voyage-3.5-lite": 2048,
    "voyage-2": 1024,
    "voyage-code-2": 1536,
    "voyage-code-3": 2048,
    "voyage-law-2": 1024,
}


class Settings:
    # Voyage AI Configuration
    VOYAGEAI_API_KEY: str = os.getenv("VOYAGEAI_API_KEY", "")
    VOYAGEAI_MODEL_NAME: str = os.getenv("VOYAGEAI_MODEL_NAME", "voyage-3-large")
    VOYAGEAI_EMBEDDING_API_URL: str = os.getenv(
        "VOYAGEAI_EMBEDDING_API_URL", "https://api.voyageai.com/v1/embeddings"
    )
    VOYAGEAI_INPUT_TYPE_DOCUMENT: str = os.getenv(
        "VOYAGEAI_INPUT_TYPE_DOCUMENT", "document"
    )
    VOYAGEAI_INPUT_TYPE_QUERY: str = os.getenv("VOYAGEAI_INPUT_TYPE_QUERY", "query")

    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", 1024))

    # Milvus Configuration
    MILVUS_ALIAS: str = os.getenv("MILVUS_ALIAS", "default")
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
    MILVUS_USER: str | None = os.getenv("MILVUS_USER")
    MILVUS_PASSWORD: str | None = os.getenv("MILVUS_PASSWORD")
    MILVUS_USE_SSL: bool = os.getenv("MILVUS_USE_SSL", "False").lower() == "true"
    MILVUS_SERVER_PEM_PATH: str | None = os.getenv("MILVUS_SERVER_PEM_PATH")
    MILVUS_SERVER_NAME: str | None = os.getenv("MILVUS_SERVER_NAME")
    MILVUS_CA_CERT_PATH: str | None = os.getenv("MILVUS_CA_CERT_PATH")
    MILVUS_CLIENT_KEY_PATH: str | None = os.getenv("MILVUS_CLIENT_KEY_PATH")
    MILVUS_CLIENT_PEM_PATH: str | None = os.getenv("MILVUS_CLIENT_PEM_PATH")

    # Firecrawl Configuration
    FIRECRAWL_API_URL: str | None = os.getenv("FIRECRAWL_API_URL")
    FIRECRAWL_API_KEY: str | None = os.getenv("FIRECRAWL_API_KEY")

    # Perplexity AI Configuration
    PERPLEXITY_API_KEY: str = os.getenv("PERPLEXITY_API_KEY", "")

    # Jina AI Configuration
    JINA_API_KEY: str = os.getenv("JINA_API_KEY", "")

    # Milvus Collection Configuration
    MILVUS_COLLECTION_NAME: str = os.getenv(
        "MILVUS_COLLECTION_NAME", "topic_context_store"
    )
    MILVUS_INDEX_FIELD_NAME: str = os.getenv("MILVUS_INDEX_FIELD_NAME", "embedding")
    MILVUS_VECTOR_DIMENSION: int
    MILVUS_ID_FIELD_NAME: str = os.getenv("MILVUS_ID_FIELD_NAME", "id")
    MILVUS_TEXT_FIELD_NAME: str = os.getenv("MILVUS_TEXT_FIELD_NAME", "text_content")
    MILVUS_TOPIC_FIELD_NAME: str = os.getenv("MILVUS_TOPIC_FIELD_NAME", "topic")
    MILVUS_SOURCE_TYPE_FIELD_NAME: str = os.getenv(
        "MILVUS_SOURCE_TYPE_FIELD_NAME", "source_type"
    )
    MILVUS_SOURCE_IDENTIFIER_FIELD_NAME: str = os.getenv(
        "MILVUS_SOURCE_IDENTIFIER_FIELD_NAME", "source_identifier"
    )
    MILVUS_METRIC_TYPE: str = os.getenv("MILVUS_METRIC_TYPE", "IP")
    MILVUS_INDEX_TYPE: str = os.getenv("MILVUS_INDEX_TYPE", "HNSW")

    # Milvus Index Parameters (examples for HNSW and IVF_FLAT)
    MILVUS_HNSW_M: int = int(os.getenv("MILVUS_HNSW_M", 32))
    MILVUS_HNSW_EF_CONSTRUCTION: int = int(
        os.getenv("MILVUS_HNSW_EF_CONSTRUCTION", 400)
    )
    MILVUS_IVF_NLIST: int = int(os.getenv("MILVUS_IVF_NLIST", 1024))

    # Milvus Search Parameters
    MILVUS_SEARCH_EF: int = int(os.getenv("MILVUS_SEARCH_EF", 64))
    MILVUS_SEARCH_EF_EXPLORATION: int = int(
        os.getenv("MILVUS_SEARCH_EF_EXPLORATION", 128)
    )
    MILVUS_SEARCH_NPROBE: int = int(os.getenv("MILVUS_SEARCH_NPROBE", 10))

    # Search parameters
    MILVUS_SEARCH_LIMIT: int = int(os.getenv("MILVUS_SEARCH_LIMIT", 5))
    JINA_SEARCH_LIMIT: int = int(os.getenv("JINA_SEARCH_LIMIT", 5))

    # Concurrency settings
    MAX_CONCURRENT_FIRECRAWL: int = int(os.getenv("MAX_CONCURRENT_FIRECRAWL", 3))

    # Caching settings
    JINA_CACHE_TTL_HOURS: int = int(os.getenv("JINA_CACHE_TTL_HOURS", 3))
    EMBEDDING_CACHE_SIZE: int = int(os.getenv("EMBEDDING_CACHE_SIZE", 1000))
    ENABLE_INDEX_WARMUP: bool = (
        os.getenv("ENABLE_INDEX_WARMUP", "True").lower() == "true"
    )

    # API Endpoints
    JINA_SEARCH_API_URL: str = "https://s.jina.ai/search"
    PERPLEXITY_CHAT_API_URL: str = "https://api.perplexity.ai/chat/completions"

    def __init__(self):
        if not self.VOYAGEAI_API_KEY:
            raise ConfigurationError("VOYAGEAI_API_KEY is not set in the environment.")
        if not self.PERPLEXITY_API_KEY:
            raise ConfigurationError(
                "PERPLEXITY_API_KEY is not set in the environment."
            )
        if not self.JINA_API_KEY:
            raise ConfigurationError("JINA_API_KEY is not set in the environment.")
        if not self.FIRECRAWL_API_URL and not self.FIRECRAWL_API_KEY:
            raise ConfigurationError(
                "Either FIRECRAWL_API_URL or FIRECRAWL_API_KEY must be set."
            )
        if not self.EMBEDDING_DIMENSION:
            dim = KNOWN_VOYAGE_MODEL_DIMS.get(self.VOYAGEAI_MODEL_NAME)
            if dim is not None:
                self.EMBEDDING_DIMENSION = dim
        self.MILVUS_VECTOR_DIMENSION = self.EMBEDDING_DIMENSION

        if self.MILVUS_METRIC_TYPE == "L2" and "voyage" in self.VOYAGEAI_MODEL_NAME:
            logger.warning(
                "Configuration Note: Voyage AI embeddings typically perform best with Cosine similarity (IP metric in Milvus), but L2 is configured."
            )
        if self.MILVUS_INDEX_TYPE == "IVF_FLAT" and self.MILVUS_METRIC_TYPE == "IP":
            logger.warning(
                "Configuration Note: For IP (Cosine) metric, HNSW is often a better Milvus index choice than IVF_FLAT."
            )


settings = Settings()
