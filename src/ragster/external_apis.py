import logging
from typing import Any

import httpx

from ragster.client_perplexity import PerplexityAPIClient
from ragster.client_jina import JinaAPIClient
from ragster.client_firecrawl import FirecrawlAPIClient


logger = logging.getLogger(__name__)


class ExternalAPIClient:
    """Facade class that coordinates all external API clients."""
    
    def __init__(self, http_client: httpx.AsyncClient | None = None):
        logger.info("Initializing ExternalAPIClient facade.")
        self.jina = JinaAPIClient(http_client)
        self.perplexity = PerplexityAPIClient(http_client)
        self.firecrawl = FirecrawlAPIClient()

    # Backward compatibility methods
    async def search_jina(self, topic: str) -> list[dict[str, Any]]:
        """Search using Jina API."""
        return await self.jina.search(topic)

    async def query_perplexity(self, topic: str) -> str:
        """Query using Perplexity API."""
        return await self.perplexity.query(topic)

    async def check_fact_perplexity(self, fact: str) -> str:
        """Fact-check using Perplexity API."""
        return await self.perplexity.check_fact(fact)

    async def crawl_url_firecrawl(self, url: str) -> dict[str, Any]:
        """Crawl URL using Firecrawl API."""
        return await self.firecrawl.crawl_url(url)

    async def close(self):
        """Close all API clients."""
        await self.jina.close()
        await self.perplexity.close()
        await self.firecrawl.close()

