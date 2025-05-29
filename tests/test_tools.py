import httpx
import pytest
from typing import Any, Dict
from ragster.tools import TopicProcessor

from ragster.server import AppContext
from ragster.tools import (
    LoadTopicToolArgs,
    QueryTopicToolArgs,
    load_topic_context,
    query_topic_context,
)

from .helpers import StubEmbeddingClient, StubExternalAPIClient, StubMilvusOperator


@pytest.mark.asyncio
async def test_load_topic_context(monkeypatch: pytest.MonkeyPatch):
    async def immediate_tasks(self: TopicProcessor, topic: str, results: list[Dict[str, Any]]) -> None:
        await self.perplexity_sub_task(topic)
        for res in results:
            if url := res.get("url"):
                await self.firecrawl_sub_task(topic, url)

    monkeypatch.setattr(TopicProcessor, "process_concurrent_tasks", immediate_tasks)

    embed = StubEmbeddingClient()
    api_client = StubExternalAPIClient(jina_result_count=3)
    milvus = StubMilvusOperator()
    async with httpx.AsyncClient() as http_client:
        ctx = AppContext(
            embedding_client=embed,
            milvus_operator=milvus,
            external_api_client=api_client,
            http_client=http_client,
        )
        args = LoadTopicToolArgs(topic="test topic")
        response = await load_topic_context(args, ctx)

    # verify API calls
    assert len(api_client.search_jina_calls) == 1
    assert len(api_client.query_perplexity_calls) == 1
    assert len(api_client.crawl_url_firecrawl_calls) == 3

    # response fields
    assert response.topic == "test topic"
    assert response.urls_processed == 3
    assert response.perplexity_queried is True
    assert response.jina_results_found == 3
    assert response.errors == []

    # embedding and insertion happened
    assert embed.embed_texts_calls  # not empty
    assert milvus.insert_calls  # not empty


@pytest.mark.asyncio
async def test_query_topic_context():
    embed = StubEmbeddingClient()
    api_client = StubExternalAPIClient()
    milvus = StubMilvusOperator()
    async with httpx.AsyncClient() as http_client:
        ctx = AppContext(
            embedding_client=embed,
            milvus_operator=milvus,
            external_api_client=api_client,
            http_client=http_client,
        )
        args = QueryTopicToolArgs(query="what is up?", top_k=2)
        response = await query_topic_context(args, ctx)

    # embedding called with query
    assert embed.embed_texts_calls[0][0] == "what is up?"

    # milvus queried with provided top_k
    assert milvus.query_calls[0][1] == 2

    # response correctness
    assert response.query == "what is up?"
    assert len(response.results) == 2
    assert response.message is not None
    assert response.message.startswith("Found")
