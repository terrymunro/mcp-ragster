import asyncio
from collections import defaultdict

import pytest

from ragster import external_apis
from ragster.external_apis import ExternalAPIClient, FirecrawlBatcher, FirecrawlError


class MockResponse:
    def __init__(
        self, status_code: int = 200, json_data: dict | None = None, text: str = ""
    ):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text

    def json(self):
        return self._json


class MockAsyncClient:
    def __init__(self, get_responses=None, post_responses=None):
        self.get_responses = get_responses or []
        self.post_responses = post_responses or []
        self.get_calls = []
        self.post_calls = []
        self.is_closed = False

    async def get(self, url, headers=None, params=None):
        self.get_calls.append((url, headers, params))
        return self.get_responses.pop(0)

    async def post(self, url, json=None, headers=None):
        self.post_calls.append((url, json, headers))
        return self.post_responses.pop(0)

    async def aclose(self):
        self.is_closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class DummyFirecrawlApp:
    def __init__(self, api_key=None, api_url=None):
        self.calls = []

    def crawl_url(self, url: str, params=None):
        self.calls.append((url, params))
        return {"markdown": f"content for {url}"}


@pytest.fixture(autouse=True)
def patch_firecrawl(monkeypatch):
    monkeypatch.setattr(external_apis, "FirecrawlApp", DummyFirecrawlApp)


@pytest.fixture
def http_client(monkeypatch):
    client = MockAsyncClient()
    monkeypatch.setattr(external_apis.httpx, "AsyncClient", lambda *a, **kw: client)
    return client


@pytest.mark.asyncio
async def test_search_jina_caching(http_client):
    http_client.get_responses.append(
        MockResponse(
            json_data={"data": [{"url": "http://x", "title": "t", "snippet": "s"}]}
        )
    )
    api_client = ExternalAPIClient(http_client=http_client)
    results1 = await api_client.search_jina("topic", num_results=1)
    assert len(http_client.get_calls) == 1
    assert results1 == [{"url": "http://x", "title": "t", "snippet": "s"}]

    results2 = await api_client.search_jina("topic", num_results=1)
    assert len(http_client.get_calls) == 1
    assert results2 == results1


@pytest.mark.asyncio
async def test_query_perplexity_success(http_client):
    http_client.post_responses.append(
        MockResponse(json_data={"choices": [{"message": {"content": "summary"}}]})
    )
    api_client = ExternalAPIClient(http_client=http_client)
    result = await api_client.query_perplexity("test")
    assert result == "summary"
    assert len(http_client.post_calls) == 1


@pytest.mark.asyncio
async def test_query_perplexity_invalid_response(http_client):
    http_client.post_responses.append(MockResponse(json_data={"choices": []}))
    api_client = ExternalAPIClient(http_client=http_client)
    with pytest.raises(external_apis.PerplexityAPIError):
        await api_client.query_perplexity("bad")


class FakeExternalClient(ExternalAPIClient):
    def __init__(self, fail_first: str | None = None):
        # Skip parent initialization to avoid network clients
        self.fail_first = fail_first
        self.calls = []
        self.attempts = defaultdict(int)
        self.active = 0
        self.max_active = 0

    async def crawl_url_firecrawl(self, url: str):
        self.calls.append(url)
        self.attempts[url] += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if url == self.fail_first and self.attempts[url] == 1:
            await asyncio.sleep(0.05)
            self.active -= 1
            raise FirecrawlError("boom")
        await asyncio.sleep(0.05)
        self.active -= 1
        return {"content": url, "source_url": url, "type": "markdown"}


@pytest.mark.asyncio
async def test_firecrawl_batcher_metrics_and_concurrency(monkeypatch):
    async def fake_sleep(_):
        return None

    monkeypatch.setattr(external_apis.asyncio, "sleep", fake_sleep)
    client = FakeExternalClient(fail_first="http://a.com/1")
    batcher = FirecrawlBatcher(client, max_concurrent=2)

    urls = ["http://a.com/1", "http://a.com/2", "http://b.com/1"]
    await batcher.crawl_urls(urls)
    metrics = batcher.get_metrics()
    assert metrics["successful_crawls"] == 3
    assert metrics["retries"] == 1
    assert client.max_active <= 2

    await batcher.crawl_urls(urls)
    metrics = batcher.get_metrics()
    assert metrics["cache_hits"] == 3
    assert metrics["cache_size"] == 3
