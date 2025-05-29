import hashlib
from typing import Any

import httpx
import pytest

from ragster.embedding_client import EmbeddingClient
from ragster.exceptions import APICallError, EmbeddingServiceError
from ragster.config import settings


@pytest.mark.asyncio
async def test_cache_key_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    client = EmbeddingClient()
    embedding = [0.1, 0.2]

    async def fake_post(
        self: httpx.AsyncClient, url: str, json: Any = None, headers: Any | None = None
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200, json={"data": [{"embedding": embedding}]}
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result1 = await client.embed_texts("hello", "document")
    assert result1 == embedding

    called = False

    async def fake_post_check(
        self: httpx.AsyncClient, url: str, json: Any = None, headers: Any | None = None
    ) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(status_code=200, json={"data": [{"embedding": [1.0]}]})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post_check)

    result2 = await client.embed_texts("hello", "document")
    assert result2 == embedding
    assert not called

    expected_key = hashlib.md5(
        f"hello:document:{settings.VOYAGEAI_MODEL_NAME}".encode()
    ).hexdigest()
    assert client._get_cache_key("hello", "document") == expected_key


@pytest.mark.asyncio
async def test_non_200_response(monkeypatch: pytest.MonkeyPatch) -> None:
    client = EmbeddingClient()

    async def fake_post(
        self: httpx.AsyncClient, url: str, json: Any = None, headers: Any | None = None
    ) -> httpx.Response:
        return httpx.Response(status_code=500, text="error")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    with pytest.raises(APICallError):
        await client.embed_texts("hi", "document")


@pytest.mark.asyncio
async def test_error_propagation(monkeypatch: pytest.MonkeyPatch) -> None:
    client = EmbeddingClient()

    async def raise_request_error(*args: Any, **kwargs: Any) -> httpx.Response:
        raise httpx.RequestError("boom", request=httpx.Request("POST", "http://x"))

    monkeypatch.setattr(httpx.AsyncClient, "post", raise_request_error)
    with pytest.raises(EmbeddingServiceError):
        await client.embed_texts("hi", "document")

    async def raise_api_error(*args: Any, **kwargs: Any) -> httpx.Response:
        raise APICallError("Voyage AI", 429, "rate limit")

    monkeypatch.setattr(httpx.AsyncClient, "post", raise_api_error)
    with pytest.raises(APICallError):
        await client.embed_texts("hi", "document")


@pytest.mark.asyncio
async def test_close_voyage_client(monkeypatch: pytest.MonkeyPatch) -> None:
    client = EmbeddingClient()
    closed = False
    assert client._voyage_client is not None
    original_aclose = client._voyage_client.aclose

    async def aclose_wrapper() -> None:
        nonlocal closed
        closed = True
        await original_aclose()

    monkeypatch.setattr(client._voyage_client, "aclose", aclose_wrapper)

    await client.close_voyage_client()
    assert closed
    assert client._voyage_client is None
