import io
import sys
import types
import pytest


# Create a minimal FastMCP stub so server imports succeed
class DummyFastMCP:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.title = kwargs.get("title")
        self.version = kwargs.get("version")


# Install stub package hierarchy mcp.server.fastmcp
mcp_mod = types.ModuleType("mcp")
server_mod = types.ModuleType("mcp.server")
fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
setattr(fastmcp_mod, "FastMCP", DummyFastMCP)
setattr(server_mod, "fastmcp", fastmcp_mod)
setattr(mcp_mod, "server", server_mod)
sys.modules.setdefault("mcp", mcp_mod)
sys.modules.setdefault("mcp.server", server_mod)
sys.modules.setdefault("mcp.server.fastmcp", fastmcp_mod)

from ragster import server  # noqa: E402


@pytest.mark.asyncio
async def test_get_package_version_pyproject(monkeypatch):
    """Fallback to pyproject when package metadata is missing."""

    def fake_version(_name: str) -> str:
        raise server.PackageNotFoundError

    toml = b'[project]\nversion = "1.2.3"\n'

    import builtins

    monkeypatch.setattr(server, "version", fake_version)
    monkeypatch.setattr(server.Path, "exists", lambda self: True)
    monkeypatch.setattr(builtins, "open", lambda *_args, **_kw: io.BytesIO(toml))

    assert server.get_package_version() == "1.2.3"


@pytest.mark.asyncio
async def test_get_package_version_default(monkeypatch):
    """Fallback to static version when metadata and file access fail."""

    def fake_version(_name: str) -> str:
        raise server.PackageNotFoundError

    monkeypatch.setattr(server, "version", fake_version)
    monkeypatch.setattr(server.Path, "exists", lambda self: False)

    assert server.get_package_version() == "0.4.0"


@pytest.mark.asyncio
async def test_cleanup_clients_order():
    order: list[str] = []

    class HttpClient:
        def __init__(self) -> None:
            self.is_closed = False

        async def aclose(self) -> None:
            order.append("http_client")
            self.is_closed = True

    class Embedding:
        async def close_voyage_client(self) -> None:
            order.append("embedding_client")

    class External:
        async def close(self) -> None:
            order.append("external_api_client")

    class Milvus:
        async def close(self) -> None:
            order.append("milvus_operator")

    clients = {
        "http_client": HttpClient(),
        "embedding_client": Embedding(),
        "external_api_client": External(),
        "milvus_operator": Milvus(),
    }

    await server._cleanup_clients(clients)

    assert order == [
        "http_client",
        "embedding_client",
        "external_api_client",
        "milvus_operator",
    ]


@pytest.mark.asyncio
async def test_perform_index_warmup(monkeypatch):
    calls: list[str] = []

    class StubMilvus:
        async def has_data(self) -> bool:
            calls.append("has_data")
            return True

        async def get_stored_topics(self, limit: int = 10):
            calls.append(f"get_stored_topics:{limit}")
            return ["topic1", "topic2"]

        async def query_data(self, vector, top_k: int = 3):
            calls.append(f"query_data:{vector}")
            return []

    class StubEmbedding:
        async def embed_texts(self, text, input_type):
            calls.append(f"embed_texts:{text}")
            return [0.1, 0.2, 0.3]

    class StubExternal:
        pass

    from typing import cast

    ctx = server.AppContext.model_construct(
        embedding_client=cast(server.EmbeddingClient, StubEmbedding()),
        milvus_operator=cast(server.MilvusOperator, StubMilvus()),
        external_api_client=cast(server.ExternalAPIClient, StubExternal()),
        http_client=cast(server.httpx.AsyncClient, object()),
    )

    await server._perform_index_warmup(ctx)

    expected_queries = [
        "topic1",
        "overview of topic1",
        "examples of topic1",
        "topic2",
        "overview of topic2",
        "examples of topic2",
    ]

    embed_calls = [c for c in calls if c.startswith("embed_texts:")]
    query_calls = [c for c in calls if c.startswith("query_data:")]

    assert embed_calls == [f"embed_texts:{q}" for q in expected_queries]
    assert len(query_calls) == len(expected_queries)


@pytest.mark.asyncio
async def test_create_mcp_server(monkeypatch):
    monkeypatch.setattr(server, "get_package_version", lambda: "9.9.9")

    app = server.create_mcp_server()

    assert isinstance(app, DummyFastMCP)
    assert app.title == "Ragster the RAG Context Server"
    assert app.version == "9.9.9"
