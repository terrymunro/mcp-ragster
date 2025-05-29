from ragster.embedding_client import EmbeddingClient, VoyageInputType
from ragster.external_apis import ExternalAPIClient
from ragster.milvus_ops import MilvusOperator


class StubEmbeddingClient(EmbeddingClient):
    def __init__(self, return_dim: int = 3):
        # intentionally skip super().__init__ to avoid httpx setup
        self.embed_texts_calls: list[tuple[object, VoyageInputType | str]] = []
        self.return_dim = return_dim

    async def embed_texts(
        self, texts: object, input_type: VoyageInputType | str
    ):  # type: ignore[override]
        self.embed_texts_calls.append((texts, input_type))
        if isinstance(texts, list):
            return [[0.0] * self.return_dim for _ in texts]
        return [0.0] * self.return_dim

    def get_embedding_dimension(self) -> int:  # type: ignore[override]
        return self.return_dim


class StubExternalAPIClient(ExternalAPIClient):
    def __init__(self, jina_result_count: int = 2):
        # skip base init to avoid network setup
        self.jina_result_count = jina_result_count
        self.search_jina_calls: list[tuple[str, int]] = []
        self.query_perplexity_calls: list[str] = []
        self.crawl_url_firecrawl_calls: list[str] = []

    async def search_jina(self, topic: str, num_results: int = 5):  # type: ignore[override]
        self.search_jina_calls.append((topic, num_results))
        return [
            {
                "url": f"http://example.com/{i}",
                "title": f"Title {i}",
                "snippet": f"Snippet {i}",
            }
            for i in range(self.jina_result_count)
        ]

    async def query_perplexity(self, topic: str) -> str:  # type: ignore[override]
        self.query_perplexity_calls.append(topic)
        return f"Summary for {topic}"

    async def crawl_url_firecrawl(self, url: str):  # type: ignore[override]
        self.crawl_url_firecrawl_calls.append(url)
        return {"content": f"Content for {url}", "source_url": url, "type": "markdown"}


class StubMilvusOperator(MilvusOperator):
    def __init__(self):
        # skip base init to avoid real Milvus connection details
        self.insert_calls: list[list[dict[str, object]]] = []
        self.query_calls: list[tuple[list[float], int, str | None, int | None]] = []

    async def insert_data(self, data_rows: list[dict[str, object]]):  # type: ignore[override]
        self.insert_calls.append(data_rows)
        return list(range(len(data_rows)))

    async def query_data(
        self,
        query_vector: list[float],
        top_k: int,
        expr: str | None = None,
        search_ef: int | None = None,
    ) -> list[dict[str, object]]:  # type: ignore[override]
        self.query_calls.append((query_vector, top_k, expr, search_ef))
        return [
            {
                "id": i,
                "text_content": f"text {i}",
                "source_type": "stub",
                "source_identifier": f"id{i}",
                "topic": "test_topic",
                "distance": float(i),
            }
            for i in range(top_k)
        ]
