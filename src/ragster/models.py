"""Pydantic models for request/response data structures."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

if __package__:
    from .config import settings
else:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    from ragster.config import settings


class LoadTopicRequest(BaseModel):
    """Request model for loading topic information."""

    topic: str = Field(
        ..., min_length=1, description="The topic to load information about."
    )


class LoadTopicResponse(BaseModel):
    """Response model for topic loading operations."""

    message: str
    topic: str
    urls_processed: int = 0
    perplexity_queried: bool = False
    jina_results_found: int = 0
    errors: list[str] = []


class QueryTopicRequest(BaseModel):
    """Request model for querying topic context."""

    query: str = Field(
        ..., min_length=1, description="The query to search for relevant context."
    )
    top_k: int | None = Field(
        default_factory=lambda: settings.MILVUS_SEARCH_LIMIT,
        gt=0,
        description="Number of results to return.",
    )


class DocumentFragment(BaseModel):
    """Model for individual search result documents."""

    id: Any
    text_content: str
    source_type: str
    source_identifier: str
    topic: str
    distance: float | None = None  # Relevance score from Milvus


class QueryTopicResponse(BaseModel):
    """Response model for topic query operations."""

    query: str
    results: list[DocumentFragment]
    message: str | None = None
