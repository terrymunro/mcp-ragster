"""Pydantic models for request/response data structures."""

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .config import settings


class LoadTopicRequest(BaseModel):
    """Request model for loading topic information."""

    # Multi-topic support with backward compatibility
    topics: list[str] = Field(
        default_factory=list,
        min_length=1,
        max_length=settings.MAX_TOPICS_PER_JOB,
        description="List of topics to research (1-10 topics). For backward compatibility, if 'topic' is provided instead, it will be converted to a single-item list.",
    )

    # Backward compatibility: single topic field
    topic: str | None = Field(
        default=None,
        min_length=1,
        description="Single topic to research (deprecated, use 'topics' instead).",
    )

    @field_validator("topics")
    @classmethod
    def validate_topics(cls, v: list[str]) -> list[str]:
        """Validate topics list."""
        if not v:
            return v  # Will be handled in model_validator

        # Strip whitespace and filter empty strings
        cleaned_topics = [topic.strip() for topic in v if topic.strip()]

        if not cleaned_topics:
            raise ValueError("All topics are empty or contain only whitespace")

        if len(cleaned_topics) > settings.MAX_TOPICS_PER_JOB:
            raise ValueError(
                f"Too many topics: {len(cleaned_topics)} > {settings.MAX_TOPICS_PER_JOB}"
            )

        return cleaned_topics

    @model_validator(mode="after")
    def ensure_topics_provided(self) -> "LoadTopicRequest":
        """Ensure either topics or topic is provided, with backward compatibility."""
        if not self.topics and not self.topic:
            raise ValueError("Either 'topics' list or 'topic' string must be provided")

        # Backward compatibility: convert single topic to topics list
        if self.topic and not self.topics:
            self.topics = [self.topic.strip()]

        # If both are provided, topics takes precedence
        if self.topics and self.topic:
            # Clear the deprecated field to avoid confusion
            self.topic = None

        return self


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
