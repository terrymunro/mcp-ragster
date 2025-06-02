"""Job management data models for asynchronous research operations."""

from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, field_validator

from .models import LoadTopicResponse


class JobStatus(str, Enum):
    """Status of a research job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TopicProgress:
    """Progress tracking for a single topic within a research job."""

    topic: str
    jina_status: str = "pending"  # pending, running, completed, failed
    perplexity_status: str = "pending"
    firecrawl_status: str = "pending"
    urls_found: int = 0
    urls_processed: int = 0
    errors: list[str] = field(default_factory=list)

    def get_completion_percentage(self) -> float:
        """Calculate completion percentage for this topic (0.0 to 1.0)."""
        stages_completed = 0
        total_stages = 3

        if self.jina_status == "completed":
            stages_completed += 1
        if self.perplexity_status == "completed":
            stages_completed += 1
        if self.firecrawl_status == "completed":
            stages_completed += 1
        elif self.firecrawl_status == "running" and self.urls_found > 0:
            # Partial credit for Firecrawl progress
            firecrawl_progress = self.urls_processed / self.urls_found
            stages_completed += firecrawl_progress

        return stages_completed / total_stages

    def is_completed(self) -> bool:
        """Check if all stages for this topic are completed."""
        return (
            self.jina_status == "completed"
            and self.perplexity_status == "completed"
            and self.firecrawl_status == "completed"
        )

    def has_errors(self) -> bool:
        """Check if this topic has any errors."""
        return len(self.errors) > 0


@dataclass
class JobMetrics:
    """Performance metrics for a research job."""

    total_api_calls: int = 0
    jina_api_calls: int = 0
    perplexity_api_calls: int = 0
    firecrawl_api_calls: int = 0
    total_processing_time_seconds: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (0.0 to 1.0)."""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return self.cache_hits / total_requests


@dataclass
class ResearchJob:
    """Represents a research job with multiple topics."""

    job_id: str
    topics: list[str]
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: dict[str, TopicProgress] = field(default_factory=dict)
    results: Optional[LoadTopicResponse] = None
    error: Optional[str] = None
    metrics: JobMetrics = field(default_factory=JobMetrics)

    def __post_init__(self) -> None:
        """Initialize progress tracking for all topics."""
        if not self.progress:
            self.progress = {topic: TopicProgress(topic=topic) for topic in self.topics}

    def get_overall_progress(self) -> float:
        """Calculate overall job progress (0.0 to 1.0)."""
        if not self.progress:
            return 0.0

        total_progress = sum(
            topic_progress.get_completion_percentage()
            for topic_progress in self.progress.values()
        )
        return total_progress / len(self.progress)

    def get_estimated_completion_time(self) -> Optional[datetime]:
        """Estimate when the job will complete based on current progress."""
        if self.status != JobStatus.RUNNING or not self.started_at:
            return None

        progress = self.get_overall_progress()
        if progress <= 0.0:
            return None

        elapsed = (datetime.now(UTC) - self.started_at).total_seconds()
        estimated_total_time = elapsed / progress
        remaining_time = estimated_total_time - elapsed

        return datetime.now(UTC) + timedelta(seconds=remaining_time)

    def get_completed_topics(self) -> list[str]:
        """Get list of topics that have completed successfully."""
        return [
            topic
            for topic, progress in self.progress.items()
            if progress.is_completed()
        ]

    def get_failed_topics(self) -> list[str]:
        """Get list of topics that have failed or have errors."""
        return [
            topic for topic, progress in self.progress.items() if progress.has_errors()
        ]

    def is_complete(self) -> bool:
        """Check if all topics in the job are completed."""
        return all(progress.is_completed() for progress in self.progress.values())

    def update_topic_progress(
        self, topic: str, stage: str, status: str, **kwargs: Any
    ) -> None:
        """Update progress for a specific topic and stage."""
        if topic not in self.progress:
            self.progress[topic] = TopicProgress(topic=topic)

        topic_progress = self.progress[topic]

        if stage == "jina":
            topic_progress.jina_status = status
            if "urls_found" in kwargs:
                topic_progress.urls_found = kwargs["urls_found"]
        elif stage == "perplexity":
            topic_progress.perplexity_status = status
        elif stage == "firecrawl":
            topic_progress.firecrawl_status = status
            if "urls_processed" in kwargs:
                topic_progress.urls_processed = kwargs["urls_processed"]

        if "error" in kwargs:
            topic_progress.errors.append(kwargs["error"])


# Pydantic models for API responses
class ResearchJobResponse(BaseModel):
    """Response when creating a new research job."""

    job_id: str = Field(..., description="Unique identifier for the research job")
    status: str = Field(..., description="Current status of the job")
    topics: list[str] = Field(..., description="List of topics being researched")
    message: str = Field(..., description="Human-readable status message")
    created_at: datetime = Field(..., description="When the job was created")
    estimated_completion_time: Optional[datetime] = Field(
        None, description="Estimated completion time based on topic count"
    )


class JobStatusResponse(BaseModel):
    """Response for job status queries."""

    job_id: str = Field(..., description="Unique identifier for the research job")
    status: str = Field(..., description="Current status of the job")
    topics: list[str] = Field(..., description="List of topics being researched")
    overall_progress: float = Field(
        ..., ge=0.0, le=1.0, description="Overall progress from 0.0 to 1.0"
    )
    topic_progress: dict[str, dict[str, Any]] = Field(
        ..., description="Detailed progress for each topic"
    )
    estimated_completion_time: Optional[datetime] = Field(
        None, description="Estimated completion time"
    )
    results: Optional[LoadTopicResponse] = Field(
        None, description="Results if job is completed"
    )
    error: Optional[str] = Field(None, description="Error message if job failed")


class MultiTopicResponse(BaseModel):
    """Response for completed multi-topic research jobs."""

    message: str = Field(..., description="Summary message")
    topics: list[str] = Field(..., description="All topics that were researched")
    total_urls_processed: int = Field(
        0, description="Total URLs processed across all topics"
    )
    successful_topics: list[str] = Field(
        default_factory=list, description="Topics that completed successfully"
    )
    failed_topics: list[str] = Field(
        default_factory=list, description="Topics that failed or had errors"
    )
    topic_results: dict[str, LoadTopicResponse] = Field(
        default_factory=dict, description="Individual results for each topic"
    )
    overall_errors: list[str] = Field(
        default_factory=list, description="Job-level errors"
    )


# Tool argument models for new MCP tools
class GetJobStatusArgs(BaseModel):
    """Arguments for get_research_status tool."""

    job_id: str = Field(
        ..., min_length=1, description="ID of the research job to query"
    )


class ListJobsArgs(BaseModel):
    """Arguments for list_research_jobs tool."""

    status_filter: Optional[str] = Field(
        None,
        description="Filter jobs by status (pending, running, completed, failed, cancelled)",
    )
    limit: int = Field(10, ge=1, le=100, description="Maximum number of jobs to return")
    offset: int = Field(0, ge=0, description="Number of jobs to skip for pagination")

    @field_validator("status_filter")
    @classmethod
    def validate_status_filter(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in {status.value for status in JobStatus}:
            raise ValueError(f"Invalid status filter: {v}")
        return v


class CancelJobArgs(BaseModel):
    """Arguments for cancel_research_job tool."""

    job_id: str = Field(
        ..., min_length=1, description="ID of the research job to cancel"
    )


class ListJobsResponse(BaseModel):
    """Response for list_research_jobs tool."""

    jobs: list[JobStatusResponse] = Field(..., description="List of jobs")
    total_count: int = Field(..., description="Total number of jobs matching filter")
    has_more: bool = Field(
        ..., description="Whether there are more jobs beyond this page"
    )


class CancelJobResponse(BaseModel):
    """Response for cancel_research_job tool."""

    job_id: str = Field(..., description="ID of the cancelled job")
    status: str = Field(..., description="New status after cancellation")
    message: str = Field(..., description="Cancellation result message")
    preserved_results: Optional[dict[str, LoadTopicResponse]] = Field(
        None, description="Any completed topic results that were preserved"
    )
