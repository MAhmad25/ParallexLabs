"""Pydantic request and response models for the RAG API."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """POST /query request body."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language question to search the corpus",
    )
    k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top results to return",
    )
    topic_id: int | None = Field(
        default=None,
        ge=0,
        description="Filter to chunks assigned to this LDA topic ID",
    )
    topic_label: str | None = Field(
        default=None,
        description="Filter to chunks with this reviewed topic label",
    )
    sentiment: str | None = Field(
        default=None,
        pattern=r"^(positive|neutral|negative|unknown)$",
        description="Filter to chunks with this VADER sentiment label",
    )


class HitResponse(BaseModel):
    """A single retrieval result."""

    rank: int
    id: str
    title: str
    topic_id: int
    topic_label: str
    topic_confidence: float
    sentiment: str
    sentiment_compound: float
    cosine_similarity: float
    text: str


class QueryResponse(BaseModel):
    """POST /query response body."""

    question: str
    filters: dict
    count: int
    hits: list[HitResponse]


class HealthResponse(BaseModel):
    """GET /health response body."""

    status: str
    collection: str
    chunk_count: int


class CollectionStats(BaseModel):
    """GET /collection/stats response body."""

    collection: str
    chunk_count: int
    embedding_model: str
    embedding_dimensions: int
    distance_metric: str


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str
