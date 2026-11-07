"""FastAPI RAG service wrapping the Week 4 retrieval index.

Endpoints
---------
GET  /health            Liveness and readiness check.
POST /query             Embed a question and retrieve top-K chunks with optional filters.
GET  /collection/stats  Index metadata (model, dimensions, count, distance metric).
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.gemini import GeminiEmbedder, Settings
from src.logging_config import RequestLoggingMiddleware, setup_logging
from src.schemas import (
    CollectionStats,
    ErrorResponse,
    HealthResponse,
    HitResponse,
    QueryRequest,
    QueryResponse,
)
from src.vector_store import build_where, open_collection, query_collection


# ---------------------------------------------------------------------------
# Lifespan — connect to the index and create the embedder once
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources on startup and tear them down on shutdown."""
    logger = logging.getLogger("rag_api")
    try:
        settings = Settings.from_env()
        embedder = GeminiEmbedder(settings)
        collection, manifest = open_collection()
    except Exception as exc:
        logger.error("Startup failed: %s", exc)
        raise

    app.state.embedder = embedder
    app.state.collection = collection
    app.state.manifest = manifest
    app.state.settings = settings
    logger.info(
        "RAG service ready — collection=%s chunks=%d",
        manifest["collection"],
        manifest["chunk_count"],
    )
    yield
    embedder.close()
    logger.info("RAG service shut down")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

setup_logging()

app = FastAPI(
    title="Week 5 RAG API",
    description="FastAPI wrapper around the Week 4 Wikipedia RAG retrieval index.",
    version="0.1.0",
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)

app.add_middleware(RequestLoggingMiddleware)


# ---------------------------------------------------------------------------
# Error handlers — structured JSON for 400 / 422 / 500
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Return 422 with a human-readable detail string."""
    errors = exc.errors()
    messages = []
    for error in errors:
        location = " → ".join(str(part) for part in error.get("loc", []))
        messages.append(f"{location}: {error.get('msg', 'validation error')}")
    return JSONResponse(status_code=422, content={"detail": "; ".join(messages)})


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Return 400 for application-level value errors."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    """Return 500 with a safe message for unexpected errors."""
    logging.getLogger("rag_api").exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check the server logs for details."},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request):
    """Liveness and readiness probe."""
    manifest = request.app.state.manifest
    collection = request.app.state.collection
    return HealthResponse(
        status="ok",
        collection=manifest["collection"],
        chunk_count=collection.count(),
    )


@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query(body: QueryRequest, request: Request):
    """Embed a question, retrieve top-K chunks, and return ranked results.

    Supports optional metadata filters: ``topic_id``, ``topic_label``,
    ``sentiment``.  Filters are combined with AND when multiple are supplied.
    """
    embedder: GeminiEmbedder = request.app.state.embedder
    collection = request.app.state.collection
    manifest = request.app.state.manifest
    identity = manifest["identity"]

    # Validate that the embedder matches the index
    if (
        embedder.settings.embedding_model != identity["embedding_model"]
        or embedder.settings.dimensions != identity["dimensions"]
    ):
        return JSONResponse(
            status_code=500,
            content={"detail": "Embedding model/dimensions mismatch with the indexed vectors."},
        )

    where = build_where(
        topic_id=body.topic_id,
        topic_label=body.topic_label,
        sentiment=body.sentiment,
    )

    query_vector = embedder.embed([body.question], query=True)[0]
    hits = query_collection(collection, query_vector, k=body.k, where=where)

    return QueryResponse(
        question=body.question,
        filters=where or {},
        count=len(hits),
        hits=[
            HitResponse(
                rank=rank,
                id=hit["id"],
                title=hit["metadata"].get("title", ""),
                topic_id=hit["metadata"]["topic_id"],
                topic_label=hit["metadata"]["topic_label"],
                topic_confidence=hit["metadata"]["topic_confidence"],
                sentiment=hit["metadata"]["sentiment"],
                sentiment_compound=hit["metadata"]["sentiment_compound"],
                cosine_similarity=hit["cosine_similarity"],
                text=hit["text"],
            )
            for rank, hit in enumerate(hits, 1)
        ],
    )


@app.get("/collection/stats", response_model=CollectionStats, tags=["ops"])
def collection_stats(request: Request):
    """Return metadata about the active retrieval index."""
    manifest = request.app.state.manifest
    collection = request.app.state.collection
    identity = manifest["identity"]
    return CollectionStats(
        collection=manifest["collection"],
        chunk_count=collection.count(),
        embedding_model=identity["embedding_model"],
        embedding_dimensions=identity["dimensions"],
        distance_metric=manifest.get("distance_metric", "cosine"),
    )
