"""Structured JSON request logging middleware for the FastAPI application."""

import json
import logging
import sys
import time
import uuid
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.paths import LOG_DIR


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge any structured extras attached by the middleware
        for key in ("request_id", "method", "path", "status_code", "duration_ms", "client"):
            value = getattr(record, key, None)
            if value is not None:
                entry[key] = value
        return json.dumps(entry, ensure_ascii=False)


def setup_logging() -> logging.Logger:
    """Configure the application-wide JSON logger."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("rag_api")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = JSONFormatter()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(LOG_DIR / "requests.jsonl", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with method, path, status, and duration."""

    async def dispatch(self, request: Request, call_next) -> Response:
        logger = logging.getLogger("rag_api")
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        # Attach request_id so downstream handlers can reference it
        request.state.request_id = request_id

        try:
            response: Response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": 500,
                    "duration_ms": duration_ms,
                    "client": request.client.host if request.client else None,
                },
            )
            from starlette.responses import JSONResponse
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error. Check the server logs for details."},
            )

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "client": request.client.host if request.client else None,
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response
