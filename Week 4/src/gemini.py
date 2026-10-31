"""Gemini query/document embeddings with bounded retries and hidden credentials."""

import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from time import sleep

import httpx
from dotenv import dotenv_values, load_dotenv
from google import genai
from google.genai import errors, types

from src.paths import ROOT, REPO_ROOT


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    embedding_model: str = "gemini-embedding-001"
    dimensions: int = 768
    timeout_seconds: int = 45

    @classmethod
    def from_env(cls):
        load_dotenv(ROOT / ".env", override=False)
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            previous = dotenv_values(REPO_ROOT / "Week 2" / ".env")
            key = previous.get("GEMINI_API_KEY") or previous.get("GOOGLE_API_KEY")
            if key:
                os.environ["GEMINI_API_KEY"] = key
        return cls(
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", ""),
            embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
            dimensions=int(os.getenv("GEMINI_EMBEDDING_DIMENSIONS", "768")),
            timeout_seconds=int(os.getenv("GEMINI_TIMEOUT_SECONDS", "45")),
        )

    def validate(self):
        if not self.api_key or self.api_key == "replace_with_your_google_ai_studio_key":
            raise ValueError("Set GEMINI_API_KEY in Week 4/.env or the shell")
        if self.embedding_model not in {"gemini-embedding-001", "gemini-embedding-2"}:
            raise ValueError("Use gemini-embedding-001 or gemini-embedding-2")
        if self.dimensions not in {768, 1536, 3072}:
            raise ValueError("Use 768, 1536, or 3072 dimensions")
        if not 5 <= self.timeout_seconds <= 180:
            raise ValueError("GEMINI_TIMEOUT_SECONDS must be between 5 and 180")


class GeminiEmbedder:
    def __init__(self, settings: Settings):
        settings.validate()
        self.settings = settings
        self.request_count = 0
        self.client = genai.Client(
            api_key=settings.api_key,
            vertexai=False,
            http_options=types.HttpOptions(
                timeout=settings.timeout_seconds * 1000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    def close(self):
        self.client.close()

    def _request(self, operation):
        for attempt in range(3):
            try:
                self.request_count += 1
                return operation()
            except errors.APIError as error:
                try:
                    code = int(error.code)
                except (TypeError, ValueError):
                    code = 0
                if code in {408, 429, 500, 502, 503, 504} and attempt < 2:
                    sleep(min(8, 2**attempt) + random.uniform(0, 0.2))
                    continue
                help_text = {
                    401: "Check GEMINI_API_KEY.",
                    403: "Check project access, quota, and billing.",
                    404: "The configured embedding model is unavailable for this key.",
                    429: "Quota or rate limit reached; wait or increase request pacing.",
                }.get(code, "Check model access, network, and quota.")
                raise RuntimeError(f"Gemini HTTP {code}. {help_text}") from None
            except (OSError, TimeoutError, httpx.TransportError) as error:
                if attempt < 2:
                    sleep(2 ** (attempt + 1))
                    continue
                raise RuntimeError(f"Gemini transport failed ({type(error).__name__}).") from None
        raise RuntimeError("Gemini retry limit exceeded")

    def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding inputs must be nonempty text")
        if any(len(text) > 8000 for text in texts):
            raise ValueError("Each embedding input must be at most 8000 characters")
        config = types.EmbedContentConfig(output_dimensionality=self.settings.dimensions)
        if self.settings.embedding_model == "gemini-embedding-001":
            config.task_type = "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT"
            response = self._request(
                lambda: self.client.models.embed_content(
                    model=self.settings.embedding_model,
                    contents=texts,
                    config=config,
                )
            )
            vectors = [item.values for item in (response.embeddings or [])]
        else:
            vectors = []
            prefix = "task: search result | query: " if query else "title: none | text: "
            for text in texts:
                response = self._request(
                    lambda text=text: self.client.models.embed_content(
                        model=self.settings.embedding_model,
                        contents=prefix + text,
                        config=config,
                    )
                )
                if not response.embeddings or len(response.embeddings) != 1:
                    raise ValueError("Gemini returned an unexpected embedding count")
                vectors.append(response.embeddings[0].values)
        if len(vectors) != len(texts):
            raise ValueError("Gemini returned an unexpected embedding count")
        normalized = []
        for vector in vectors:
            if len(vector) != self.settings.dimensions or not all(math.isfinite(v) for v in vector):
                raise ValueError("Gemini returned an invalid vector dimension or value")
            norm = math.hypot(*vector)
            if norm == 0:
                raise ValueError("Gemini returned a zero-magnitude vector")
            normalized.append([float(value / norm) for value in vector])
        return normalized
