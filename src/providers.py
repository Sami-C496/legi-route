"""
Provider abstraction for LLM and embedding APIs.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Iterator

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    retry_if_exception,
    before_sleep_log,
)

from src.config import Provider, settings

logger = logging.getLogger(__name__)


def _is_query_retriable(exc: Exception) -> bool:
    """Retry on transient errors (503, 429, timeouts), not on client errors (400, 403)."""
    exc_str = str(exc).lower()
    if any(code in exc_str for code in ("400", "invalid", "permission", "403", "404")):
        return False
    if any(p in exc_str for p in ("503", "unavailable", "429", "rate", "timeout", "deadline")):
        return True
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    return False


def _query_retry():
    """Retry decorator for interactive (query-time) API calls."""
    return retry(
        stop=stop_after_attempt(settings.QUERY_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=settings.QUERY_RETRY_MIN_WAIT, max=settings.QUERY_RETRY_MAX_WAIT) + wait_random(0, 1),
        retry=retry_if_exception(_is_query_retriable),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


class LLMProvider(ABC):
    """Unified interface for LLM + embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str], task_type: str = "document") -> list[list[float]]:
        """Embed a list of texts. task_type: 'document' or 'query'."""

    @abstractmethod
    def generate_stream(self, prompt: str, system: str, **kwargs) -> Iterator[str]:
        """Stream a completion."""

    @abstractmethod
    def classify_intent(self, query: str, system: str) -> dict:
        """Return a structured JSON dict from the classifier model."""


class GeminiProvider(LLMProvider):

    def __init__(self):
        from google import genai
        from google.genai import types
        self._types = types
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)

    @_query_retry()
    def embed(self, texts, task_type="document"):
        task = "RETRIEVAL_DOCUMENT" if task_type == "document" else "RETRIEVAL_QUERY"
        response = self.client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=texts,
            config=self._types.EmbedContentConfig(task_type=task),
        )
        return [e.values for e in response.embeddings]

    def generate_stream(self, prompt, system, **kwargs):
        """Stream a completion with manual retry (tenacity doesn't support generators)."""
        last_exc = None
        for attempt in range(settings.QUERY_MAX_RETRIES):
            try:
                response = self.client.models.generate_content_stream(
                    model=settings.GENERATION_MODEL,
                    contents=prompt,
                    config=self._types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=kwargs.get("temperature", settings.GENERATION_TEMPERATURE),
                        max_output_tokens=kwargs.get("max_tokens", settings.GENERATION_MAX_TOKENS),
                    ),
                )
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
                return
            except Exception as e:
                last_exc = e
                if not _is_query_retriable(e):
                    raise
                wait = min(settings.QUERY_RETRY_MIN_WAIT * (2 ** attempt), settings.QUERY_RETRY_MAX_WAIT)
                logger.warning("generate_stream attempt %d/%d failed: %s. Retrying in %.1fs",
                               attempt + 1, settings.QUERY_MAX_RETRIES, e, wait)
                time.sleep(wait)
        raise last_exc

    @_query_retry()
    def classify_intent(self, query, system):
        from src.classifier import INTENT_SCHEMA

        response = self.client.models.generate_content(
            model=settings.CLASSIFIER_MODEL,
            contents=f"Message utilisateur : {query}",
            config=self._types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.0,
                max_output_tokens=50,
                response_mime_type="application/json",
                response_schema=INTENT_SCHEMA,
            ),
        )
        return json.loads(response.text)


def get_provider(provider: Provider = None) -> LLMProvider:
    """Factory: returns the configured provider instance."""
    provider = provider or settings.PROVIDER
    if provider == Provider.GEMINI:
        return GeminiProvider()
    raise ValueError(f"Unknown provider: {provider}")
