"""
Provider abstraction for LLM and embedding APIs.
Supports Gemini and Mistral with a unified interface.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Iterator

from src.config import Provider, settings

logger = logging.getLogger(__name__)


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

    def embed(self, texts, task_type="document"):
        task = "RETRIEVAL_DOCUMENT" if task_type == "document" else "RETRIEVAL_QUERY"
        response = self.client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=texts,
            config=self._types.EmbedContentConfig(task_type=task),
        )
        return [e.values for e in response.embeddings]

    def generate_stream(self, prompt, system, **kwargs):
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