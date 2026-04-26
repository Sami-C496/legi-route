"""Singleton dependency providers for the API."""

from functools import lru_cache

from src.rag import RAG


@lru_cache(maxsize=1)
def get_rag() -> RAG:
    """Return a process-wide RAG instance (loaded on first request)."""
    return RAG()
