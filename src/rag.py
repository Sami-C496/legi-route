"""
Unified RAG interface for LégiRoute.

Usage:
    from src.rag import RAG

    rag = RAG()                          # uses default provider from .env
    rag = RAG(provider="gemini")         # explicit

    # Single query
    answer = rag.query("Vitesse sur autoroute ?")
    print(answer.response)
    print(answer.sources)

    # Streaming
    for chunk in rag.stream("Vitesse sur autoroute ?"):
        print(chunk, end="")

    # Batch (for evaluation)
    results = rag.batch(["Q1", "Q2", "Q3"])
"""

import logging
from dataclasses import dataclass, field
from typing import Iterator

from src.config import settings, Provider
from src.providers import get_provider, LLMProvider
from src.classifier import IntentClassifier, Intent
from src.retrieval import TrafficRetriever
from src.generation import TrafficGenerator
from src.models import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    query: str
    intent: Intent
    response: str
    sources: list[RetrievalResult] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)


class RAG:

    def __init__(self, provider: str = None):
        if provider:
            settings.PROVIDER = Provider(provider)

        self.provider: LLMProvider = get_provider()
        self.classifier = IntentClassifier(self.provider)
        self.retriever = TrafficRetriever(self.provider)
        self.generator = TrafficGenerator(self.provider)

    def query(self, question: str, k: int = 3) -> RAGResponse:
        """Full RAG pipeline: classify -> retrieve -> generate."""
        intent = self.classifier.classify(question)

        if intent == Intent.OFF_TOPIC:
            return RAGResponse(
                query=question,
                intent=intent,
                response="Je suis spécialisé dans le Code de la Route français. "
                         "Je ne peux pas répondre à cette question.",
            )

        sources = []
        if intent == Intent.LEGAL_QUERY:
            results = self.retriever.search(question, k=k)
            sources = [r for r in results if r.score > settings.RELEVANCE_THRESHOLD]

        response = self.generator.generate(question, sources)
        contexts = [
            f"Article {r.article.article_number} : {r.article.content}"
            for r in sources
        ]

        return RAGResponse(
            query=question,
            intent=intent,
            response=response,
            sources=sources,
            contexts=contexts,
        )

    def stream(self, question: str, k: int = 3) -> Iterator[str]:
        """Streaming variant. Yields response chunks."""
        intent = self.classifier.classify(question)

        if intent == Intent.OFF_TOPIC:
            yield "Je suis spécialisé dans le Code de la Route français."
            return

        sources = []
        if intent == Intent.LEGAL_QUERY:
            results = self.retriever.search(question, k=k)
            sources = [r for r in results if r.score > settings.RELEVANCE_THRESHOLD]

        yield from self.generator.generate_stream(question, sources)

    def batch(self, questions: list[str], k: int = 3) -> list[RAGResponse]:
        """Run multiple queries. Useful for evaluation."""
        return [self.query(q, k=k) for q in questions]
