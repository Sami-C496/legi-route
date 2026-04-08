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

from src.config import settings, Provider, HISTORY_WINDOW
from src.providers import get_provider, LLMProvider
from src.classifier import IntentClassifier, Intent
from src.retrieval import TrafficRetriever
from src.generation import TrafficGenerator
from src.models import RetrievalResult

logger = logging.getLogger(__name__)

_OFF_TOPIC_RESPONSE = (
    "Je suis spécialisé dans le Code de la Route français. "
    "Je ne peux pas répondre à cette question."
)

_UNAVAILABLE_RESPONSE = (
    "Le service est momentanément indisponible. "
    "Veuillez réessayer dans quelques instants."
)

_REWRITE_PROMPT = """Tu reformules des questions pour un moteur de recherche juridique.
Si la question fait référence à la conversation précédente (pronoms, "et", "mais", "ça", "il", "elle", références implicites à un sujet précédent), reformule-la en question complète et autonome.
Si la question est déjà autonome et complète, retourne-la EXACTEMENT telle quelle, sans aucune modification.
Réponds UNIQUEMENT avec la question. Aucune explication."""


@dataclass
class RAGResponse:
    query: str
    intent: Intent
    response: str
    sources: list[RetrievalResult] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    error: bool = False


class RAG:

    def __init__(self, provider: str = None):
        prov = Provider(provider) if provider else settings.PROVIDER
        self.provider: LLMProvider = get_provider(prov)
        self.classifier = IntentClassifier(self.provider)
        self.retriever = TrafficRetriever(self.provider)
        self.generator = TrafficGenerator(self.provider)

    def rewrite_query(self, question: str, history: list[dict]) -> str:
        """Reformulate a follow-up question into a standalone search query."""
        if not history:
            return question
        turns = []
        for msg in history[-HISTORY_WINDOW:]:
            role = "Utilisateur" if msg["role"] == "user" else "LégiRoute"
            turns.append(f"{role} : {msg['content']}")
        prompt = "Historique :\n" + "\n".join(turns) + f"\n\nQuestion : {question}"
        return "".join(self.provider.generate_stream(prompt, _REWRITE_PROMPT, temperature=0.0)).strip()

    def _retrieve(self, question: str, intent: Intent, k: int, history: list[dict] | None) -> list[RetrievalResult]:
        """Shared retrieval logic: rewrite -> search -> filter."""
        if intent != Intent.LEGAL_QUERY:
            return []
        search_query = self.rewrite_query(question, history or [])
        results = self.retriever.search(search_query, k=k)
        return [r for r in results if r.score > settings.RELEVANCE_THRESHOLD]

    def query(self, question: str, k: int = 5, history: list[dict] | None = None) -> RAGResponse:
        """Full RAG pipeline: classify -> retrieve -> generate."""
        try:
            intent = self.classifier.classify(question)
        except Exception as e:
            logger.error("Classification failed (retries exhausted): %s", e)
            intent = Intent.LEGAL_QUERY

        if intent == Intent.OFF_TOPIC:
            return RAGResponse(query=question, intent=intent, response=_OFF_TOPIC_RESPONSE)

        try:
            sources = self._retrieve(question, intent, k, history)
        except Exception as e:
            logger.error("Retrieval failed (retries exhausted): %s", e)
            return RAGResponse(query=question, intent=intent, response=_UNAVAILABLE_RESPONSE, error=True)

        try:
            response = self.generator.generate(question, sources, history=history)
        except Exception as e:
            logger.error("Generation failed (retries exhausted): %s", e)
            return RAGResponse(query=question, intent=intent, response=_UNAVAILABLE_RESPONSE, error=True)

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

    def stream(self, question: str, k: int = 5, history: list[dict] | None = None) -> Iterator[str]:
        """Streaming variant with graceful degradation."""
        try:
            intent = self.classifier.classify(question)
        except Exception as e:
            logger.error("Classification failed (retries exhausted): %s", e)
            intent = Intent.LEGAL_QUERY

        if intent == Intent.OFF_TOPIC:
            yield _OFF_TOPIC_RESPONSE
            return

        try:
            sources = self._retrieve(question, intent, k, history)
        except Exception as e:
            logger.error("Retrieval failed (retries exhausted): %s", e)
            yield _UNAVAILABLE_RESPONSE
            return

        try:
            yield from self.generator.generate_stream(question, sources, history=history)
        except Exception as e:
            logger.error("Generation failed (retries exhausted): %s", e)
            yield _UNAVAILABLE_RESPONSE

    def batch(self, questions: list[str], k: int = 3) -> list[RAGResponse]:
        """Run multiple queries. Useful for evaluation."""
        return [self.query(q, k=k) for q in questions]
