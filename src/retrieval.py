import logging
from pinecone import Pinecone

from src.config import settings
from src.models import RetrievalResult, TrafficLawArticle
from src.providers import LLMProvider

logger = logging.getLogger(__name__)


class TrafficRetriever:

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = pc.Index(settings.PINECONE_INDEX_NAME)

    def search(self, query: str, k: int = None) -> list[RetrievalResult]:
        k = k or settings.DEFAULT_TOP_K

        if not query or len(query.strip()) < 3:
            return []

        try:
            vectors = self.provider.embed([query], task_type="query")
            query_vector = vectors[0]
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []

        try:
            results = self.index.query(
                vector=query_vector,
                top_k=k,
                include_metadata=True,
            )
        except Exception as e:
            logger.error(f"Pinecone query failed: {e}")
            return []

        clean_results = []
        for match in results.matches:
            try:
                meta = match.metadata
                article = TrafficLawArticle(
                    id=meta.get("article_id", "unknown"),
                    article_number=meta.get("num", "N/A"),
                    content=meta.get("content", ""),
                    context=meta.get("category", "Code de la Route"),
                    url=meta.get("url"),
                )
                clean_results.append(RetrievalResult(article=article, score=match.score))
            except Exception as e:
                logger.warning(f"Failed to parse match {match.id}: {e}")

        return clean_results
