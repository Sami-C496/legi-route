import os
import logging
import chromadb

from src.config import settings
from src.models import RetrievalResult, TrafficLawArticle
from src.providers import LLMProvider

logger = logging.getLogger(__name__)


class TrafficRetriever:

    def __init__(self, provider: LLMProvider, db_path: str = None):
        self.provider = provider
        resolved_path = db_path or str(settings.CHROMA_DB_PATH)

        self.chroma_client = chromadb.PersistentClient(path=resolved_path)
        self.collection = self.chroma_client.get_collection(name=settings.COLLECTION_NAME)

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
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []

        clean_results = []
        if results and results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                try:
                    meta = results["metadatas"][0][i]
                    distance = results["distances"][0][i] if results["distances"] else 0.0
                    raw_content = meta.get("content") or results["documents"][0][i]

                    article = TrafficLawArticle(
                        id=meta.get("article_id", "unknown"),
                        article_number=meta.get("num", "N/A"),
                        content=raw_content,
                        context=meta.get("category", "Code de la Route"),
                        url=meta.get("url"),
                    )
                    clean_results.append(RetrievalResult(article=article, score=distance))
                except Exception as e:
                    logger.warning(f"Failed to parse result {i}: {e}")

        return clean_results
