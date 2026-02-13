"""
Retrieval Module for Traffic Law RAG.

Handles semantic search operations:
1. Converting user queries into vector embeddings (RETRIEVAL_QUERY task type).
2. Querying the ChromaDB vector store.
3. Rehydrating results into typed TrafficLawArticle objects.
"""

import os
import logging
import chromadb
from typing import List
from google import genai
from google.genai import types

from src.config import settings
from src.models import RetrievalResult, TrafficLawArticle

# --- Logging ---
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class TrafficRetriever:
    """Facade for the Vector Database and Embedding Service."""

    def __init__(self, db_path: str = None):
        """
        Args:
            db_path: Filesystem path to ChromaDB. Defaults to settings.CHROMA_DB_PATH.
        """
        api_key = settings.GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing from environment variables.")

        self.client = genai.Client(api_key=api_key)

        resolved_path = db_path or str(settings.CHROMA_DB_PATH)

        if not os.path.exists(resolved_path):
            logger.warning(f"Database path '{resolved_path}' does not exist yet.")

        try:
            self.chroma_client = chromadb.PersistentClient(path=resolved_path)
            self.collection = self.chroma_client.get_collection(name=settings.COLLECTION_NAME)
            logger.info(f"TrafficRetriever initialized. Collection: '{settings.COLLECTION_NAME}'")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise RuntimeError("Database connection failed. Did you run the ingestion script?") from e

    def _embed_query(self, query: str) -> List[float]:
        """
        Generates embedding for a query using RETRIEVAL_QUERY task type.
        This asymmetric approach improves search quality vs symmetric cosine.
        """
        try:
            response = self.client.models.embed_content(
                model=settings.EMBEDDING_MODEL,
                contents=query,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY"
                )
            )
            return response.embeddings[0].values
        except Exception as e:
            logger.error(f"Embedding API failed for query: '{query}'. Error: {e}")
            raise

    def search(self, query: str, k: int = None) -> List[RetrievalResult]:
        """
        Performs a semantic search against the traffic law database.

        Args:
            query: The user's natural language question.
            k: Number of results to retrieve. Defaults to settings.DEFAULT_TOP_K.

        Returns:
            A list of RetrievalResult objects containing the article and distance score.
        """
        k = k or settings.DEFAULT_TOP_K

        if not query or len(query.strip()) < 3:
            logger.warning("Query too short, skipping search.")
            return []

        # 1. Vectorize the Question
        try:
            query_vector = self._embed_query(query)
        except Exception:
            return []

        # 2. Vector Search
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []

        # 3. Rehydration: Raw Chroma Data → Domain Objects
        clean_results = []

        if results and results['documents'] and results['documents'][0]:
            num_results = len(results['documents'][0])

            for i in range(num_results):
                try:
                    meta = results['metadatas'][0][i]
                    content = results['documents'][0][i]
                    distance = results['distances'][0][i] if results['distances'] else 0.0

                    article = TrafficLawArticle(
                        id=meta.get('article_id', 'unknown'),
                        article_number=meta.get('num', 'N/A'),
                        content=content,
                        context=meta.get('category', 'Code de la Route'),
                        url=meta.get('url')
                    )

                    result_obj = RetrievalResult(article=article, score=distance)
                    clean_results.append(result_obj)

                except Exception as e:
                    logger.warning(f"Failed to parse result index {i}: {e}")
                    continue

        return clean_results


# --- Standalone Test ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        retriever = TrafficRetriever()
        test_query = "Quelle est la vitesse sur autoroute par temps de pluie ?"
        print(f"\nTesting search with: '{test_query}'")

        results = retriever.search(test_query, k=3)

        for res in results:
            print(f"Ref: {res.article.article_number} | Score: {res.score:.4f}")
            print(f"Excerpt: {res.article.content[:100]}...\n")

    except Exception as e:
        print(f"Test failed: {e}")
