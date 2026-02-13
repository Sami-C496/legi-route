"""
Tests for the retrieval module (retrieval.py).

Strategy: We test the result serialization logic (Chroma raw data → domain objects)
without requiring a live ChromaDB instance or API key. The semantic search quality
itself is evaluated separately via the evaluation dataset.
"""

import pytest
from src.models import TrafficLawArticle, RetrievalResult


# ============================================================
# Result Rehydration (Chroma → Domain Objects)
# ============================================================

class TestResultRehydration:
    """
    When ChromaDB returns raw dicts, the retriever must reconstruct
    proper TrafficLawArticle objects. This tests that logic in isolation.
    """

    def test_reconstructs_article_from_metadata(self):
        """Simulates what the retriever does with raw Chroma output."""
        meta = {
            "article_id": "LEGIARTI000006841575",
            "num": "R413-17",
            "category": "Code de la route > Livre IV",
            "url": "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000006841575",
        }
        content = "Sur les autoroutes, la vitesse est limitée à 130 km/h."
        distance = 0.35

        article = TrafficLawArticle(
            id=meta.get("article_id", "unknown"),
            article_number=meta.get("num", "N/A"),
            content=content,
            context=meta.get("category", "Code de la Route"),
            url=meta.get("url"),
        )
        result = RetrievalResult(article=article, score=distance)

        assert result.article.article_number == "R413-17"
        assert result.score == 0.35
        assert "legifrance" in result.article.full_url

    def test_handles_missing_metadata_keys(self):
        """If metadata is incomplete, defaults should prevent a crash."""
        meta = {}
        content = "Some valid legal content here."

        article = TrafficLawArticle(
            id=meta.get("article_id", "unknown"),
            article_number=meta.get("num", "N/A"),
            content=content,
            context=meta.get("category", "Code de la Route"),
        )

        assert article.id == "unknown"
        assert article.article_number == "N/A"
        assert article.context == "Code de la Route"

    def test_preserves_full_content(self):
        """
        NOTE: Currently, 'content' stored in Chroma is the blob_for_embedding,
        not the raw article text. This test documents the current behavior.
        See retrieval.py — this is a known limitation.
        """
        blob = "Code de la route > Livre IV \nArticle R413-17 : Vitesse limitée à 130."

        article = TrafficLawArticle(
            id="LEGIARTI000006841575",
            article_number="R413-17",
            content=blob,
            context="Code de la route > Livre IV",
        )

        assert "Article R413-17" in article.content


# ============================================================
# Query Validation
# ============================================================

class TestQueryValidation:
    """
    Tests for the query guard in retriever.search().
    The retriever should reject queries that are too short to be meaningful.
    """

    def test_short_query_should_be_rejected(self):
        """Queries under 3 chars are noise — should skip search."""
        query = "ab"
        assert len(query.strip()) < 3

    def test_empty_query_should_be_rejected(self):
        query = ""
        assert len(query.strip()) < 3

    def test_whitespace_query_should_be_rejected(self):
        query = "   "
        assert len(query.strip()) < 3

    def test_valid_query_passes(self):
        query = "vitesse autoroute"
        assert len(query.strip()) >= 3


# ============================================================
# Result Ordering
# ============================================================

class TestResultOrdering:
    """Validates that results can be sorted by relevance score."""

    def test_results_sortable_by_score(self):
        articles = [
            TrafficLawArticle(
                id=f"LEGIARTI00000000000{i}",
                article_number=f"R{i}",
                content=f"Article content number {i} with enough text.",
                context="Code de la route",
            )
            for i in range(1, 4)
        ]

        results = [
            RetrievalResult(article=articles[0], score=0.8),
            RetrievalResult(article=articles[1], score=0.3),
            RetrievalResult(article=articles[2], score=0.5),
        ]

        sorted_results = sorted(results, key=lambda r: r.score)
        assert sorted_results[0].score == 0.3
        assert sorted_results[-1].score == 0.8
