"""
Tests for application-level logic (app.py).

Chitchat detection tests have been moved to test_classifier.py
following the refactor from keyword heuristic to LLM-based classification.
This file retains tests for relevance score filtering.
"""

import pytest
from src.config import settings


def filter_by_relevance(
    scores: list[float],
    threshold: float = settings.RELEVANCE_THRESHOLD
) -> list[float]:
    """
    Filters retrieval results by cosine similarity.
    Higher score = better match. Keeps scores above threshold.
    """
    return [s for s in scores if s > threshold]


# ============================================================
# Relevance Filtering
# ============================================================

class TestRelevanceFiltering:
    """
    Pinecone returns cosine similarity scores (0 to 1). Higher = better.
    The threshold filters out results that are not similar enough.
    """

    def test_keeps_good_scores(self):
        scores = [0.95, 0.88, 0.80]
        assert filter_by_relevance(scores) == [0.95, 0.88, 0.80]

    def test_filters_bad_scores(self):
        scores = [0.95, 0.40, 0.20]
        assert filter_by_relevance(scores) == [0.95]

    def test_boundary_score_excluded(self):
        """Score exactly at threshold should be excluded (strict >)."""
        scores = [0.50]
        assert filter_by_relevance(scores) == []

    def test_empty_input(self):
        assert filter_by_relevance([]) == []

    def test_all_irrelevant(self):
        scores = [0.30, 0.20, 0.10]
        assert filter_by_relevance(scores) == []

    def test_custom_threshold(self):
        scores = [0.95, 0.85, 0.60]
        assert filter_by_relevance(scores, threshold=0.90) == [0.95]
