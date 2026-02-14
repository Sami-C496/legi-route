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
    Filters retrieval results by semantic distance.
    Lower score = better match (ChromaDB L2 distance).
    """
    return [s for s in scores if s < threshold]


# ============================================================
# Relevance Filtering
# ============================================================

class TestRelevanceFiltering:
    """
    ChromaDB returns L2 distances. Lower = better.
    The threshold filters out results that are too semantically distant.
    """

    def test_keeps_good_scores(self):
        scores = [0.3, 0.5, 0.8]
        assert filter_by_relevance(scores) == [0.3, 0.5, 0.8]

    def test_filters_bad_scores(self):
        scores = [0.3, 1.5, 2.0]
        assert filter_by_relevance(scores) == [0.3]

    def test_boundary_score_excluded(self):
        """Score exactly at threshold should be excluded (strict <)."""
        scores = [1.1]
        assert filter_by_relevance(scores) == []

    def test_empty_input(self):
        assert filter_by_relevance([]) == []

    def test_all_irrelevant(self):
        scores = [1.5, 2.0, 3.0]
        assert filter_by_relevance(scores) == []

    def test_custom_threshold(self):
        scores = [0.3, 0.5, 0.8]
        assert filter_by_relevance(scores, threshold=0.4) == [0.3]
