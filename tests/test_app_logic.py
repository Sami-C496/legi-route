"""
Tests for application-level logic (app.py).

The Streamlit UI itself is not unit-testable, but the business logic
embedded in it IS. These tests validate the chitchat detection heuristic
and the semantic score filtering.

NOTE: If these functions are later extracted into a dedicated module
(recommended refactor), these tests remain valid without changes.
"""

import pytest
from src.config import settings


# --- Extracted Logic (mirrors app.py) ---

def is_chitchat(prompt: str) -> bool:
    """
    Heuristic to detect non-legal conversational messages.
    Returns True if the message is likely chitchat (skip retrieval).
    """
    return (
        any(kw in prompt.lower() for kw in settings.CHITCHAT_KEYWORDS)
        and len(prompt) < settings.MAX_CHITCHAT_LENGTH
    )


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
# Chitchat Detection
# ============================================================

class TestChitchatDetection:
    """
    The chitchat detector saves API calls by short-circuiting
    the retrieval + generation pipeline for non-legal queries.
    False negatives (legal query classified as chitchat) are worse
    than false positives (chitchat going through retrieval).
    """

    # --- True Positives: Should be detected as chitchat ---

    def test_detects_simple_greeting(self):
        assert is_chitchat("Bonjour") is True

    def test_detects_hello(self):
        assert is_chitchat("Hello !") is True

    def test_detects_thanks(self):
        assert is_chitchat("Merci beaucoup") is True

    def test_detects_identity_question(self):
        assert is_chitchat("Qui es-tu ?") is True

    def test_case_insensitive(self):
        assert is_chitchat("BONJOUR") is True

    # --- True Negatives: Legal queries must NOT be classified as chitchat ---

    def test_legal_question_not_chitchat(self):
        assert is_chitchat("Quelle est la vitesse maximale sur autoroute ?") is False

    def test_long_greeting_with_legal_question(self):
        """A greeting followed by a real question should go through retrieval."""
        prompt = "Bonjour, quelle est la sanction pour un téléphone au volant ?"
        assert is_chitchat(prompt) is False  # len > MAX_CHITCHAT_LENGTH saves us

    def test_short_legal_query_not_chitchat(self):
        """Short but legal query — no keyword match."""
        assert is_chitchat("Alcool au volant ?") is False

    # --- Edge Cases ---

    def test_empty_string(self):
        assert is_chitchat("") is False

    def test_keyword_in_long_legal_context(self):
        """'merci' appears but the message is clearly legal."""
        prompt = "Merci de me dire quelle est la limite de vitesse en agglomération selon le code"
        assert is_chitchat(prompt) is False


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
