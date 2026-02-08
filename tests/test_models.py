"""
Tests for the Pydantic data contracts (models.py).

These tests validate the core data layer that every other module depends on.
A failure here means corrupted data could silently pollute the vector database.
"""

import pytest
from pydantic import ValidationError
from src.models import TrafficLawArticle, RetrievalResult


# --- Fixtures ---

@pytest.fixture
def sample_article_data():
    """Minimal valid article data, reusable across tests."""
    return {
        "id": "LEGIARTI000006841575",
        "article_number": "R413-17",
        "content": "Sur les autoroutes, la vitesse des véhicules est limitée à 130 km/h.",
        "context": "Code de la route > Partie réglementaire > Livre IV > Titre I",
    }


@pytest.fixture
def sample_article(sample_article_data):
    return TrafficLawArticle(**sample_article_data)


# ============================================================
# TrafficLawArticle — Happy Path
# ============================================================

class TestTrafficLawArticleCreation:
    """Validates that well-formed data produces correct objects."""

    def test_creates_valid_article(self, sample_article, sample_article_data):
        assert sample_article.id == sample_article_data["id"]
        assert sample_article.article_number == "R413-17"
        assert "130 km/h" in sample_article.content

    def test_url_field_is_optional(self, sample_article_data):
        """URL is optional metadata — should default to None."""
        article = TrafficLawArticle(**sample_article_data)
        assert article.url is None

    def test_url_field_accepts_value(self, sample_article_data):
        sample_article_data["url"] = "https://example.com"
        article = TrafficLawArticle(**sample_article_data)
        assert article.url == "https://example.com"


# ============================================================
# TrafficLawArticle — Computed Fields
# ============================================================

class TestComputedFields:
    """
    Computed fields encapsulate business logic.
    blob_for_embedding determines what the vector model 'sees'.
    full_url determines the Légifrance link shown to users.
    Any regression here silently degrades retrieval quality or breaks citations.
    """

    def test_blob_contains_context(self, sample_article):
        blob = sample_article.blob_for_embedding
        assert sample_article.context in blob

    def test_blob_contains_article_number(self, sample_article):
        blob = sample_article.blob_for_embedding
        assert "Article R413-17" in blob

    def test_blob_contains_content(self, sample_article):
        blob = sample_article.blob_for_embedding
        assert sample_article.content in blob

    def test_blob_format_is_deterministic(self, sample_article):
        """The blob must be reproducible — same input = same vector."""
        expected = (
            f"{sample_article.context} \n"
            f"Article {sample_article.article_number} : {sample_article.content}"
        )
        assert sample_article.blob_for_embedding == expected

    def test_full_url_uses_legifrance_pattern(self, sample_article):
        assert sample_article.full_url == (
            "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000006841575"
        )

    def test_full_url_changes_with_id(self, sample_article_data):
        sample_article_data["id"] = "LEGIARTI999999999999"
        article = TrafficLawArticle(**sample_article_data)
        assert "LEGIARTI999999999999" in article.full_url


# ============================================================
# TrafficLawArticle — Validation (Rejection Cases)
# ============================================================

class TestArticleValidation:
    """
    The data contract must REJECT malformed input at ingestion time.
    Letting bad data through silently corrupts the vector DB.
    """

    def test_rejects_empty_content(self, sample_article_data):
        sample_article_data["content"] = ""
        with pytest.raises(ValidationError, match="empty or too short"):
            TrafficLawArticle(**sample_article_data)

    def test_rejects_whitespace_only_content(self, sample_article_data):
        sample_article_data["content"] = "    \n\t  "
        with pytest.raises(ValidationError, match="empty or too short"):
            TrafficLawArticle(**sample_article_data)

    def test_rejects_content_under_5_chars(self, sample_article_data):
        sample_article_data["content"] = "abc"
        with pytest.raises(ValidationError, match="empty or too short"):
            TrafficLawArticle(**sample_article_data)

    def test_accepts_content_at_minimum_length(self, sample_article_data):
        sample_article_data["content"] = "abcde"
        article = TrafficLawArticle(**sample_article_data)
        assert article.content == "abcde"

    def test_rejects_missing_id(self, sample_article_data):
        del sample_article_data["id"]
        with pytest.raises(ValidationError):
            TrafficLawArticle(**sample_article_data)

    def test_rejects_missing_article_number(self, sample_article_data):
        del sample_article_data["article_number"]
        with pytest.raises(ValidationError):
            TrafficLawArticle(**sample_article_data)


# ============================================================
# RetrievalResult
# ============================================================

class TestRetrievalResult:
    """Validates the search result wrapper."""

    def test_creates_valid_result(self, sample_article):
        result = RetrievalResult(article=sample_article, score=0.42)
        assert result.score == 0.42
        assert result.article.article_number == "R413-17"

    def test_str_representation(self, sample_article):
        result = RetrievalResult(article=sample_article, score=0.1234)
        assert "[0.1234] R413-17" in str(result)

    def test_rejects_missing_score(self, sample_article):
        with pytest.raises(ValidationError):
            RetrievalResult(article=sample_article)

    def test_preserves_article_computed_fields(self, sample_article):
        """Ensure nesting doesn't break computed fields on the inner article."""
        result = RetrievalResult(article=sample_article, score=0.5)
        assert "legifrance.gouv.fr" in result.article.full_url
        assert result.article.blob_for_embedding is not None
