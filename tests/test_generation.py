"""
Tests for the generation module (generation.py).

Strategy: We test the prompt construction and context formatting logic
WITHOUT calling the real Gemini API. The LLM is an external dependency —
we validate everything we control (prompt quality, context assembly).
"""

import pytest
from unittest.mock import patch, MagicMock

from src.models import TrafficLawArticle, RetrievalResult
from src.generation import TrafficGenerator


# --- Fixtures ---

@pytest.fixture
def mock_generator():
    """
    Creates a TrafficGenerator without a real API key.
    We patch both settings and the genai Client to avoid any network call.
    """
    with patch("src.generation.settings") as mock_settings:
        mock_settings.GOOGLE_API_KEY = "test-key-fake"
        mock_settings.GENERATION_MODEL = "models/gemini-2.5-flash"
        mock_settings.GENERATION_TEMPERATURE = 0.0
        mock_settings.GENERATION_MAX_TOKENS = 1000
        with patch("src.generation.genai.Client"):
            generator = TrafficGenerator()
    return generator


@pytest.fixture
def sample_results():
    """Two RetrievalResults simulating a real search output."""
    art1 = TrafficLawArticle(
        id="LEGIARTI000006841575",
        article_number="R413-17",
        content="Sur les autoroutes, la vitesse est limitée à 130 km/h.",
        context="Code de la route > Partie réglementaire > Livre IV",
    )
    art2 = TrafficLawArticle(
        id="LEGIARTI000006842000",
        article_number="R413-2",
        content="La vitesse est limitée à 50 km/h en agglomération.",
        context="Code de la route > Partie réglementaire > Livre IV",
    )
    return [
        RetrievalResult(article=art1, score=0.35),
        RetrievalResult(article=art2, score=0.48),
    ]


# ============================================================
# System Prompt
# ============================================================

class TestSystemPrompt:
    """
    The system prompt defines the persona and the grounding rules.
    Any change here directly affects answer quality.
    """

    def test_prompt_defines_identity(self, mock_generator):
        prompt = mock_generator._build_system_prompt()
        assert "LégiRoute" in prompt

    def test_prompt_enforces_citations(self, mock_generator):
        prompt = mock_generator._build_system_prompt()
        assert "citer" in prompt.lower() or "citation" in prompt.lower()

    def test_prompt_sets_limits_on_non_legal_queries(self, mock_generator):
        """Chitchat questions should not trigger fake citations."""
        prompt = mock_generator._build_system_prompt()
        assert "juridique" in prompt.lower() or "loi" in prompt.lower()


# ============================================================
# Context Formatting
# ============================================================

class TestContextFormatting:
    """
    _format_context transforms RetrievalResults into a text block
    that gets injected into the LLM prompt. The format directly
    affects whether the model can locate and cite the right article.
    """

    def test_empty_results_returns_no_article_message(self, mock_generator):
        context = mock_generator._format_context([])
        assert "AUCUN ARTICLE" in context

    def test_includes_article_numbers(self, mock_generator, sample_results):
        context = mock_generator._format_context(sample_results)
        assert "R413-17" in context
        assert "R413-2" in context

    def test_includes_article_content(self, mock_generator, sample_results):
        context = mock_generator._format_context(sample_results)
        assert "130 km/h" in context
        assert "50 km/h" in context

    def test_includes_legifrance_urls(self, mock_generator, sample_results):
        context = mock_generator._format_context(sample_results)
        assert "legifrance.gouv.fr" in context

    def test_includes_context_hierarchy(self, mock_generator, sample_results):
        context = mock_generator._format_context(sample_results)
        assert "Livre IV" in context

    def test_sources_are_numbered(self, mock_generator, sample_results):
        """Numbered sources help the LLM reference specific articles."""
        context = mock_generator._format_context(sample_results)
        assert "SOURCE 1" in context
        assert "SOURCE 2" in context

    def test_single_result_formatting(self, mock_generator, sample_results):
        """Edge case: only one result should still format correctly."""
        context = mock_generator._format_context(sample_results[:1])
        assert "SOURCE 1" in context
        assert "SOURCE 2" not in context
