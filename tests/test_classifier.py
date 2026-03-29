"""
Tests for the LLM-based intent classifier (classifier.py).

Strategy: The LLM provider is fully mocked — no API calls.
We test the classification logic, structured output parsing,
and fallback behavior.

"""

import json
import pytest
from unittest.mock import MagicMock

from src.providers import LLMProvider
from src.classifier import IntentClassifier, Intent, CLASSIFICATION_PROMPT, INTENT_SCHEMA


# --- Fixtures ---

@pytest.fixture
def mock_provider():
    return MagicMock(spec=LLMProvider)


@pytest.fixture
def mock_classifier(mock_provider):
    return IntentClassifier(mock_provider)


def _set_intent(provider, intent_value: str):
    """Helper: makes the mocked provider return a structured intent."""
    provider.classify_intent.return_value = {"intent": intent_value}


# ============================================================
# Intent Classification
# ============================================================

class TestIntentClassification:
    """
    Tests that the classifier correctly routes queries based on
    structured LLM output. The LLM is mocked — we test routing logic.
    """

    def test_legal_query_routed_correctly(self, mock_provider, mock_classifier):
        _set_intent(mock_provider, "LEGAL_QUERY")
        result = mock_classifier.classify("Quelle est la vitesse en agglomération ?")
        assert result == Intent.LEGAL_QUERY

    def test_chitchat_routed_correctly(self, mock_provider, mock_classifier):
        _set_intent(mock_provider, "CHITCHAT")
        result = mock_classifier.classify("Bonjour !")
        assert result == Intent.CHITCHAT

    def test_off_topic_routed_correctly(self, mock_provider, mock_classifier):
        _set_intent(mock_provider, "OFF_TOPIC")
        result = mock_classifier.classify("Quelle est la recette de la ratatouille ?")
        assert result == Intent.OFF_TOPIC

    def test_empty_query_returns_chitchat(self, mock_classifier):
        """Empty input is treated as chitchat — no LLM call needed."""
        result = mock_classifier.classify("")
        assert result == Intent.CHITCHAT

    def test_single_char_returns_chitchat(self, mock_classifier):
        result = mock_classifier.classify("a")
        assert result == Intent.CHITCHAT


# ============================================================
# Structured Output Parsing
# ============================================================

class TestStructuredOutputParsing:
    """
    The provider returns a dict like {"intent": "LEGAL_QUERY"}.
    These tests verify the parser handles valid and edge-case responses.
    """

    def test_parses_valid_response(self, mock_provider, mock_classifier):
        _set_intent(mock_provider, "LEGAL_QUERY")
        result = mock_classifier.classify("Alcool au volant ?")
        assert result == Intent.LEGAL_QUERY

    def test_missing_intent_key_defaults_to_legal(self, mock_provider, mock_classifier):
        """Provider returns a dict without the 'intent' key."""
        mock_provider.classify_intent.return_value = {"category": "LEGAL_QUERY"}
        result = mock_classifier.classify("Vitesse autoroute ?")
        assert result == Intent.LEGAL_QUERY

    def test_invalid_enum_value_defaults_to_legal(self, mock_provider, mock_classifier):
        """Dict has 'intent' but the value is not in the enum."""
        mock_provider.classify_intent.return_value = {"intent": "UNKNOWN_CATEGORY"}
        result = mock_classifier.classify("Question quelconque")
        assert result == Intent.LEGAL_QUERY

    def test_empty_dict_defaults_to_legal(self, mock_provider, mock_classifier):
        mock_provider.classify_intent.return_value = {}
        result = mock_classifier.classify("Feux rouges ?")
        assert result == Intent.LEGAL_QUERY


# ============================================================
# Fallback Behavior (Safety)
# ============================================================

class TestFallbackBehavior:
    """
    Critical: if the classifier fails or the API errors out,
    we MUST default to LEGAL_QUERY. It's better to run an
    unnecessary search than to skip a real legal question.
    """

    def test_api_error_defaults_to_legal(self, mock_provider, mock_classifier):
        """Network failure or API error should not block legal queries."""
        mock_provider.classify_intent.side_effect = Exception("API timeout")
        result = mock_classifier.classify("Sanctions excès de vitesse ?")
        assert result == Intent.LEGAL_QUERY

    def test_provider_returns_none_defaults_to_legal(self, mock_provider, mock_classifier):
        mock_provider.classify_intent.return_value = None
        result = mock_classifier.classify("Stationnement interdit ?")
        assert result == Intent.LEGAL_QUERY


# ============================================================
# Schema & Prompt Validation
# ============================================================

class TestSchemaAndPrompt:
    """Validates the structured output schema and classification prompt."""

    def test_schema_has_intent_field(self):
        assert "intent" in INTENT_SCHEMA["properties"]

    def test_schema_enum_matches_intent_class(self):
        schema_values = set(INTENT_SCHEMA["properties"]["intent"]["enum"])
        enum_values = {i.value for i in Intent}
        assert schema_values == enum_values

    def test_schema_intent_is_required(self):
        assert "intent" in INTENT_SCHEMA["required"]

    def test_prompt_contains_all_intents(self):
        assert "LEGAL_QUERY" in CLASSIFICATION_PROMPT
        assert "CHITCHAT" in CLASSIFICATION_PROMPT
        assert "OFF_TOPIC" in CLASSIFICATION_PROMPT

    def test_prompt_mentions_code_de_la_route(self):
        assert "code de la route" in CLASSIFICATION_PROMPT.lower()
