"""
Tests for the LLM-based intent classifier (classifier.py).

Strategy: The LLM client is fully mocked — no API calls.
We test the classification logic, structured output parsing,
and fallback behavior.

"""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.classifier import IntentClassifier, Intent, CLASSIFICATION_PROMPT, INTENT_SCHEMA


# --- Fixtures ---

@pytest.fixture
def mock_classifier():
    """Creates an IntentClassifier with a mocked LLM client."""
    with patch("src.classifier.settings") as mock_settings:
        mock_settings.GOOGLE_API_KEY = "test-key-fake"
        mock_settings.CLASSIFIER_MODEL = "models/gemini-2.5-flash-lite"
        with patch("src.classifier.genai.Client"):
            classifier = IntentClassifier()
    return classifier


def _mock_llm_response(classifier, intent_value: str):
    """Helper: makes the mocked LLM return a structured JSON response."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({"intent": intent_value})
    classifier.client.models.generate_content.return_value = mock_response


# ============================================================
# Intent Classification
# ============================================================

class TestIntentClassification:
    """
    Tests that the classifier correctly routes queries based on
    structured LLM output. The LLM is mocked — we test routing logic.
    """

    def test_legal_query_routed_correctly(self, mock_classifier):
        _mock_llm_response(mock_classifier, "LEGAL_QUERY")
        result = mock_classifier.classify("Quelle est la vitesse en agglomération ?")
        assert result == Intent.LEGAL_QUERY

    def test_chitchat_routed_correctly(self, mock_classifier):
        _mock_llm_response(mock_classifier, "CHITCHAT")
        result = mock_classifier.classify("Bonjour !")
        assert result == Intent.CHITCHAT

    def test_off_topic_routed_correctly(self, mock_classifier):
        _mock_llm_response(mock_classifier, "OFF_TOPIC")
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
    The model returns JSON like {"intent": "LEGAL_QUERY"}.
    These tests verify the parser handles valid and edge-case responses.
    """

    def test_parses_valid_json(self, mock_classifier):
        _mock_llm_response(mock_classifier, "LEGAL_QUERY")
        result = mock_classifier.classify("Alcool au volant ?")
        assert result == Intent.LEGAL_QUERY

    def test_invalid_json_defaults_to_legal(self, mock_classifier):
        """If structured output fails, raw text is not valid JSON."""
        mock_response = MagicMock()
        mock_response.text = "not json at all"
        mock_classifier.client.models.generate_content.return_value = mock_response
        result = mock_classifier.classify("Permis de conduire")
        assert result == Intent.LEGAL_QUERY

    def test_missing_intent_key_defaults_to_legal(self, mock_classifier):
        """JSON is valid but doesn't contain the 'intent' key."""
        mock_response = MagicMock()
        mock_response.text = json.dumps({"category": "LEGAL_QUERY"})
        mock_classifier.client.models.generate_content.return_value = mock_response
        result = mock_classifier.classify("Vitesse autoroute ?")
        assert result == Intent.LEGAL_QUERY

    def test_invalid_enum_value_defaults_to_legal(self, mock_classifier):
        """JSON has 'intent' but the value is not in the enum."""
        mock_response = MagicMock()
        mock_response.text = json.dumps({"intent": "UNKNOWN_CATEGORY"})
        mock_classifier.client.models.generate_content.return_value = mock_response
        result = mock_classifier.classify("Question quelconque")
        assert result == Intent.LEGAL_QUERY

    def test_empty_json_defaults_to_legal(self, mock_classifier):
        mock_response = MagicMock()
        mock_response.text = "{}"
        mock_classifier.client.models.generate_content.return_value = mock_response
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

    def test_api_error_defaults_to_legal(self, mock_classifier):
        """Network failure or API error should not block legal queries."""
        mock_classifier.client.models.generate_content.side_effect = Exception("API timeout")
        result = mock_classifier.classify("Sanctions excès de vitesse ?")
        assert result == Intent.LEGAL_QUERY

    def test_api_returns_none_defaults_to_legal(self, mock_classifier):
        mock_response = MagicMock()
        mock_response.text = None
        mock_classifier.client.models.generate_content.return_value = mock_response
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
