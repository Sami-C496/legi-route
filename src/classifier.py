"""
Intent Classifier for the LégiRoute RAG system.

Routes incoming user queries into one of three intents:
- LEGAL_QUERY  → full RAG pipeline (embed → search → generate)
- CHITCHAT     → direct LLM response (no retrieval needed)
- OFF_TOPIC    → polite refusal

Uses Gemini's structured output (response_schema) to guarantee
a valid enum response — no fuzzy text parsing needed.

Replaces the V1 keyword + length heuristic which had known false positives.

Safety: on any failure, defaults to LEGAL_QUERY.
"""

import json
import logging
from enum import Enum
from google import genai
from google.genai import types

from src.config import settings

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    """Possible user intents for the routing layer."""
    LEGAL_QUERY = "LEGAL_QUERY"
    CHITCHAT = "CHITCHAT"
    OFF_TOPIC = "OFF_TOPIC"


CLASSIFICATION_PROMPT = """Tu es un classificateur d'intention pour un assistant juridique spécialisé dans le Code de la Route français.

Classifie le message de l'utilisateur dans EXACTEMENT une de ces catégories :

LEGAL_QUERY : Question sur le droit routier, le code de la route, les infractions, les sanctions, les limitations de vitesse, le permis de conduire, l'alcool au volant, le stationnement, la signalisation, les équipements obligatoires, etc.
CHITCHAT : Salutation, remerciement, question sur l'identité du bot, bavardage sans rapport juridique.
OFF_TOPIC : Question sérieuse mais hors du domaine du code de la route (cuisine, sport, politique, médecine, etc.)."""


# Schema for structured output — forces the model to return a valid enum value
INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["LEGAL_QUERY", "CHITCHAT", "OFF_TOPIC"],
        }
    },
    "required": ["intent"],
}


class IntentClassifier:
    """
    Lightweight LLM-based intent classifier.

    Uses Gemini Flash Lite with structured output for fast, cheap,
    and type-safe classification.
    Falls back to LEGAL_QUERY on any failure to avoid blocking real questions.
    """

    def __init__(self, model_name: str = None):
        if not settings.GOOGLE_API_KEY:
            raise ValueError("Critical: GOOGLE_API_KEY is missing.")

        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.model_name = model_name or settings.CLASSIFIER_MODEL
        logger.info(f"IntentClassifier initialized with model: {self.model_name}")

    def classify(self, query: str) -> Intent:
        """
        Classifies a user query into an Intent.

        Args:
            query: The raw user input.

        Returns:
            Intent enum value. Defaults to LEGAL_QUERY on failure.
        """
        if not query or len(query.strip()) < 2:
            return Intent.CHITCHAT

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=f"Message utilisateur : {query}",
                config=types.GenerateContentConfig(
                    system_instruction=CLASSIFICATION_PROMPT,
                    temperature=0.0,
                    max_output_tokens=50,
                    response_mime_type="application/json",
                    response_schema=INTENT_SCHEMA,
                )
            )

            return self._parse_response(response.text)

        except Exception as e:
            logger.warning(f"Classification failed for '{query}': {e}. Defaulting to LEGAL_QUERY.")
            return Intent.LEGAL_QUERY

    def _parse_response(self, raw: str) -> Intent:
        """
        Parses the structured JSON response into an Intent enum.

        With response_schema enforced, the model returns valid JSON like:
            {"intent": "LEGAL_QUERY"}
        If parsing fails, we log a warning and default to LEGAL_QUERY.
        """
        try:
            data = json.loads(raw)
            return Intent(data["intent"])
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse structured response '{raw}': {e}. Defaulting to LEGAL_QUERY.")
            return Intent.LEGAL_QUERY
