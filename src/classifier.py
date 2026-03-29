import json
import logging
from enum import Enum

from src.providers import LLMProvider

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    LEGAL_QUERY = "LEGAL_QUERY"
    CHITCHAT = "CHITCHAT"
    OFF_TOPIC = "OFF_TOPIC"


CLASSIFICATION_PROMPT = """Tu es un classificateur d'intention pour un assistant juridique spécialisé dans le Code de la Route français.

Classifie le message de l'utilisateur dans EXACTEMENT une de ces catégories :

LEGAL_QUERY : Question sur le droit routier, le code de la route, les infractions, les sanctions, les limitations de vitesse, le permis de conduire, l'alcool au volant, le stationnement, la signalisation, les équipements obligatoires, etc.
CHITCHAT : Salutation, remerciement, question sur l'identité du bot, bavardage sans rapport juridique.
OFF_TOPIC : Question sérieuse mais hors du domaine du code de la route (cuisine, sport, politique, médecine, etc.).

Réponds UNIQUEMENT avec un objet JSON: {"intent": "LEGAL_QUERY"} ou {"intent": "CHITCHAT"} ou {"intent": "OFF_TOPIC"}"""

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

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def classify(self, query: str) -> Intent:
        if not query or len(query.strip()) < 2:
            return Intent.CHITCHAT

        try:
            data = self.provider.classify_intent(query, CLASSIFICATION_PROMPT)
            return Intent(data["intent"])
        except Exception as e:
            logger.warning(f"Classification failed: {e}. Defaulting to LEGAL_QUERY.")
            return Intent.LEGAL_QUERY
