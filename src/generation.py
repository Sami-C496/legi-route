"""
Generation Module for Traffic Law RAG.

Responsible for:
1. Constructing a context-aware prompt from retrieval results.
2. Interfacing with the Gemini LLM for answer generation.
3. Handling response streaming for optimal UX.
"""

import logging
from typing import List, Iterator
from google import genai
from google.genai import types

from src.config import settings
from src.models import RetrievalResult

# --- Logging ---
logger = logging.getLogger(__name__)


class TrafficGenerator:
    """Orchestrates the LLM generation process."""

    def __init__(self, model_name: str = None):
        if not settings.GOOGLE_API_KEY:
            raise ValueError("Critical: GOOGLE_API_KEY is missing.")

        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.model_name = model_name or settings.GENERATION_MODEL
        logger.info(f"TrafficGenerator initialized with model: {self.model_name}")

    def _build_system_prompt(self) -> str:
        return """Tu es **LégiRoute**, un assistant juridique indépendant spécialisé dans le Code de la Route français.

        DIRECTIVES DE PERSONNALITÉ :
        1. **IDENTITÉ** : Si l'on te demande qui tu es, réponds simplement : "Je suis LégiRoute, un assistant d'intelligence artificielle conçu pour vous aider à naviguer dans le Code de la Route." (Ne mentionne jamais tes technologies sous-jacentes ni qui t'a entraîné).
        2. **STYLE** : Professionnel, concis, serviable.

        RÈGLES DE RÉPONSE JURIDIQUE :
        1. **CITATIONS** : Pour toute question sur la loi, tu DOIS citer les articles (ex: "Article R413-17").
        2. **HONNÊTETÉ** : Ne dis jamais "Le document dit que...". Affirme les faits : "Selon l'article..."
        3. **LIMITES** : Si la question n'est pas juridique (ex: "Qui es-tu ?", "Bonjour"), réponds naturellement SANS inventer de citations de loi.
        """

    def _format_context(self, results: List[RetrievalResult]) -> str:
        """Transforms structured RetrievalResults into a text block for the LLM."""
        if not results:
            return "AUCUN ARTICLE TROUVÉ."

        context_str = ""
        for i, res in enumerate(results, 1):
            art = res.article
            context_str += f"\n--- SOURCE {i} : Article {art.article_number} ---\n"
            context_str += f"Chemin : {art.context}\n"
            context_str += f"Contenu : {art.content}\n"
            context_str += f"URL : {art.full_url}\n"

        return context_str

    def generate_stream(self, query: str, results: List[RetrievalResult]) -> Iterator[str]:
        """
        Generates a streaming response based on the query and retrieved documents.

        Yields:
            String chunks of the generated response.
        """
        system_instruction = self._build_system_prompt()
        context_block = self._format_context(results)

        full_prompt = f"""
CONTEXTE JURIDIQUE :
{context_block}

QUESTION DE L'UTILISATEUR :
{query}

RÉPONSE :
"""

        try:
            response_stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=settings.GENERATION_TEMPERATURE,
                    max_output_tokens=settings.GENERATION_MAX_TOKENS
                )
            )

            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            yield "Une erreur est survenue lors de la génération de la réponse."
