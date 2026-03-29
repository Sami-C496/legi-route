import logging
from typing import Iterator

from src.config import settings
from src.models import RetrievalResult
from src.providers import LLMProvider

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Tu es **LégiRoute**, un assistant juridique spécialisé dans le Code de la Route français.

        RÈGLES STRICTES :
        1. **NE TE PRÉSENTE JAMAIS** spontanément. Réponds directement à la question posée. Tu ne dois mentionner ton nom ou ta nature que si l'utilisateur te demande explicitement qui tu es.
        2. **CITATIONS OBLIGATOIRES** : Pour toute question juridique, cite les articles de loi (ex: "Selon l'article R413-17...").
        3. **PAS D'INVENTION** : Ne cite que les articles fournis dans le contexte. Si aucun article pertinent n'est disponible, dis-le honnêtement.
        4. **STYLE** : Réponds de manière concise et structurée. Va droit au but.
        5. **HORS SUJET** : Si la question n'est pas liée au Code de la Route, réponds naturellement sans inventer de citations juridiques.
        """


class TrafficGenerator:

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def _format_context(self, results: list[RetrievalResult]) -> str:
        if not results:
            return "AUCUN ARTICLE TROUVÉ."

        parts = []
        for i, res in enumerate(results, 1):
            art = res.article
            parts.append(
                f"--- SOURCE {i} : Article {art.article_number} ---\n"
                f"Chemin : {art.context}\n"
                f"Contenu : {art.content}\n"
                f"URL : {art.full_url}"
            )
        return "\n".join(parts)

    def generate_stream(self, query: str, results: list[RetrievalResult]) -> Iterator[str]:
        context = self._format_context(results)
        prompt = f"CONTEXTE JURIDIQUE :\n{context}\n\nQUESTION :\n{query}\n\nRÉPONSE :"
        yield from self.provider.generate_stream(prompt, SYSTEM_PROMPT)

    def generate(self, query: str, results: list[RetrievalResult]) -> str:
        """Non-streaming generation. Collects all chunks into a single string."""
        return "".join(self.generate_stream(query, results))
