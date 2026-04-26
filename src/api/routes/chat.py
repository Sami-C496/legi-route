import logging
import time
from typing import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.api.deps import get_rag
from src.api.schemas import ChatRequest
from src.api.sse import sse
from src.classifier import Intent
from src.config import settings
from src.rag import RAG

logger = logging.getLogger(__name__)
router = APIRouter()

_UNAVAILABLE_MSG = (
    "Le service est momentanément surchargé. Veuillez réessayer dans quelques instants."
)
_OFF_TOPIC_MSG = (
    "Je suis spécialisé dans le Code de la Route. Je ne peux pas répondre à cette question."
)


def _excerpt(text: str, length: int = 250) -> str:
    return text[:length] + "..." if len(text) > length else text


def _stream_chat(req: ChatRequest, rag: RAG) -> Iterator[str]:
    history = [m.model_dump() for m in req.history]
    prompt = req.prompt
    t0 = time.monotonic()

    try:
        intent = rag.classifier.classify(prompt)
    except Exception as e:
        logger.error("Classification unavailable: %s", e)
        intent = Intent.LEGAL_QUERY
    yield sse("intent", {"intent": intent.value})

    if intent == Intent.OFF_TOPIC:
        yield sse("token", {"text": _OFF_TOPIC_MSG})
        yield sse("done", {"elapsed": round(time.monotonic() - t0, 3)})
        return

    sources = []
    if intent == Intent.LEGAL_QUERY:
        try:
            search_query = rag.rewrite_query(prompt, history)
            results = rag.retriever.search(search_query, k=req.k)
            sources = [r for r in results if r.score > settings.RELEVANCE_THRESHOLD]
        except Exception as e:
            logger.error("Retrieval unavailable: %s", e)
            yield sse("token", {"text": _UNAVAILABLE_MSG})
            yield sse("done", {"elapsed": round(time.monotonic() - t0, 3)})
            return

        yield sse(
            "sources",
            [
                {
                    "article_number": r.article.article_number,
                    "url": r.article.full_url,
                    "excerpt": _excerpt(r.article.content),
                    "score": round(r.score, 4),
                }
                for r in sources
            ],
        )

    try:
        for chunk in rag.generator.generate_stream(prompt, sources, history=history):
            yield sse("token", {"text": chunk})
    except Exception as e:
        logger.error("Generation error: %s", e)
        yield sse("error", {"message": _UNAVAILABLE_MSG})
        yield sse("done", {"elapsed": round(time.monotonic() - t0, 3)})
        return

    elapsed = round(time.monotonic() - t0, 3)
    logger.info(
        "chat | intent=%s | sources=%d | elapsed=%.2fs", intent.value, len(sources), elapsed
    )
    yield sse("done", {"elapsed": elapsed})


@router.post("/chat")
def chat(req: ChatRequest, rag: RAG = Depends(get_rag)) -> StreamingResponse:
    return StreamingResponse(
        _stream_chat(req, rag),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
