"""Integration tests for the FastAPI chat endpoint (SSE)."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.deps import get_rag
from src.api.main import app
from src.classifier import Intent
from src.models import RetrievalResult, TrafficLawArticle


def _fake_article(number: str = "R412-6-1") -> TrafficLawArticle:
    return TrafficLawArticle(
        id="LEGIARTI000000000000",
        article_number=number,
        content="Le fait, pour tout conducteur, d'utiliser un téléphone tenu en main est puni d'une amende.",
        context="Code de la route > Partie réglementaire",
    )


def _make_rag(intent: Intent, sources: list[RetrievalResult], tokens: list[str]) -> MagicMock:
    rag = MagicMock()
    rag.classifier.classify.return_value = intent
    rag.rewrite_query.return_value = "rewritten"
    rag.retriever.search.return_value = sources
    rag.generator.generate_stream.return_value = iter(tokens)
    return rag


def _parse_sse(body: str) -> list[tuple[str, str]]:
    """Return list of (event, data) tuples from an SSE response body."""
    events = []
    for block in body.strip().split("\n\n"):
        if not block:
            continue
        event = data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = line[len("data: "):]
        if event:
            events.append((event, data))
    return events


@pytest.fixture
def client():
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_chat_legal_query_streams_intent_sources_tokens_done(client, monkeypatch):
    monkeypatch.setattr("src.api.routes.chat.settings.RELEVANCE_THRESHOLD", 0.3)
    sources = [RetrievalResult(article=_fake_article(), score=0.91)]
    rag = _make_rag(Intent.LEGAL_QUERY, sources, ["D'après ", "l'article R412-6-1, ", "..."])
    app.dependency_overrides[get_rag] = lambda: rag

    response = client.post(
        "/api/chat",
        json={"prompt": "Sanction téléphone au volant ?", "history": [], "k": 3},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    names = [e for e, _ in events]
    assert names[0] == "intent"
    assert "sources" in names
    assert names.count("token") == 3
    assert names[-1] == "done"


def test_chat_off_topic_short_circuits(client):
    rag = _make_rag(Intent.OFF_TOPIC, [], [])
    app.dependency_overrides[get_rag] = lambda: rag

    response = client.post("/api/chat", json={"prompt": "Quelle est la météo ?"})
    assert response.status_code == 200

    events = _parse_sse(response.text)
    names = [e for e, _ in events]
    assert names == ["intent", "token", "done"]
    rag.retriever.search.assert_not_called()
    rag.generator.generate_stream.assert_not_called()


def test_chat_chitchat_skips_retrieval_but_generates(client):
    rag = _make_rag(Intent.CHITCHAT, [], ["Bonjour ", "!"])
    app.dependency_overrides[get_rag] = lambda: rag

    response = client.post("/api/chat", json={"prompt": "Salut"})
    assert response.status_code == 200

    events = _parse_sse(response.text)
    names = [e for e, _ in events]
    assert "sources" not in names
    assert names.count("token") == 2
    assert names[-1] == "done"
    rag.retriever.search.assert_not_called()


def test_chat_retrieval_failure_returns_unavailable_message(client):
    rag = MagicMock()
    rag.classifier.classify.return_value = Intent.LEGAL_QUERY
    rag.rewrite_query.return_value = "x"
    rag.retriever.search.side_effect = RuntimeError("pinecone down")
    app.dependency_overrides[get_rag] = lambda: rag

    response = client.post("/api/chat", json={"prompt": "Vitesse autoroute ?"})
    assert response.status_code == 200

    events = _parse_sse(response.text)
    names = [e for e, _ in events]
    assert names == ["intent", "token", "done"]
    rag.generator.generate_stream.assert_not_called()


def test_chat_rejects_empty_prompt(client):
    rag = _make_rag(Intent.LEGAL_QUERY, [], [])
    app.dependency_overrides[get_rag] = lambda: rag

    response = client.post("/api/chat", json={"prompt": ""})
    assert response.status_code == 422


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
