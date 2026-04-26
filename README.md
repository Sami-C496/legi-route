# LégiRoute

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/Dependency-Poetry-blueviolet)](https://python-poetry.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React_18-61DAFB)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Bundler-Vite_5-646CFF)](https://vitejs.dev/)
[![Pinecone](https://img.shields.io/badge/Vector_DB-Pinecone-1C3C3C)](https://www.pinecone.io/)
[![Gemini Embedding](https://img.shields.io/badge/Embedding-Gemini_001-512BD4)](https://ai.google.dev/gemini-api/docs/models/gemini#text-embedding)
[![Gemini](https://img.shields.io/badge/Model-Gemini_2.5_Flash-4285F4)](https://deepmind.google/technologies/gemini/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

RAG system over the French Highway Code (*Code de la Route*). Ask a question in natural language, the pipeline retrieves the relevant legal articles from Légifrance and generates a cited, grounded answer.

> **Live demo**: [legiroute.onrender.com](https://legiroute.onrender.com)

- Answers grounded strictly on retrieved articles, each cited by number with a direct link to the official Légifrance source
- Only articles currently in force (`ETAT=VIGUEUR`) are indexed. Citing a repealed law is a critical failure, not an edge case
- Conversational: follow-up questions ("et en agglomération ?") are rewritten into standalone queries before retrieval, so context carries through without polluting the vector search
- Intent routing classifies every query before hitting the vector DB. Chitchat gets a direct response, off-topic is refused, only legal queries trigger retrieval
- Database updated daily via the Légifrance PISTE API (OAuth 2.0), new articles are upserted and repealed ones deleted automatically

## Results

Evaluated on a **61-question dataset** across 18 categories, scored by an LLM judge (Gemini).

| Metric | Score | What it measures |
|--------|-------|-----------------|
| **Faithfulness** | **0.935** | Answer does not go beyond the retrieved sources |
| **Context Precision** | **0.940** | Retrieved articles are relevant to the question |

These two metrics isolate where failures come from:
- High Context Precision + low Faithfulness: retrieval works, the LLM is hallucinating, fix the prompt
- Low Context Precision + high Faithfulness: wrong articles retrieved, the LLM is faithful to them, fix the retriever

Other RAGAS metrics (Answer Relevance, Answer Correctness) were not included because they require annotated ground truth answers, which are not available for this dataset. Faithfulness and Context Precision only need the retrieved context, so they work without reference answers.

## Architecture

```
                        +-----------------+
          user query -->| IntentClassifier|---> OFF_TOPIC: polite refusal
       + history        | (Gemini Flash)  |---> CHITCHAT: direct response
                        +-------+---------+
                                |
                           LEGAL_QUERY
                                |
                                v
                        +-----------------+
                        |  QueryRewriter  |  rewrites follow-ups into
                        | (Gemini Flash)  |  standalone search queries
                        +-------+---------+
                                |
                         rewritten query
                                |
                                v
                        +-----------------+
                        | TrafficRetriever|  Pinecone (cosine similarity)
                        | (embed + search)|  3072-dim Gemini embeddings
                        +-------+---------+
                                |
                          top-k articles
                          filtered by relevance threshold
                                |
                                v
                        +-----------------+
                        | TrafficGenerator|  Gemini Flash (streaming)
                        | (prompt + cite) |  grounded on retrieved articles
                        |                 |  + conversation history (3 turns)
                        +-----------------+
                                |
                                v
                        cited answer + Légifrance URLs
```

Pipeline modules are decoupled via an `LLMProvider` interface: the classifier, retriever, and generator each receive a provider instance at construction time. Swapping to another LLM backend means implementing three methods: `embed`, `generate_stream`, `classify_intent`.

Conversation history (last 3 messages) is maintained per session. Follow-up questions ("et en agglomération ?") are rewritten into standalone queries before hitting Pinecone, so retrieval is always grounded on a complete question. The original question and full history are then passed to the generator.

## Data Source

Articles are sourced from the [DILA LEGI XML dataset](https://www.data.gouv.fr/fr/datasets/legi-codes-lois-et-reglements-consolides/) (official open data), not scraped from a website or extracted from a PDF. The XML format gives structured metadata (hierarchy, article status, dates) that scraping would lose.

**1163 articles** indexed, covering speed limits, alcohol limits, equipment, penalties, signage, EDPM regulations, and more.

Only articles with `ETAT=VIGUEUR` are indexed. The filter is binary: if an article is not currently in force, it does not go in.

The Code de la Route frequently references external codes (Code pénal, Code des assurances). We don't cross-index those, only the citation text is kept. The LLM can still see and quote the reference without us having to ingest all 73 French legal codes.

The database is kept current by a GitHub Actions workflow running daily at 3 AM UTC (see [Automatic database updates](#automatic-database-updates)).

## Architecture Decisions

### Intent classification before retrieval

The first implementation used a keyword + length heuristic (`len < 30` + keyword list). False positive rate was too high: "Peut-on klaxonner ?" (a legal question, 19 chars) kept getting classified as chitchat.

The current approach uses a lightweight LLM classifier (Gemini Flash Lite, ~50 tokens) that routes each query before touching the vector DB. This avoids wasting embedding calls on greetings or off-topic questions, and the system can respond naturally to chitchat without fabricating legal citations.

Three intents:
- `LEGAL_QUERY` -> query rewriting -> full RAG pipeline (embed -> search -> generate with history)
- `CHITCHAT` -> direct LLM response (no retrieval)
- `OFF_TOPIC` -> polite refusal

**Safety default**: any classification failure falls back to `LEGAL_QUERY`. Better to run unnecessary retrieval than to block a real question.

**Trade-off**: adds one LLM call per query (~100ms). Worth it. Embedding a greeting and returning fabricated legal citations is worse on both latency and quality.

### Structured output for classification

The classifier uses Gemini's `response_schema` with `response_mime_type="application/json"` to force a valid JSON enum (`LEGAL_QUERY | CHITCHAT | OFF_TOPIC`). No regex parsing, no retry logic. A dedicated test verifies the Python `Intent` enum and the JSON schema enum stay in sync.

### Conversational query rewriting

Follow-up questions ("et en agglomération ?", "quelle est la sanction ?") lose their meaning without context. Injecting conversation history directly into the retrieval query pollutes the embedding space with irrelevant turns.

Instead, a dedicated rewriting step runs before retrieval: the LLM receives the conversation history and either rewrites the question into a self-contained query, or returns it unchanged if it already is. The rewriter runs at `temperature=0` and is explicitly instructed not to touch standalone questions. If the current turn has nothing to do with previous ones, retrieval proceeds on the original question with no modification.

### Embedding strategy

**Asymmetric embeddings**: Gemini's embedding API supports `RETRIEVAL_DOCUMENT` and `RETRIEVAL_QUERY` task types natively. Using the right task type for each role improves recall when short questions ("vitesse autoroute ?") need to match long legal articles.

**Full context hierarchy in the embedding blob**: each article is embedded as `"{context_path}\nArticle {number} : {content}"` rather than raw content alone. This lets the embedding capture where the article sits in the legal hierarchy. "Livre IV > Titre I > Vitesses" disambiguates articles that all mention "50 km/h" but in different legal contexts.

**Content/blob separation**: the index stores raw content separately from the embedding input. The LLM receives clean, structured sources (article number, hierarchy path, content, Légifrance URL as distinct fields) rather than a monolithic blob.

### Chunking: one article = one chunk

Median article length in the Code de la Route is 103 words; 95% of articles are under 500 words. Sub-article chunking would break legal atomicity: the LLM needs to cite "R413-17", not "R413-17 chunk 2 of 4". Only 9 articles out of 1163 exceed 1000 words, so single-chunk indexing is the right call here.

If extended to other legal codes (e.g., Code pénal, which the Code de la Route frequently cites), a different chunking strategy will be needed.

### Pinecone with cosine similarity + relevance threshold

ChromaDB was the initial vector store. It requires a local filesystem, which breaks in ephemeral cloud containers. Pinecone is a managed index accessible over HTTP with no local state, which is what makes deployment on Render possible.

Cosine similarity scores range from 0 to 1. A hard threshold (`score > 0.5`) filters out results that are technically the best available but still a poor match. This stops the generator from producing an answer when the knowledge base genuinely doesn't cover the question.

**Ingestion rate limiting**: batches of 5 articles with sleep between batches, tuned for Gemini's free tier. Exponential backoff (tenacity) handles transient 429s. 400 errors (invalid input, token overflow) are not retried since they would fail on every attempt.

### Provider abstraction

All LLM calls go through an `LLMProvider` ABC (`embed`, `generate_stream`, `classify_intent`). No module imports `google.genai` directly. In tests, you mock the interface, not the SDK. Adding a new provider means subclassing `LLMProvider`; nothing else changes.

### Generation

The system prompt enforces: no self-introduction, mandatory article citations, no fabrication beyond provided sources, concise structured responses. Each source block contains article number, hierarchy path, content, and Légifrance URL as explicit citation targets.

Max tokens was raised from 1000 to 2048 after seeing truncated responses on complex multi-article questions.

### Schema design

| Field | Role |
|-------|------|
| `id` | Primary key, direct traceability to source XML |
| `article_number` | Citation key used by the LLM ("Selon l'article R413-17...") |
| `content` | Raw text, separated from the embedding blob |
| `context` | Hierarchy path: `Code de la route > Partie réglementaire > Livre IV > ...` |
| `blob_for_embedding` | Computed field, context + number + content concatenated at ingestion |
| `full_url` | Computed field, Légifrance URL reconstructed from the article ID |

## Tests

```bash
make test
```

72 tests covering classification routing, structured output parsing, fallback behavior, context formatting, model validation, XML parsing, and retrieval logic. All tests run without live API calls, mock or pure computation only.

## Automatic database updates

A GitHub Actions workflow runs daily at 3 AM UTC. It authenticates with the [Légifrance PISTE API](https://api.piste.gouv.fr/dila/legifrance/lf-engine-app) via OAuth 2.0 client credentials, fetches the current table of contents for the Code de la Route, and retrieves only the articles not yet in the processed dataset. New articles are embedded and upserted to Pinecone; repealed ones are deleted.

A `latest_update.md` file is committed daily with the list of added and removed articles.

## Setup

```bash
# Install dependencies
make install

# Set your API keys
cp .env.example .env
# Fill in GOOGLE_API_KEY, PINECONE_API_KEY, LEGIFRANCE_CLIENT_ID, LEGIFRANCE_CLIENT_SECRET

# Update articles from Légifrance API (or use the committed JSON directly)
make download

# Build the vector index
make index

# Run the FastAPI backend (http://localhost:8000)
make run-api

# In a second terminal, run the React frontend (http://localhost:5173)
make dev-web

# Or the legacy Streamlit app
make run

# Run CLI
make cli

# Run evaluation
make eval
```

### Local backend with Ollama

The `LLMProvider` abstraction supports a fully local backend through [Ollama](https://ollama.com/), using open-source models with no API key. Useful for development, offline work, or avoiding paid API quotas.

The embedding dimension differs between backends (Gemini: 3072, nomic-embed-text: 768), so Ollama requires its own Pinecone index. The runtime backend is selected per process via the `PROVIDER` env var; switching is non-destructive as long as you keep separate index names.

```bash
# 1. Install Ollama and pull the models
curl -fsSL https://ollama.com/install.sh | sh
make ollama-pull   # qwen2.5:7b-instruct + nomic-embed-text

# 2. Point the app at Ollama and use a dedicated index
# In .env:
#   PROVIDER=ollama
#   PINECONE_INDEX_NAME=traffic-law-ollama

# 3. Build the Ollama-backed index (one-off, embeds all articles locally)
make index-ollama

# 4. Run the app against Ollama
make run-ollama
```

Notes:
- Generation latency is hardware-bound. On a CPU-only machine the 7b model is slow; smaller variants (`qwen2.5:3b-instruct`, `qwen2.5:1.5b-instruct`) trade quality for speed.
- The deployed Render instance runs on CPU and stays on Gemini; Ollama is intended for local use.
- To switch back to Gemini, set `PROVIDER=gemini` and `PINECONE_INDEX_NAME=traffic-law-v1`.

### Hosted backend with Groq

[Groq](https://groq.com/) serves open-source models (Llama 3, Qwen, etc.) via an OpenAI-compatible API with a generous free tier and fast inference (LPU).

Groq has no embeddings API, so `GroqProvider` keeps embeddings on Gemini.

```bash
# In .env:
#   PROVIDER=groq
#   GROQ_API_KEY=...
#   GOOGLE_API_KEY=...   # still needed for embeddings only

# 2. Run the app
make run-groq
```

Models used:
- Generation: `llama-3.3-70b-versatile`
- Classification: `llama-3.1-8b-instant`

---

## Web stack: FastAPI + React

The original Streamlit UI is still available (`make run`), but the production frontend is now a React single-page app served by Vite, talking to a FastAPI backend that wraps the existing `RAG` pipeline.

### Why move off Streamlit

- **Streaming**: Streamlit renders token-by-token by re-running the script, which scales poorly and forces server-side state. FastAPI streams tokens over Server-Sent Events (SSE), the React client appends them to the DOM directly. No re-renders, no flicker.
- **Decoupling**: the RAG pipeline is now reachable from any HTTP client (CLI, mobile, another service). Streamlit locked it to a single Python process.
- **UI control**: editorial typography (Fraunces serif for answers, Inter sans for UI, JetBrains Mono for article numbers), explicit citation cards, and a header status indicator that ships with the design system.

### Backend: `src/api/`

```
src/api/
  main.py           # FastAPI app + CORS + router mounting
  deps.py           # @lru_cache get_rag() — singleton RAG instance, overridable in tests
  schemas.py        # Pydantic v2 request / response models
  sse.py            # SSE event formatter
  routes/
    health.py       # GET /api/health
    chat.py         # POST /api/chat → text/event-stream
```

`POST /api/chat` returns an SSE stream. Events emitted in order:

| Event | When | Payload |
|-------|------|---------|
| `intent` | always, first | `{ "intent": "LEGAL_QUERY" \| "CHITCHAT" \| "OFF_TOPIC" }` |
| `sources` | LEGAL_QUERY only | `[{ "article_number", "url", "excerpt", "score" }]` |
| `token` | repeated | `{ "text": "..." }` |
| `done` | terminal | `{}` |
| `error` | on failure | `{ "message": "..." }` |

The handler short-circuits OFF_TOPIC with a polite refusal message, runs query rewriting + retrieval for LEGAL_QUERY, and falls back to a direct LLM response for CHITCHAT — same behaviour as the Streamlit app, exposed over HTTP.

Run it standalone:

```bash
make run-api          # uvicorn src.api.main:app --reload --port 8000
curl http://localhost:8000/api/health
```

Six integration tests cover the full event flow per intent, the empty-prompt 422, and the retrieval-failure path. They use `TestClient` with `app.dependency_overrides[get_rag]` to inject a mocked RAG, so no live API keys are required.

### Frontend: `frontend/`

Vite + React 18 + TypeScript + Tailwind. Editorial palette (paper, ink, marine, signal red/green) defined in `tailwind.config.ts`.

```
frontend/
  src/
    App.tsx                 # Main shell: header / messages / input
    components/
      Wordmark.tsx          # Logo + brand text
      StatusDot.tsx         # 3-dot traffic light (red while loading, green when idle)
      Message.tsx           # Assistant answer in Fraunces serif + cited SourceCards
      SourceCard.tsx        # Article number + excerpt + Légifrance link
      ChatInput.tsx         # Textarea, Enter to send, Shift+Enter for newline
    lib/
      sse.ts                # Browser SSE parser (async generator over `Response.body`)
      api.ts                # streamChat(prompt, history, callbacks)
    styles/globals.css      # Tailwind base + font imports
  vite.config.ts            # /api proxied to localhost:8000
```

Run it (with the API already running on port 8000):

```bash
make dev-web          # cd frontend && npm install && npm run dev
# → http://localhost:5173
```

Production build:

```bash
make build-web        # cd frontend && npm run build → frontend/dist/
```

### Talking to the API from a notebook

`docs/api_walkthrough.ipynb` walks through the SSE protocol with a minimal `httpx`-based parser, calls each endpoint live, and demos in-process testing via `TestClient`.

---

## Roadmap

**Evaluation**
- Build an annotated ground truth dataset to unlock the remaining RAGAS metrics (Answer Relevance, Answer Correctness). The current eval dataset has questions but no reference answers, which limits what can be measured without a judge model
- Swap the LLM judge for a different model than the generator to eliminate self-bias in the RAGAS scores (current setup uses Gemini for both)

**Multi-provider support**
- A local Ollama backend is already wired in (open-source models, no API key). See [Local backend with Ollama](#local-backend-with-ollama)
- A hosted Groq backend is available for fast, free inference on open-source models. See [Hosted backend with Groq](#hosted-backend-with-groq)
- Benchmark retrieval quality and answer faithfulness across providers on the same eval dataset

**Retrieval**
- Cross-code indexing: the Code de la Route frequently references Code pénal and Code des assurances. Index those codes as a separate namespace and retrieve across namespaces when the query references penalties or insurance
- Re-ranking: add a cross-encoder pass after Pinecone retrieval to reorder the top-k results before generation. Especially useful for multi-article questions where the dense retriever returns partially relevant results

**Data**
- HTML tag stripping in the ingestion pipeline. A small number of articles contain inline HTML that is currently passed raw to the LLM
- Structured metadata filtering: allow pre-filtering by legal book or title before vector search (e.g., "only search in Livre IV" for speed-related queries)

---

**Author**: Sami Contesenne - [sami.contesenne496@gmail.com](mailto:sami.contesenne496@gmail.com)
