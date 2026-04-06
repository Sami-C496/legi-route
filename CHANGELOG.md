# Changelog

All notable changes to this project are documented here.

## [1.0.0] - 2026-04-06

First stable release.

### Features
- **RAG pipeline**: intent classification, semantic retrieval (Pinecone), grounded generation (Gemini 2.5 Flash) with mandatory article citations
- **Intent routing**: LLM classifier (Gemini Flash Lite, structured JSON output) routes queries into LEGAL_QUERY, CHITCHAT, or OFF_TOPIC before touching the vector DB
- **Conversational memory**: follow-up questions are rewritten into standalone queries before retrieval. Last 3 turns of history are passed to the generator for continuity
- **Asymmetric embeddings**: Gemini embedding API with RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY task types. 3072-dim vectors, cosine similarity
- **Full legal hierarchy**: each article is embedded with its hierarchy path (Code de la route > Partie reglementaire > Livre IV > ...) for disambiguation
- **Relevance threshold**: hard cutoff (score > 0.5) prevents the generator from answering when the knowledge base genuinely does not cover the question
- **Daily database sync**: GitHub Actions workflow authenticates with Legifrance PISTE API (OAuth 2.0), fetches updated articles, upserts new ones and deletes repealed ones from Pinecone
- **Streamlit web UI**: streaming responses, expandable source citations with Legifrance links
- **CLI entry point**: interactive command-line interface via `make cli`
- **RAGAS evaluation**: Faithfulness (0.935) and Context Precision (0.940) on a 61-question dataset

### Architecture
- Provider abstraction (`LLMProvider` ABC) decouples all modules from the Gemini SDK
- One article = one chunk. Median article is 103 words, 95% under 500 words. No sub-article chunking needed
- Pinecone serverless (AWS us-east-1) replaced ChromaDB for stateless cloud deployment
- Ingestion rate-limited with exponential backoff (tenacity) for Gemini free tier

### Infrastructure
- CI workflow runs tests on push to main and on PRs
- Daily LEGI update workflow (3 AM UTC) with `[skip ci]` to avoid triggering CI on data commits
- Docker image (python:3.13-slim) deployed on Render
- `.dockerignore` excludes .git, tests, eval, raw data from the image
- Makefile for all common operations (`make test`, `make run`, `make release`, etc.)

### Data
- 1163 articles from the Code de la Route (DILA LEGI XML dataset)
- Only articles with ETAT=VIGUEUR are indexed
- Processed JSON committed to repo for reproducibility without the raw XML dump
