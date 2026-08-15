# RAG Production App

A production-shaped **Retrieval-Augmented Generation** pipeline that ingests PDFs, embeds them into a vector store, and answers questions grounded in that content — orchestrated as **durable, event-driven workflows** instead of fragile synchronous scripts.

## Why this is interesting

- **Durable execution with [Inngest](https://www.inngest.com/)** — ingestion and Q&A aren't plain function calls, they're event-triggered functions (`rag/ingest_pdf`, `rag/query_pdf_ai`) built from independently retryable steps (`context.step.run`). Each step is checkpointed, so a transient failure in embedding or upsert reruns just that step, not the whole pipeline.
- **AI inference as a first-class workflow step** — the answer-generation call goes through `context.step.ai.infer(...)`, letting Inngest treat the Gemini call itself as a durable, observable, retryable step in the run graph rather than an opaque network call.
- **Fully decoupled frontend** — the Streamlit UI never talks to the LLM or vector DB directly. It fires events at Inngest and polls the run API for results, so ingestion and querying are async, horizontally scalable, and swappable behind any frontend.
- **Typed data contracts throughout** — every step boundary (chunking, embedding, upsert, search) passes strongly-typed Pydantic models (`RAGChunkAndSrc`, `RAGSearchResult`, `RAGUpsetResult`), and Inngest is configured with `PydanticSerializer` so those types survive the event bus intact.
- **Real vector search, not a toy** — [Qdrant](https://qdrant.tech/) backs retrieval with cosine similarity over 3072-dim `gemini-embedding-001` vectors, with deterministic UUIDv5 point IDs so re-ingesting a source is idempotent.
- **Clean separation of concerns** — `data_loader` (LlamaIndex PDF parsing + sentence-aware chunking), `vector_db` (Qdrant storage), `main` (FastAPI + Inngest functions), and `streamlist_app` (Streamlit UI) each own one responsibility.

## Architecture

```
Streamlit UI  ──event──▶  Inngest  ──▶  FastAPI-hosted functions
                                         │
                     ┌───────────────────┼────────────────────┐
                     ▼                                        ▼
          rag/ingest_pdf                              rag/query_pdf_ai
     load PDF → chunk → embed → upsert           embed question → search Qdrant
     (LlamaIndex)         (Gemini)  (Qdrant)      → step.ai.infer (Gemini) → answer
```

## Tech stack

| Layer | Tool |
|---|---|
| Workflow orchestration | Inngest (durable functions, event triggers, AI steps) |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Embeddings & LLM | Google Gemini (`gemini-embedding-001`, `gemini-3.1-flash-lite`) |
| Vector store | Qdrant |
| PDF parsing / chunking | LlamaIndex (`PDFReader`, `SentenceSplitter`) |
| Data contracts | Pydantic |

## Running it

```bash
uv sync
uv run uvicorn ragproductionapp.main:app --reload   # FastAPI + Inngest functions
uv run streamlit run src/ragproductionapp/streamlist_app.py   # UI
```

Requires a Qdrant instance (e.g. `docker run -p 6333:6333 qdrant/qdrant`), an Inngest dev server, and a `GEMINI_API_KEY` in your environment.
