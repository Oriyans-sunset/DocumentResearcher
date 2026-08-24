# DocumentResearcher

A **Retrieval-Augmented Generation** pipeline that ingests PDFs, embeds them into a vector store, and answers questions grounded in that context; orchestrated as a **durable, event-driven workflow** instead of a synchronous script.

## Key Details

- **Durable execution with [Inngest](https://www.inngest.com/)** — ingestion and Q&A aren't plain function calls, they're event-triggered functions (`rag/ingest_pdf`, `rag/query_pdf_ai`) built from independently retryable steps (`context.step.run`). Each step is checkpointed, so a failure in embedding or upsert, reruns just that step, not the whole pipeline.
- **AI inference as a first-class workflow step** — the answer-generation call goes through `context.step.ai.infer(...)`, letting Inngest treat the Gemini call itself as a durable, observable, retryable step in the run rather than an opaque network call.
- **Fully decoupled frontend** — the Streamlit UI never talks to the LLM or vector DB directly. It fires events at Inngest and polls the run API for results, so ingestion and querying are async, horizontally scalable, and swappable behind any frontend.
- **Vector search with QdrantDB** — [Qdrant](https://qdrant.tech/) backs retrieval with cosine similarity over 3072-dim `gemini-embedding-001` vectors, with UUIDv5 IDs so re-ingesting a source is idempotent.

## Architecture

```
Streamlit UI  ──event──▶  Inngest  ──▶  FastAPI-hosted functions
                                         │
                     ┌───────────────────┼────────────────────┐
                     ▼                                        ▼
          rag/ingest_pdf                              rag/query_pdf_ai
 load PDF → chunk → embed → upsert     embed question → search Qdrant  → step.ai.infer → answer
 (LlamaIndex)      (Gemini) (Qdrant)                                        (Gemini)
```

## Tech stack

| Layer | Tool |
|---|---|
| Workflow orchestration | Inngest (durable-step functions & event triggers) |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Embeddings & LLM | Google Gemini (`gemini-embedding-002`, `gemini-2.5-flash-lite`) |
| Vector store | Qdrant |
| PDF parsing / chunking | LlamaIndex (`PDFReader`, `SentenceSplitter`) |
| Data contracts | Pydantic |

## Running it

```bash
uv sync
uv pip install -r requirements.txt
uv run uvicorn ragproductionapp.main:app --reload   # FastAPI + Inngest functions
uv run streamlit run src/ragproductionapp/streamlist_app.py   # UI
```

Requires a Qdrant instance via
```
docker run -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
```
an Inngest dev server, and a `GEMINI_API_KEY` in your environment.
