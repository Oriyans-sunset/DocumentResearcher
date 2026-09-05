# DocumentResearcher

A **Retrieval-Augmented Generation** pipeline that ingests PDFs, embeds them into a vector store, and answers questions grounded in that context; orchestrated as a **durable, event-driven workflow** instead of a synchronous script.

## Key Details

- **Durable execution with [Inngest](https://www.inngest.com/)** — ingestion and Q&A aren't plain function calls, they're event-triggered functions (`rag/ingest_pdf`, `rag/query_pdf_ai`) built from independently retryable steps (`context.step.run`). Each step is checkpointed, so a failure in embedding or upsert, reruns just that step, not the whole pipeline.
- **AI inference as a first-class workflow step** — the answer-generation call goes through `context.step.ai.infer(...)`, letting Inngest treat the Gemini call itself as a durable, observable, retryable step in the run rather than an opaque network call.
- **Fully decoupled frontend** — the Streamlit UI never talks to the LLM or vector DB directly. It fires events at Inngest and polls the run API for results, so ingestion and querying are async, horizontally scalable, and swappable behind any frontend.
- **Vector search with QdrantDB** — [Qdrant](https://qdrant.tech/) backs retrieval with cosine similarity over 3072-dim `gemini-embedding-001` vectors, with UUIDv5 IDs so re-ingesting a source is idempotent.

## Architecture
<img width="6908" height="1156" alt="image" src="https://github.com/user-attachments/assets/f5b003da-6921-42a8-9470-30fa9b973f57" />

(p.s. - diagrams made with draw.io)


## Tech stack

| Layer | Tool |
|---|---|
| Workflow orchestration | Inngest (durable-step functions & event triggers) |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Embeddings & LLM | Google Gemini (`gemini-embedding-002`, `gemini-2.5-flash-lite`) |
| Vector store | Qdrant |
| PDF parsing / chunking | LlamaIndex (`PDFReader`, `SentenceSplitter`) |
| Data models | Pydantic |
| Containerization | Docker Compose (Qdrant, API, Inngest dev server, Streamlit UI) |
| Deployment | ECS Fargate, ECR, S3 |

## Running it

### Option A — Docker Compose (recommended)

The whole stack — Qdrant, the FastAPI/Inngest API, the Inngest dev server, and the Streamlit UI — is containerized. The `api` and `ui` services build from the same `Dockerfile` and just run different commands.

```bash
# create a .env with GEMINI_API_KEY (and any other required vars)
docker compose up --build
```

- API: `http://localhost:8000`
- Inngest dev server UI: `http://localhost:8288`
- Streamlit UI: `http://localhost:8501`

Services start in dependency order via healthchecks (`qdrant` → `api` → `inngest`/`ui`). Qdrant data persists to `./qdrant_storage` (bind mount); uploaded PDFs persist to a shared `uploads` named volume used by both `api` and `ui`.

### Option B — Run locally without Docker

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
