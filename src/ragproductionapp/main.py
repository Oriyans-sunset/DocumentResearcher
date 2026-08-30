import logging

from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
import datetime
from inngest.experimental.ai import gemini
from llama_index.core.output_parsers import pydantic
from .data_loader import load_and_chunk_pdf, embed_text
from .vector_db import QdrantStorage
from .custom_types import RAGChunkAndSrc, RAGQueryResult, RAGSearchResult, RAGUpsetResult

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    serializer=inngest.PydanticSerializer(),
)

@inngest_client.create_function(
    fn_id="RAG: Inngest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(context: inngest.Context):
    def _load(context: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = context.event.data["pdf_path"]
        source_id = context.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsetResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vectors = embed_text(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vectors, payloads)
        return RAGUpsetResult(ingested=len(chunks))

    chunks_and_src = await context.step.run("load-and_chunk", lambda: _load(context), output_type=RAGChunkAndSrc)
    ingested = await context.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsetResult)
    return ingested.model_dump()

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(context: inngest.Context):
    def _search(question, top_k = 5):
        query_vec = embed_text([question])[0]
        search_result = QdrantStorage().search(query_vector=query_vec, top_k=top_k)
        return RAGSearchResult(context=search_result["context"], sources=search_result["source"])

    question = context.event.data["question"]
    top_k = int(context.event.data["top_k"])

    found = await context.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)
    context_block = "\n\n".join(f"- {c}" for c in found.context)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context: \n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    adapter = gemini.Adapter(
        auth_key=os.environ["GEMINI_API_KEY"],
        model="gemini-3.1-flash-lite"
    )


    res = await context.step.ai.infer(
        "generate-answer",
        adapter=adapter,
        body={
            "systemInstruction": {
                "parts": [{"text": "You answer questions based only on the context provided."}]
            },
            "contents": [
                {"role": "user", "parts": [{"text": user_content}]}
            ],
            "generationConfig": {
                "maxOutputTokens": 1024,
                "temperature": 0.2
            }
        }
    )

    answer = res["candidates"][0]["content"]["parts"][0]["text"].strip()
    return {"answer": answer, "sources": found.sources, "num_context": len(found.context)}

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])