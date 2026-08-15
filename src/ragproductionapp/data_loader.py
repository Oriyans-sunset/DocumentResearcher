import os
from urllib import response

from google import genai
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()
EMBEDDING_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    # TODO: convert path string to pure path
    docs = PDFReader().load_data(file=path) # return a list of Documents. 1 Document = 1 page of pdf
    pages = [d.text for d in docs if getattr(d, "text", None)]

    chunks = []
    for page in pages:
        chunks.extend(splitter.split_text(page))
    return chunks

def embed_text(texts: list[str]) -> list[list[float]]:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts
    )

    return [item.values for item in response.embeddings]
