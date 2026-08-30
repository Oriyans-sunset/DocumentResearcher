import os
import tempfile
from pathlib import Path

from google import genai
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()
EMBEDDING_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def _resolve_pdf_path(path: str) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """
    Resolve a PDF source into a local file path.

    If `path` looks like an S3 key (an S3_BUCKET env var is set and the file
    isn't already present on local disk), download it from S3 to a temp file
    first. Otherwise, treat `path` as a plain local filesystem path
    (preserves existing local/docker-compose behavior).

    Returns a (local_path, tmpdir) tuple. Caller should clean up tmpdir (if
    not None) after use.
    """
    bucket = os.getenv("S3_BUCKET")
    local_path = Path(path)

    if not bucket or local_path.exists():
        return local_path, None

    import boto3

    tmpdir = tempfile.TemporaryDirectory()
    dest = Path(tmpdir.name) / Path(path).name
    boto3.client("s3").download_file(bucket, path, str(dest))
    return dest, tmpdir


def load_and_chunk_pdf(path: str):
    local_path, tmpdir = _resolve_pdf_path(path)
    try:
        docs = PDFReader().load_data(file=local_path)  # return a list of Documents. 1 Document = 1 page of pdf
        pages = [d.text for d in docs if getattr(d, "text", None)]

        chunks = []
        for page in pages:
            chunks.extend(splitter.split_text(page))
        return chunks
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()

def embed_text(texts: list[str]) -> list[list[float]]:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts
    )

    return [item.values for item in response.embeddings]
