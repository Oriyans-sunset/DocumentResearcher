import os

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantStorage:
    def __init__(self, url=None, collection="docs", dim=3072):
        url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection

        # make the collection if it doesn't exist
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(collection_name=self.collection,
                                          vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
                                          )

        self.dim = dim

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k):
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k
        )
        print(results)
        context = [] # relevant chunks
        sources = set() # where the relevant chunks were found

        for r in results.points:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                context.append(text)
                sources.add(source)

        return {"context": context, "source": list(sources)}

