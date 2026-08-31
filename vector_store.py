"""Persistent Chroma vector store for academic corpus chunks."""

import hashlib
from pathlib import Path


class ChromaVectorStore:
    collection_name = "academic_writing_corpus"

    def __init__(self, path: Path, embedder):
        try:
            import chromadb
        except ImportError as error:
            raise RuntimeError(
                "向量 RAG 需要 chromadb；请运行 pip install -r requirements.txt"
            ) from error
        self.client = chromadb.PersistentClient(path=str(path))
        self.embedder = embedder

    @staticmethod
    def _id(chunk: dict) -> str:
        key = f"{chunk['source']}#{chunk['location']}#{chunk['chunk_no']}"
        return hashlib.sha256(key.encode()).hexdigest()

    def ensure_index(self, chunks: list[dict], fingerprint: str) -> int:
        collection = self.client.get_or_create_collection(
            self.collection_name, metadata={"fingerprint": fingerprint}
        )
        if collection.metadata.get("fingerprint") == fingerprint and collection.count():
            return collection.count()
        self.client.delete_collection(self.collection_name)
        collection = self.client.create_collection(
            self.collection_name,
            metadata={"fingerprint": fingerprint, "hnsw:space": "cosine"},
        )
        batch_size = 64
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            collection.add(
                ids=[self._id(chunk) for chunk in batch],
                documents=[chunk["content"] for chunk in batch],
                embeddings=self.embedder.embed([chunk["content"] for chunk in batch]),
                metadatas=[{"source": chunk["source"],
                            "location": chunk["location"],
                            "chunk_no": chunk["chunk_no"]} for chunk in batch],
            )
        return collection.count()

    def search(self, query: str, top_k: int) -> list[dict]:
        collection = self.client.get_collection(self.collection_name)
        result = collection.query(
            query_embeddings=self.embedder.embed([query]),
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        rows = []
        for rank, (document, metadata, distance) in enumerate(zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ), 1):
            rows.append({**metadata, "content": document, "vector_rank": rank,
                         "vector_distance": distance})
        return rows
