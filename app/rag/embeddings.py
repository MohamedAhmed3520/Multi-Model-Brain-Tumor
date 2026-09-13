from __future__ import annotations

from typing import Any, Iterable

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class SentenceTransformerEmbeddingService:
    """Thin Sentence Transformers wrapper meant for dense document and query embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        if SentenceTransformer is not None:
            self.model = SentenceTransformer(model_name)

    def embed(self, text: str | list[str]) -> list[list[float]] | list[float]:
        if self.model is None:
            return [] if isinstance(text, list) else []
        if isinstance(text, str):
            embedding = self.model.encode(text)
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        embeddings = self.model.encode(list(text))
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)

    def embed_query(self, query: str) -> list[float]:
        return self.embed(query)

    def embed_documents(self, documents: Iterable[str]) -> list[list[float]]:
        items = list(documents)
        if not items:
            return []
        embeddings = self.embed(items)
        return embeddings if isinstance(embeddings, list) and all(isinstance(item, list) for item in embeddings) else []
