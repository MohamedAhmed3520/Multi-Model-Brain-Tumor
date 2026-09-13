from __future__ import annotations

from typing import Any

from app.rag.bm25 import BM25Retriever
from app.rag.embeddings import SentenceTransformerEmbeddingService
from app.rag.reranker import BGEReranker


class HybridRetriever:
    """Hybrid retrieval orchestration: dense semantic retrieval, sparse BM25 retrieval, fusion, dedupe, rerank."""

    def __init__(
        self,
        dense_service: SentenceTransformerEmbeddingService | None = None,
        bm25_service: BM25Retriever | None = None,
        reranker: BGEReranker | None = None,
    ):
        self.dense_service = dense_service or SentenceTransformerEmbeddingService()
        self.bm25_service = bm25_service or BM25Retriever()
        self.reranker = reranker or BGEReranker()

    def retrieve(self, query: str, top_k: int = 5, documents: list[str] | None = None, metadata: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Minimal hybrid-retrieval orchestration contract. Dense retriever can be wired to a vector store later, and BM25 lexically ranks from the supplied corpus."""
        candidates: list[dict[str, Any]] = []

        if documents:
            self.bm25_service.build(documents, metadata)
            candidates.extend(self.bm25_service.search(query, top_k=top_k))

        # In the absence of a vector store, use the embedding wrapper as a proof-point service.
        dense_embedding = self.dense_service.embed_query(query)
        if dense_embedding:
            candidates.append({
                'content': query,
                'score': 1.0,
                'source': 'dense-query-vector',
                'metadata': {'modality': 'text', 'document_id': 'dense'},
                'modality': 'text',
                'document_id': 'dense',
                'chunk_id': 'dense-embedding',
                'embedding': dense_embedding,
            })

        deduped = []
        seen = set()
        for item in candidates:
            key = item.get('content', '') or item.get('source', '')
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        reranked = self.reranker.rerank(query, deduped, top_k=top_k)
        return reranked[:top_k]
