from __future__ import annotations

from typing import Any

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None


class BM25Retriever:
    """Sparse lexical retrieval component using rank-bm25 BM25Okapi."""

    def __init__(self, tokenizer: Any | None = None):
        self.tokenizer = tokenizer or (lambda text: text.lower().split())
        self._index = None
        self._documents: list[str] = []
        self._metadata: list[dict[str, Any]] = []

    def build(self, documents: list[str], metadata: list[dict[str, Any]] | None = None):
        self._documents = documents
        self._metadata = metadata or [{} for _ in documents]
        if BM25Okapi is None:
            self._index = None
            return self
        tokenized = [self.tokenizer(doc) for doc in documents]
        self._index = BM25Okapi(tokenized)
        return self

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._index is None or not self._documents:
            return []
        scores = self._index.get_scores(self.tokenizer(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            results.append({
                'content': self._documents[idx],
                'score': float(score),
                'source': self._metadata[idx].get('source', 'unknown'),
                'metadata': self._metadata[idx],
                'modality': self._metadata[idx].get('modality', 'text'),
                'document_id': self._metadata[idx].get('document_id'),
                'chunk_id': self._metadata[idx].get('chunk_id'),
            })
        return results
