from __future__ import annotations

from typing import Any

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
except Exception:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    torch = None


class BGEReranker:
    """BGE reranker adapter that scores retriever candidates with a cross-encoder model."""

    def __init__(self, model_name: str = 'BAAI/bge-reranker-large'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        if AutoTokenizer is not None and AutoModelForSequenceClassification is not None and torch is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.eval()
            except Exception:
                self.model = None
                self.tokenizer = None

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if self.model is None or self.tokenizer is None or torch is None:
            scored = sorted(candidates, key=lambda item: float(item.get('score', 0.0)), reverse=True)
            return scored[:top_k]

        reranked = []
        for item in candidates:
            content = item.get('content', '')
            pair = [query, content]
            try:
                inputs = self.tokenizer(pair, padding=True, truncation=True, return_tensors='pt')
                with torch.no_grad():
                    scores = self.model(**inputs).logits
                score = float(torch.softmax(scores, dim=1)[0][1].item())
            except Exception:
                score = float(item.get('score', 0.0))
            new_item = dict(item)
            new_item['rerank_score'] = score
            reranked.append(new_item)

        return sorted(reranked, key=lambda item: float(item.get('rerank_score', item.get('score', 0.0))), reverse=True)[:top_k]
