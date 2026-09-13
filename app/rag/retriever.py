from app.schemas.rag import RetrievedEvidence


class Retriever:
    """Vector retriever abstraction returning structured evidence objects."""

    def __init__(self, vector_store=None):
        self.vector_store = vector_store

    def search(self, query: str, top_k: int = 5) -> list[RetrievedEvidence]:
        if self.vector_store is None:
            return []
        results = self.vector_store.similarity_search(query, k=top_k)
        evidence = []
        for item in results:
            metadata = item.metadata if hasattr(item, 'metadata') else {}
            evidence.append(RetrievedEvidence(
                content=item.page_content,
                source=metadata.get('source', 'unknown'),
                page=metadata.get('page'),
                section=metadata.get('section'),
                score=getattr(item, 'score', None),
                modality=metadata.get('modality', 'text'),
                document_id=metadata.get('document_id'),
                chunk_id=metadata.get('chunk_id'),
            ))
        return evidence
