from pathlib import Path
from hashlib import sha256
from datetime import datetime

from app.rag.loaders import RAGLoaderFactory
from app.rag.chunking import SimpleChunker


class IngestionService:
    def __init__(self, vector_store=None):
        self.vector_store = vector_store

    def ingest(self, document_path: str, modality: str = 'text') -> dict:
        path = Path(document_path)
        digest = sha256(path.read_bytes()).hexdigest()
        loaded = RAGLoaderFactory.load(str(path))
        docs = loaded.load()
        chunks = SimpleChunker.chunk(docs)
        for i, chunk in enumerate(chunks):
            chunk.metadata = {
                'source': str(path),
                'filename': path.name,
                'page': None,
                'section': None,
                'document_type': path.suffix.lower().lstrip('.'),
                'chunk_index': i,
                'file_hash': digest,
                'ingestion_timestamp': datetime.utcnow().isoformat(),
                'modality': modality,
            }
        if self.vector_store:
            self.vector_store.add_texts([c.page_content for c in chunks], metadatas=[c.metadata for c in chunks])
        return {'document_hash': digest, 'chunks': len(chunks), 'status': 'ingested'}
