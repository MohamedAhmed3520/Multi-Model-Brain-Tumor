from dataclasses import dataclass
from datetime import datetime


@dataclass
class DocumentMetadata:
    source: str
    filename: str
    page: int | None = None
    section: str | None = None
    document_type: str | None = None
    chunk_index: int | None = None
    file_hash: str | None = None
    ingestion_timestamp: str | None = None
    modality: str = 'text'
