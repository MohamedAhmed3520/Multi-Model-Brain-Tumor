from pydantic import BaseModel


class RetrievedEvidence(BaseModel):
    content: str
    source: str
    page: int | None = None
    section: str | None = None
    score: float | None = None
    modality: str = 'text'
    document_id: str | None = None
    chunk_id: str | None = None


class EvidenceBundle(BaseModel):
    user_query: str | None = None
    classification: object | None = None
    detection: object | None = None
    segmentation: object | None = None
    transcription: object | None = None
    retrieved_documents: list[RetrievedEvidence] = []
    web_sources: list[object] = []
    agent_findings: dict = {}
