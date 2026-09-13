from pydantic import BaseModel


class MedicalAIResponse(BaseModel):
    summary: str
    classification: object | None = None
    detection: object | None = None
    segmentation: object | None = None
    transcription: object | None = None
    retrieved_evidence: list[object] = []
    web_evidence: list[object] = []
    limitations: list[str] = []
    citations: list[str] = []
