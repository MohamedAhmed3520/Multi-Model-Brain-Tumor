from typing import TypedDict

from app.schemas.classification import ClassificationResult
from app.schemas.detection import DetectionResult
from app.schemas.rag import RetrievedEvidence
from app.schemas.segmentation import SegmentationResult
from app.schemas.stt import STTResult
from app.schemas.web import WebEvidence


class MedicalAgentState(TypedDict, total=False):
    session_id: str
    user_query: str | None
    image_paths: list[str]
    audio_paths: list[str]
    document_paths: list[str]
    classification: ClassificationResult | None
    detection: DetectionResult | None
    segmentation: SegmentationResult | None
    transcription: STTResult | None
    retrieved_evidence: list[RetrievedEvidence]
    web_evidence: list[WebEvidence]
    evidence_bundle: object | None
    agent_findings: dict
    final_response: object | None
    errors: list[str]
