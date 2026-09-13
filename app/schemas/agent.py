from pydantic import BaseModel


class AgentFinding(BaseModel):
    agent_name: str
    status: str
    output_summary: str
    error_message: str | None = None


class MedicalAgentState(BaseModel):
    session_id: str | None = None
    user_query: str | None = None
    image_paths: list[str] = []
    audio_paths: list[str] = []
    document_paths: list[str] = []
    classification: object | None = None
    detection: object | None = None
    segmentation: object | None = None
    transcription: object | None = None
    retrieved_evidence: list[RetrievedEvidence] = []
    web_evidence: list[WebEvidence] = []
    evidence_bundle: object | None = None
    agent_findings: dict = {}
    final_response: object | None = None
    errors: list[str] = []
