from pydantic import BaseModel


class STTResult(BaseModel):
    text: str
    language: str | None = None
    model_name: str
    model_version: str | None = None
    confidence: float | None = None
    duration_seconds: float | None = None
    inference_time_ms: float | None = None
