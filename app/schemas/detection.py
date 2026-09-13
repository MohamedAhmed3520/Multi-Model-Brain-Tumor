from pydantic import BaseModel, Field


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: list[float] | None = None


class DetectionResult(BaseModel):
    detections: list[Detection]
    model_name: str
    model_version: str | None = None
    inference_time_ms: float | None = None
