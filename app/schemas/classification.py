from pydantic import BaseModel


class ClassificationResult(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    model_name: str
    model_version: str | None = None
    inference_time_ms: float | None = None
