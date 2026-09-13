from pydantic import BaseModel


class SegmentationResult(BaseModel):
    tumor_detected: bool
    threshold: float = 0.5
    coverage_ratio: float | None = None
    dice_score: float | None = None
    mask_path: str | None = None
    overlay_path: str | None = None
    model_name: str
    model_version: str | None = None
    inference_time_ms: float | None = None
