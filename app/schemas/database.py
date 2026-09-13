from pydantic import BaseModel


class SessionRecord(BaseModel):
    session_id: str
    created_at: str | None = None
    updated_at: str | None = None
    input_modality: str | None = None
    original_filename: str | None = None
    file_type: str | None = None
    file_size: int | None = None
    file_hash: str | None = None
    processing_status: str | None = None
    requested_operation: str | None = None
    user_query: str | None = None
    processing_duration: float | None = None
