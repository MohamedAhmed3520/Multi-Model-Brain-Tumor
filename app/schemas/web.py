from pydantic import BaseModel


class WebEvidence(BaseModel):
    title: str
    url: str
    domain: str
    snippet: str
    rank: int | None = None
    relevance_score: float | None = None
