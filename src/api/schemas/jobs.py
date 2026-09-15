from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class SubmitJobRequest(BaseModel):
    job_id: str = Field(min_length=3, max_length=128)
    source_url: HttpUrl
    callback_url: HttpUrl
    criteria: int = 0


class JobAcceptedResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    error: str | None = None
    coarse_label: str | None = None
    coarse_score: float | None = None
    detail_label: str | None = None
    detail_score: float | None = None
    inspection: dict | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
