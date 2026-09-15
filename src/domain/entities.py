from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class JobStatus(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    UPLOADING = "uploading"
    CALLBACK = "callback"
    DONE = "done"
    FAILED = "failed"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class Job:
    job_id: str
    media_type: MediaType
    source_url: str
    callback_url: str
    criteria: int = 0
    created_at: datetime = field(default_factory=utcnow)
    trace_id: str = ""


@dataclass(slots=True)
class ClassificationResult:
    coarse_label: str
    coarse_score: float
    detail_label: str | None = None
    detail_score: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JobRecord:
    job: Job
    status: JobStatus
    updated_at: datetime = field(default_factory=utcnow)
    result: ClassificationResult | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None
