from __future__ import annotations

from threading import Lock

from src.domain.entities import ClassificationResult, Job, JobRecord, JobStatus, utcnow


class InMemoryJobStore:
    def __init__(self) -> None:
        self._items: dict[str, JobRecord] = {}
        self._lock = Lock()

    def create(self, job: Job) -> JobRecord:
        with self._lock:
            if job.job_id in self._items:
                return self._items[job.job_id]
            record = JobRecord(job=job, status=JobStatus.QUEUED)
            self._items[job.job_id] = record
        return record

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._items.get(job_id)

    def update_status(self, job_id: str, status: JobStatus, error: str | None = None) -> JobRecord:
        with self._lock:
            item = self._items[job_id]
            item.status = status
            item.error = error
            item.updated_at = utcnow()
            return item

    def save_result(self, job_id: str, result: ClassificationResult, artifacts: dict[str, str]) -> JobRecord:
        with self._lock:
            item = self._items[job_id]
            item.result = result
            item.artifacts = artifacts
            item.status = JobStatus.DONE
            item.updated_at = utcnow()
            return item
