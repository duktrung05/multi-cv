from __future__ import annotations

from src.domain.entities import Job, JobRecord, JobStatus, MediaType
from src.domain.ports import JobStorePort, QueuePort
from src.shared.constants import PIPELINE_DOWNLOAD_STAGE


class JobService:
    def __init__(self, queue: QueuePort, store: JobStorePort) -> None:
        self._queue = queue
        self._store = store

    async def submit(
        self,
        job_id: str,
        media_type: MediaType,
        source_url: str,
        callback_url: str,
        criteria: int = 0,
        trace_id: str = "",
    ) -> JobRecord:
        job = Job(
            job_id=job_id,
            media_type=media_type,
            source_url=source_url,
            callback_url=callback_url,
            criteria=criteria,
            trace_id=trace_id,
        )
        record = self._store.create(job)
        self._store.update_status(job_id, JobStatus.QUEUED)
        await self._queue.publish(PIPELINE_DOWNLOAD_STAGE, job_id)
        return record

    def get(self, job_id: str) -> JobRecord | None:
        return self._store.get(job_id)
