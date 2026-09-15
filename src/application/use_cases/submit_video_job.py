from __future__ import annotations

from src.application.services.job_service import JobService
from src.domain.entities import JobRecord, MediaType


class SubmitVideoJobUseCase:
    def __init__(self, job_service: JobService) -> None:
        self._job_service = job_service

    async def execute(
        self,
        job_id: str,
        source_url: str,
        callback_url: str,
        criteria: int,
        trace_id: str,
    ) -> JobRecord:
        return await self._job_service.submit(
            job_id=job_id,
            media_type=MediaType.VIDEO,
            source_url=source_url,
            callback_url=callback_url,
            criteria=criteria,
            trace_id=trace_id,
        )
