from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException

from src.api.dependencies import get_container
from src.api.schemas.jobs import JobAcceptedResponse, JobStatusResponse, SubmitJobRequest
from src.bootstrap import AppContainer
from src.domain.entities import MediaType

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("/image", response_model=JobAcceptedResponse, status_code=202)
async def submit_image_job(
    payload: SubmitJobRequest,
    x_trace_id: str = Header(default=""),
    container: AppContainer = Depends(get_container),
) -> JobAcceptedResponse:
    record = await container.job_service.submit(
        job_id=payload.job_id,
        media_type=MediaType.IMAGE,
        source_url=str(payload.source_url),
        callback_url=str(payload.callback_url),
        criteria=payload.criteria,
        trace_id=x_trace_id,
    )
    return JobAcceptedResponse(job_id=record.job.job_id, status=record.status.value)


@router.post("/video", response_model=JobAcceptedResponse, status_code=202)
async def submit_video_job(
    payload: SubmitJobRequest,
    x_trace_id: str = Header(default=""),
    container: AppContainer = Depends(get_container),
) -> JobAcceptedResponse:
    record = await container.job_service.submit(
        job_id=payload.job_id,
        media_type=MediaType.VIDEO,
        source_url=str(payload.source_url),
        callback_url=str(payload.callback_url),
        criteria=payload.criteria,
        trace_id=x_trace_id,
    )
    return JobAcceptedResponse(job_id=record.job.job_id, status=record.status.value)


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    job_id: str,
    container: AppContainer = Depends(get_container),
) -> JobStatusResponse:
    record = container.job_service.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=record.job.job_id,
        status=record.status.value,
        error=record.error,
        coarse_label=record.result.coarse_label if record.result else None,
        coarse_score=record.result.coarse_score if record.result else None,
        detail_label=record.result.detail_label if record.result else None,
        detail_score=record.result.detail_score if record.result else None,
        inspection=record.result.meta if record.result else None,
        artifacts=record.artifacts,
    )
