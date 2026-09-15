from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.application.services.classification_service import ClassificationService
from src.domain.entities import ClassificationResult, JobStatus, MediaType
from src.domain.ports import CallbackPort, JobStorePort, QueuePort, StoragePort, VideoPort
from src.shared.config import Settings
from src.shared.constants import (
    PIPELINE_CALLBACK_STAGE,
    PIPELINE_DOWNLOAD_STAGE,
    PIPELINE_PROCESS_STAGE,
    PIPELINE_UPLOAD_STAGE,
)
from src.shared.logging import get_logger
from src.shared.metrics import MetricsRegistry


class AsyncPipelineWorkers:
    def __init__(
        self,
        settings: Settings,
        queue: QueuePort,
        store: JobStorePort,
        storage: StoragePort,
        callback: CallbackPort,
        video: VideoPort,
        classifier: ClassificationService,
        metrics: MetricsRegistry,
    ) -> None:
        self._settings = settings
        self._queue = queue
        self._store = store
        self._storage = storage
        self._callback = callback
        self._video = video
        self._classifier = classifier
        self._metrics = metrics
        self._log = get_logger("pipeline")
        self._tasks: list[asyncio.Task] = []
        self._ctx: dict[str, dict[str, Any]] = {}

    async def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._download_loop(), name="download-worker"),
            asyncio.create_task(self._process_loop(), name="process-worker"),
            asyncio.create_task(self._upload_loop(), name="upload-worker"),
            asyncio.create_task(self._callback_loop(), name="callback-worker"),
        ]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    async def run_download_worker(self) -> None:
        await self._download_loop()

    async def run_process_worker(self) -> None:
        await self._process_loop()

    async def run_upload_worker(self) -> None:
        await self._upload_loop()

    async def _download_loop(self) -> None:
        while True:
            job_id = await self._queue.consume(PIPELINE_DOWNLOAD_STAGE)
            try:
                with self._metrics.timer("download"):
                    record = self._store.update_status(job_id, JobStatus.DOWNLOADING)
                    job = record.job
                    job_dir = Path(self._storage.ensure_job_dir(job_id))
                    ext = self._guess_ext(job.source_url, job.media_type)
                    src_path = str(job_dir / f"source.{ext}")
                    self._storage.download(job.source_url, src_path)
                    self._ctx[job_id] = {"source_path": src_path, "job_dir": str(job_dir)}
                    await self._queue.publish(PIPELINE_PROCESS_STAGE, job_id)
            except Exception as exc:
                self._store.update_status(job_id, JobStatus.FAILED, error=str(exc))

    async def _process_loop(self) -> None:
        while True:
            job_id = await self._queue.consume(PIPELINE_PROCESS_STAGE)
            try:
                with self._metrics.timer("process"):
                    record = self._store.update_status(job_id, JobStatus.PROCESSING)
                    job = record.job
                    ctx = self._ctx[job_id]
                    if job.media_type == MediaType.IMAGE:
                        result = self._classifier.classify_image(ctx["source_path"], criteria=job.criteria)
                        artifacts = {"source_path": ctx["source_path"]}
                    else:
                        frame_dir = os.path.join(ctx["job_dir"], "frames")
                        frames = self._video.extract_keyframes(
                            ctx["source_path"],
                            frame_dir,
                            frame_step=self._settings.frame_step,
                        )
                        result = self._classifier.classify_video_frames(frames, criteria=job.criteria)
                        artifacts = {"source_path": ctx["source_path"], "frame_dir": frame_dir}
                    ctx["result"] = result
                    ctx["artifacts"] = artifacts
                    await self._queue.publish(PIPELINE_UPLOAD_STAGE, job_id)
            except Exception as exc:
                self._store.update_status(job_id, JobStatus.FAILED, error=str(exc))

    async def _upload_loop(self) -> None:
        while True:
            job_id = await self._queue.consume(PIPELINE_UPLOAD_STAGE)
            try:
                with self._metrics.timer("upload"):
                    self._store.update_status(job_id, JobStatus.UPLOADING)
                    ctx = self._ctx[job_id]
                    result: ClassificationResult = ctx["result"]
                    result_json = os.path.join(ctx["job_dir"], "result.json")
                    with open(result_json, "w", encoding="utf-8") as f:
                        json.dump(asdict(result), f, ensure_ascii=False, indent=2)
                    uploaded = self._storage.upload(result_json, f"{job_id}_result.json")
                    ctx["artifacts"]["result_json"] = uploaded
                    await self._queue.publish(PIPELINE_CALLBACK_STAGE, job_id)
            except Exception as exc:
                self._store.update_status(job_id, JobStatus.FAILED, error=str(exc))

    async def _callback_loop(self) -> None:
        while True:
            job_id = await self._queue.consume(PIPELINE_CALLBACK_STAGE)
            try:
                with self._metrics.timer("callback"):
                    record = self._store.update_status(job_id, JobStatus.CALLBACK)
                    ctx = self._ctx[job_id]
                    result: ClassificationResult = ctx["result"]
                    payload = {
                        "job_id": job_id,
                        "status": "done",
                        "media_type": record.job.media_type.value,
                        "coarse_label": result.coarse_label,
                        "coarse_score": result.coarse_score,
                        "detail_label": result.detail_label,
                        "detail_score": result.detail_score,
                        "classification": result.meta,
                        "artifacts": ctx["artifacts"],
                    }
                    self._callback.send(record.job.callback_url, payload)
                    self._store.save_result(job_id, result, ctx["artifacts"])
                    self._ctx.pop(job_id, None)
            except Exception as exc:
                self._store.update_status(job_id, JobStatus.FAILED, error=str(exc))

    @staticmethod
    def _guess_ext(source_url: str, media_type: MediaType) -> str:
        default = "jpg" if media_type == MediaType.IMAGE else "mp4"
        clean = source_url.split("?", 1)[0].lower()
        if "." not in clean:
            return default
        ext = clean.rsplit(".", 1)[-1]
        if not ext:
            return default
        return ext
