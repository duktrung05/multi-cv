from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.application.services.classification_service import ClassificationService
from src.application.services.job_service import JobService
from src.infrastructure.callback.http_callback import HttpCallbackAdapter
from src.infrastructure.ml import ModelFactory, ProductInspectorAdapter
from src.infrastructure.queue import InMemoryStageQueue
from src.infrastructure.storage import LocalStorageAdapter
from src.infrastructure.store import InMemoryJobStore
from src.infrastructure.video import FfmpegVideoAdapter
from src.shared.config import Settings
from src.shared.metrics import MetricsRegistry
from src.workers.pipeline import AsyncPipelineWorkers


@dataclass
class AppContainer:
    settings: Settings
    queue: InMemoryStageQueue
    store: InMemoryJobStore
    storage: Any
    callback: Any
    video: Any
    classifier_service: ClassificationService
    job_service: JobService
    metrics: MetricsRegistry
    workers: AsyncPipelineWorkers


def build_container(settings: Settings) -> AppContainer:
    queue = InMemoryStageQueue(max_size=settings.queue_max_size)
    store = InMemoryJobStore()
    storage = LocalStorageAdapter(settings.work_dir)
    callback = HttpCallbackAdapter(timeout_s=settings.callback_timeout_s)
    video = FfmpegVideoAdapter()
    metrics = MetricsRegistry()

    inspector = ModelFactory.build(
        config=settings.inspection_config,
        checkpoint=settings.inspection_ckpt,
        device=settings.model_device,
    )
    classifier_adapter = ProductInspectorAdapter(inspector)
    classifier_service = ClassificationService(
        classifier_adapter,
        max_concurrent_gpu_jobs=settings.max_concurrent_gpu_jobs,
    )
    job_service = JobService(queue=queue, store=store)
    workers = AsyncPipelineWorkers(
        settings=settings,
        queue=queue,
        store=store,
        storage=storage,
        callback=callback,
        video=video,
        classifier=classifier_service,
        metrics=metrics,
    )
    return AppContainer(
        settings=settings,
        queue=queue,
        store=store,
        storage=storage,
        callback=callback,
        video=video,
        classifier_service=classifier_service,
        job_service=job_service,
        metrics=metrics,
        workers=workers,
    )
