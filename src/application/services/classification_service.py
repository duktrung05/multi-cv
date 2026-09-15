from __future__ import annotations

from threading import Semaphore

from src.domain.entities import ClassificationResult
from src.domain.ports import ClassifierPort
from src.domain.forensics import aggregate_frames


class ClassificationService:
    def __init__(self, classifier: ClassifierPort, max_concurrent_gpu_jobs: int = 1) -> None:
        self._classifier = classifier
        self._gpu_sem = Semaphore(max(1, max_concurrent_gpu_jobs))

    def classify_image(self, image_path: str, criteria: int = 0) -> ClassificationResult:
        with self._gpu_sem:
            return self._classifier.classify_image(image_path, criteria=criteria)

    def classify_video_frames(self, frame_paths: list[str], criteria: int = 0) -> ClassificationResult:
        if not frame_paths:
            return ClassificationResult(
                coarse_label="FileFormatError",
                coarse_score=0.0,
                detail_label=None,
                detail_score=None,
            )
        results = [self.classify_image(frame, criteria) for frame in frame_paths]
        return aggregate_frames(results, frame_paths)
