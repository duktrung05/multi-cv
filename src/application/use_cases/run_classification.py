from __future__ import annotations

from src.application.services.classification_service import ClassificationService
from src.domain.entities import ClassificationResult


class RunClassificationUseCase:
    def __init__(self, service: ClassificationService) -> None:
        self._service = service

    def classify_image(self, image_path: str, criteria: int = 0) -> ClassificationResult:
        return self._service.classify_image(image_path, criteria=criteria)

    def classify_video(self, frame_paths: list[str], criteria: int = 0) -> ClassificationResult:
        return self._service.classify_video_frames(frame_paths, criteria=criteria)
