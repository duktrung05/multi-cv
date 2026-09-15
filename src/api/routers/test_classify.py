"""Synchronous product inspection endpoints; legacy URLs remain as aliases."""
from __future__ import annotations
import shutil
import tempfile
from pathlib import Path
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image
from src.api.dependencies import get_container
from src.api.schemas.test_classify import TestClassifyVideoBody
from src.domain.inspection import inspection_payload

router = APIRouter(tags=["inspection"])


def _tempdir(container):
    Path(container.settings.tmp_dir).mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(prefix="inspection_", dir=container.settings.tmp_dir)


@router.post("/inspect/image")
@router.post("/test/classify/image", include_in_schema=False)
def inspect_image(file: UploadFile = File(...), container=Depends(get_container)):
    directory = _tempdir(container)
    path = Path(directory) / "input.image"
    try:
        with path.open("wb") as output:
            shutil.copyfileobj(file.file, output)
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Invalid or empty image") from exc
        result = container.classifier_service.classify_image(str(path))
        return {"ai_result": inspection_payload(result)}
    finally:
        shutil.rmtree(directory, ignore_errors=True)


@router.post("/inspect/video")
@router.post("/test/classify/video", include_in_schema=False)
def inspect_video(payload: TestClassifyVideoBody, container=Depends(get_container)):
    directory = _tempdir(container)
    try:
        video_path = str(Path(directory) / "source.mp4")
        container.storage.download(payload.url, video_path)
        step = payload.frame_step if payload.frame_step is not None else container.settings.frame_step
        if step <= 0:
            raise HTTPException(status_code=422, detail="frame_step must be positive")
        frames = container.video.extract_keyframes(video_path, str(Path(directory) / "frames"), frame_step=step)
        if not frames:
            raise HTTPException(status_code=422, detail="No decodable video frames")
        result = container.classifier_service.classify_video_frames(frames, criteria=payload.criteria)
        output = inspection_payload(result)
        # Temporary files are deleted below; return the index, not a dead file link.
        output["worst_frame_index"] = frames.index(output.pop("worst_frame"))
        return {"video_id": payload.video_id, "ai_result": output, "frame_step": step}
    finally:
        shutil.rmtree(directory, ignore_errors=True)
