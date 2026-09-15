from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8126
    inspection_config: str = "configs/product_inspection.yml"
    inspection_ckpt: str = "ckpts/product_inspection.pth"
    model_device: str = "auto"
    work_dir: str = "outputs/jobs"
    tmp_dir: str = "outputs/tmp"
    frame_step: int = 24
    queue_max_size: int = 4096
    callback_timeout_s: float = 5.0
    classify_batch_size: int = 16
    max_concurrent_gpu_jobs: int = 1

    @staticmethod
    def from_env() -> "Settings":
        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
        return Settings(
            host=os.getenv("APP_HOST", "0.0.0.0"),
            port=int(os.getenv("APP_PORT", "8126")),
            inspection_config=os.getenv("INSPECTION_CONFIG", "configs/product_inspection.yml"),
            inspection_ckpt=os.getenv("INSPECTION_CKPT", "ckpts/product_inspection.pth"),
            model_device=os.getenv("MODEL_DEVICE", "auto"),
            work_dir=os.getenv("WORK_DIR", "outputs/jobs"),
            tmp_dir=os.getenv("TMP_DIR", "outputs/tmp"),
            frame_step=int(os.getenv("VIDEO_FRAME_STEP", "24")),
            queue_max_size=int(os.getenv("QUEUE_MAX_SIZE", "4096")),
            callback_timeout_s=float(os.getenv("CALLBACK_TIMEOUT_S", "5.0")),
            classify_batch_size=int(os.getenv("CLASSIFY_BATCH_SIZE", "16")),
            max_concurrent_gpu_jobs=int(os.getenv("MAX_CONCURRENT_GPU_JOBS", "1")),
        )
