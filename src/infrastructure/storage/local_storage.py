from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path


class LocalStorageAdapter:
    def __init__(self, work_dir: str) -> None:
        self._work_dir = Path(work_dir)
        self._downloads = self._work_dir / "downloads"
        self._uploads = self._work_dir / "uploads"
        self._downloads.mkdir(parents=True, exist_ok=True)
        self._uploads.mkdir(parents=True, exist_ok=True)

    def ensure_job_dir(self, job_id: str) -> str:
        job_dir = self._work_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return str(job_dir)

    def download(self, url: str, dst_path: str) -> str:
        Path(os.path.dirname(dst_path)).mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dst_path)
        return dst_path

    def upload(self, src_path: str, dst_name: str) -> str:
        dst = self._uploads / dst_name
        shutil.copy2(src_path, dst)
        return str(dst)
