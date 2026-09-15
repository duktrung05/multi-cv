from __future__ import annotations

import subprocess
from pathlib import Path


class FfmpegVideoAdapter:
    def extract_keyframes(self, video_path: str, output_dir: str, frame_step: int) -> list[str]:
        if frame_step <= 0:
            raise ValueError("frame_step must be positive")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        pattern = str(Path(output_dir) / "frame_%06d.jpg")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vf",
            f"select=not(mod(n\\,{frame_step}))",
            "-vsync",
            "vfr",
            pattern,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return sorted(str(p) for p in Path(output_dir).glob("frame_*.jpg"))
