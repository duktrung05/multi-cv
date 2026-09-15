#!/usr/bin/env python3
"""Offline sliding-window video moderation with the DINOv3 ViT multilabel classifier.

Unlike ``sliding_windows.py`` (which uses a YOLO detector), this script loads the
DINOv3 ViT multilabel classifier the same way ``infer_vas.py`` does, and runs a
frame-by-frame classifier over sampled frames.

Sliding-window rule
-------------------
Each NG class keeps its own window of the last ``--windows`` (default 4) sampled
frames. A class is considered *present in the video* (making the video NG) when:

    * the class was "detected" in ``--windows`` **consecutive** sampled frames, i.e.
      every frame in the window has a per-frame probability >= ``--frame-thres``
      (default 0.1 — below this the class is treated as absent for that frame), AND
    * the average probability over that window is >= the class threshold
      (``--class-conf-thres CLASS=THRES`` if given, else ``--conf-thres``).

A video is OK when no NG class ever triggers.

Usage
-----
    # single video
    uv run python sliding_windows_dinov3.py --source path/to/video.mp4

    # directory (recursive), write an evaluation CSV
    uv run python sliding_windows_dinov3.py --source data/videos --csv-output out.csv

    # per-class thresholds
    uv run python sliding_windows_dinov3.py --source vid.mp4 \
        --class-conf-thres FEMALE_GENITALIA=0.6 EXPLICIT=0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import numpy as np
import torch
import torchvision.transforms.v2.functional as TF
from PIL import Image

from infer_vas import LoadedModel, load_model
from src.infrastructure.ml.multilabel_predict import predict_multilabel

REPO_ROOT = Path(__file__).resolve().parent

# 16-class DINOv3 ViT model (see test_ias.sh).
DEFAULT_CONFIG = "configs/deimv2/ias_dinov3_vit_s_multilabel_all.yml"
DEFAULT_WEIGHTS = (
    "outputs/ias_multilabel_dinov3_vit_s_all/summary/multilabel_classification_20260729_061317/best.pth"
)

# NG (not-good) classes: presence of any of these makes the video NG. These names
# must match entries of the model's class_list. Adjust to match the loaded model.
DEFAULT_NG_CLASSES = ("FEMALE_GENITALIA", "MALE_SEXUAL")

DIR_BATCH_VIDEO_OUT = REPO_ROOT / "outputs" / "sliding_videos"
DIR_BATCH_JSONL_OUT = REPO_ROOT / "outputs" / "jsonl"
DIR_EVIDENCES = REPO_ROOT / "outputs" / "evidences"


# --------------------------------------------------------------------------- #
# Sliding-window state (self-contained copy of the streaming semantics)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SlidingWindowConfig:
    consecutive_frames: int = 4
    frame_threshold: float = 0.4
    avg_threshold: float = 0.7
    class_thresholds: tuple[tuple[str, float], ...] = ()


class SlidingWindowState:
    """One score deque per class; a class triggers when its window is full,
    every slot qualifies (>= frame_threshold), and the window average is >=
    that class's threshold."""

    def __init__(self, config: SlidingWindowConfig):
        self._config = config
        self._class_thres: dict[str, float] = dict(config.class_thresholds)
        self._class_scores: dict[str, deque[float]] = {}

    def _qualifying_for(self, dq: deque[float]) -> list[float]:
        return [s for s in dq if s >= self._config.frame_threshold]

    def observe(
        self, class_scores: dict[str, float]
    ) -> tuple[bool, dict[str, float], dict[str, int], dict[str, list[float]]]:
        """Feed per-class scores for one frame and return (triggered, avgs, counts, windows).

        Classes seen previously but absent from ``class_scores`` are fed 0.0 so
        every window advances on the same frame timeline. ``windows`` holds each
        class's current raw score window (oldest first, up to ``consecutive_frames``
        entries).
        """
        class_avgs: dict[str, float] = {}
        class_counts: dict[str, int] = {}
        class_windows: dict[str, list[float]] = {}
        triggered = False

        all_classes = set(self._class_scores) | set(class_scores)
        for cls in all_classes:
            score = class_scores.get(cls, 0.0)
            dq = self._class_scores.setdefault(
                cls, deque(maxlen=self._config.consecutive_frames)
            )
            dq.append(score)
            qualifying = self._qualifying_for(dq)
            qual_count = len(qualifying)
            avg = (sum(qualifying) / qual_count) if qual_count else 0.0
            class_avgs[cls] = avg
            class_counts[cls] = qual_count
            class_windows[cls] = list(dq)

            if (
                len(dq) < self._config.consecutive_frames
                or qual_count < self._config.consecutive_frames
            ):
                continue
            threshold = self._class_thres.get(cls, self._config.avg_threshold)
            if avg >= threshold:
                triggered = True

        return triggered, class_avgs, class_counts, class_windows


# --------------------------------------------------------------------------- #
# Result dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FrameObservation:
    frame_idx: int
    timestamp_s: float
    class_scores: dict[str, float]
    class_avgs: dict[str, float]
    triggered: bool


@dataclass(frozen=True)
class VideoModerationResult:
    video_path: str
    is_ng: bool
    frames_sampled: int
    trigger_at_s: float | None = None
    trigger_frame_idx: int | None = None
    trigger_class: str | None = None
    trigger_avg_conf: float | None = None
    window_count_at_trigger: int | None = None
    output_video_path: str | None = None
    evidence_path: str | None = None
    peak_scores: dict[str, float] = field(default_factory=dict)
    peak_window_avg: dict[str, float] = field(default_factory=dict)
    trigger_classes: tuple[str, ...] = ()
    # Per class: the raw per-frame scores of the window at the moment that class
    # first triggered (oldest first), and that window's average.
    trigger_window_scores: dict[str, list[float]] = field(default_factory=dict)
    trigger_window_avg: dict[str, float] = field(default_factory=dict)
    observations: tuple[FrameObservation, ...] = ()


# --------------------------------------------------------------------------- #
# Model inference on a single BGR frame
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_frame_probs(loaded: LoadedModel, frame_bgr: np.ndarray) -> dict[str, float]:
    """Return {class_name: probability} for one BGR (OpenCV) frame."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    tensor = loaded.transform(img)
    tensor = tensor.unsqueeze(0).to(loaded.device)
    probs, _ = predict_multilabel(
        loaded.model, tensor, class_names=loaded.class_list, threshold=loaded.threshold
    )
    return {name: float(p) for name, p in zip(loaded.class_list, probs[0].tolist())}


# --------------------------------------------------------------------------- #
# Video helpers
# --------------------------------------------------------------------------- #
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}


def iter_video_paths(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def _video_fps(cap: cv2.VideoCapture) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    return fps if fps > 1e-6 else 30.0


def _sample_step(video_fps: float, target_fps: float) -> int:
    if target_fps <= 0:
        raise ValueError("fps must be > 0")
    return max(1, int(round(video_fps / target_fps)))


def _draw_hud(
    frame: np.ndarray,
    *,
    is_ng: bool,
    triggered: bool,
    class_avgs: dict[str, float],
    class_counts: dict[str, int],
    windows: int,
) -> None:
    status = "NG" if is_ng else ("TRIGGER" if triggered else "OK")
    color = (0, 0, 255) if is_ng or triggered else (0, 200, 0)
    lines = [status]
    for cls, avg in sorted(class_avgs.items(), key=lambda kv: kv[1], reverse=True):
        cnt = class_counts.get(cls, 0)
        lines.append(f"{cls}: avg={avg:.3f} qual={cnt}/{windows}")
    y = 28
    for line in lines:
        cv2.putText(
            frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
        )
        y += 26


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("mp4v", "avc1", "XVID"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    return None


# --------------------------------------------------------------------------- #
# Core inference
# --------------------------------------------------------------------------- #
def infer_video_sliding_window(
    video_path: str | Path,
    *,
    loaded: LoadedModel,
    ng_classes: set[str],
    windows: int = 4,
    conf_thres: float = 0.7,
    fps: float = 5.0,
    frame_thres: float = 0.3,
    class_conf_thres: dict[str, float] | None = None,
    collect_observations: bool = False,
    collect_all_triggers: bool = False,
    output_video_path: str | Path | None = None,
    save_evidence: bool = True,
    log_progress: bool = True,
) -> VideoModerationResult:
    """Run sliding-window moderation on a single video with the DINOv3 classifier.

    Only ``ng_classes`` participate in the NG decision. Each such class has its
    own window; the video is NG as soon as any class window triggers.
    """
    if windows < 1:
        raise ValueError("windows must be >= 1")
    if fps <= 0.0:
        raise ValueError("fps must be > 0")

    source = Path(video_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")

    ng_classes = {c for c in ng_classes if c in loaded.class_list}
    if not ng_classes:
        print(
            f"warning: none of the requested NG classes are in the model class_list "
            f"{loaded.class_list}; video will always be OK",
            file=sys.stderr,
        )

    per_class_thres = {k: v for k, v in (class_conf_thres or {}).items()}

    window = SlidingWindowState(
        SlidingWindowConfig(
            consecutive_frames=windows,
            frame_threshold=frame_thres,
            avg_threshold=conf_thres,
            class_thresholds=tuple(per_class_thres.items()),
        )
    )

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source!r}")
    timeline_fps = _video_fps(cap)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    step = _sample_step(timeline_fps, fps)

    writer: cv2.VideoWriter | None = None
    saved_video: Path | None = None
    if output_video_path is not None and width > 0 and height > 0:
        saved_video = Path(output_video_path).resolve()
        writer = _open_writer(saved_video, max(1.0, fps), (width, height))
        if writer is None:
            print(f"warning: could not open writer for {saved_video}", file=sys.stderr)
            saved_video = None

    if log_progress:
        print(
            f"[{source.name}] timeline_fps={timeline_fps:.2f} total_frames={total_frames} "
            f"sample_fps={fps} step={step} ng_classes={sorted(ng_classes)} "
            f"device={loaded.device}",
            file=sys.stderr,
            flush=True,
        )

    observations: list[FrameObservation] = []
    peak_scores: dict[str, float] = {c: 0.0 for c in ng_classes}
    peak_window_avg: dict[str, float] = {c: 0.0 for c in ng_classes}
    # Per-class buffer of the last `windows` (frame, score) pairs, kept in lockstep
    # with SlidingWindowState's internal deques so that once a class's window
    # triggers, this buffer holds exactly the frames that made up that window.
    frame_buffers: dict[str, deque[tuple[np.ndarray, float]]] = {
        c: deque(maxlen=windows) for c in ng_classes
    }
    # Captured once per class, the first time its window triggers.
    trigger_window_scores: dict[str, list[float]] = {}
    trigger_window_avg: dict[str, float] = {}
    frames_sampled = 0
    is_ng = False
    trigger_at_s: float | None = None
    trigger_frame_idx: int | None = None
    trigger_class: str | None = None
    trigger_avg_conf: float | None = None
    window_count_at_trigger: int | None = None
    evidence_path: str | None = None
    triggered_classes: set[str] = set()

    t_start = time.perf_counter()
    infer_time_s = 0.0
    frame_idx = 0
    try:
        while True:
            # Skip step-1 frames cheaply, decode only sampled frames.
            eof = False
            for _ in range(step - 1):
                if not cap.grab():
                    eof = True
                    break
                frame_idx += 1
            if eof:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            timestamp_s = frame_idx / timeline_fps

            t0 = time.perf_counter()
            all_probs = predict_frame_probs(loaded, frame)
            infer_time_s += time.perf_counter() - t0
            frames_sampled += 1

            # Restrict to NG classes for the window decision.
            class_scores = {c: all_probs.get(c, 0.0) for c in ng_classes}
            for c, s in class_scores.items():
                if s > peak_scores.get(c, 0.0):
                    peak_scores[c] = s
                frame_buffers[c].append((frame.copy(), s))

            triggered, class_avgs, class_counts, class_windows = window.observe(class_scores)
            for cls, avg in class_avgs.items():
                if avg > peak_window_avg.get(cls, 0.0):
                    peak_window_avg[cls] = avg

            if collect_observations:
                observations.append(
                    FrameObservation(
                        frame_idx=frame_idx,
                        timestamp_s=round(timestamp_s, 3),
                        class_scores={k: round(v, 4) for k, v in class_scores.items()},
                        class_avgs={k: round(v, 4) for k, v in class_avgs.items()},
                        triggered=triggered,
                    )
                )

            if writer is not None:
                vis = frame.copy()
                _draw_hud(
                    vis,
                    is_ng=is_ng or triggered,
                    triggered=triggered,
                    class_avgs=class_avgs,
                    class_counts=class_counts,
                    windows=windows,
                )
                writer.write(vis)

            if triggered:
                # Every class whose window fired on this frame counts as triggered.
                fired_now = [
                    cls
                    for cls, avg in class_avgs.items()
                    if class_counts.get(cls, 0) >= windows
                    and avg >= per_class_thres.get(cls, conf_thres)
                ]
                triggered_classes.update(fired_now)
                for cls in fired_now:
                    if cls not in trigger_window_scores:
                        trigger_window_scores[cls] = [
                            round(s, 4) for s in class_windows.get(cls, [])
                        ]
                        trigger_window_avg[cls] = round(class_avgs[cls], 4)

            if triggered and not is_ng:
                is_ng = True
                # Pick the first class that actually fired for the single trigger_* fields.
                trigger_class = fired_now[0] if fired_now else None
                if trigger_class is not None:
                    trigger_avg_conf = class_avgs[trigger_class]
                    window_count_at_trigger = class_counts.get(trigger_class, 0)
                trigger_at_s = timestamp_s
                trigger_frame_idx = frame_idx
                if log_progress:
                    print(
                        f"[{source.name}] TRIGGER frame={frame_idx} t={timestamp_s:.2f}s "
                        f"classes={sorted(fired_now)} avg={trigger_avg_conf}",
                        file=sys.stderr,
                        flush=True,
                    )
                if save_evidence:
                    DIR_EVIDENCES.mkdir(parents=True, exist_ok=True)
                    evidence_frame = frame
                    buf = frame_buffers.get(trigger_class) if trigger_class else None
                    if buf:
                        evidence_frame, _ = max(buf, key=lambda item: item[1])
                    avg_tag = f"_avg{trigger_avg_conf:.3f}" if trigger_avg_conf is not None else ""
                    ev = DIR_EVIDENCES / f"{source.stem}_{frame_idx}{avg_tag}.jpg"
                    cv2.imwrite(str(ev), evidence_frame)
                    evidence_path = str(ev)
                if writer is None and not collect_all_triggers:
                    break  # no video to finish and caller doesn't need every trigger class

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if log_progress:
        elapsed = time.perf_counter() - t_start
        model_fps = frames_sampled / infer_time_s if infer_time_s > 0 else 0.0
        print(
            f"[{source.name}] done elapsed={elapsed:.1f}s sampled={frames_sampled} "
            f"model_fps={model_fps:.1f} ng={is_ng}",
            file=sys.stderr,
            flush=True,
        )

    return VideoModerationResult(
        video_path=str(source),
        is_ng=is_ng,
        frames_sampled=frames_sampled,
        trigger_at_s=trigger_at_s,
        trigger_frame_idx=trigger_frame_idx,
        trigger_class=trigger_class,
        trigger_avg_conf=trigger_avg_conf,
        window_count_at_trigger=window_count_at_trigger,
        output_video_path=str(saved_video) if saved_video is not None else None,
        evidence_path=evidence_path,
        peak_scores={k: round(v, 4) for k, v in peak_scores.items()},
        peak_window_avg={k: round(v, 4) for k, v in peak_window_avg.items()},
        trigger_classes=tuple(sorted(triggered_classes)),
        trigger_window_scores=trigger_window_scores,
        trigger_window_avg=trigger_window_avg,
        observations=tuple(observations),
    )


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def _truth_label_from_path(video: Path) -> str:
    s = str(video).lower()
    return "NG" if ("_ng" in s or "ng_" in s) else "OK"


def _result_to_jsonable(result: VideoModerationResult) -> dict:
    payload = asdict(result)
    if not result.observations:
        payload.pop("observations", None)
    return payload


def _write_result_jsonl(path: Path, result: VideoModerationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(_result_to_jsonable(result), ensure_ascii=False, indent=4) + "\n")


_CSV_FIELDNAMES = (
    "path_video",
    "truth_label",
    "predict_label",
    "trigger_class",
    "trigger_at_s",
    "trigger_avg_conf",
    "trigger_window_scores",
    "output_video_path",
    "jsonl_path",
    "evidence_path",
)


def _result_csv_row(result: VideoModerationResult, *, jsonl_path: str | None = None) -> dict:
    video = Path(result.video_path)
    return {
        "path_video": result.video_path,
        "truth_label": _truth_label_from_path(video),
        "predict_label": "NG" if result.is_ng else "OK",
        "trigger_class": result.trigger_class or "",
        "trigger_at_s": "" if result.trigger_at_s is None else f"{result.trigger_at_s:.3f}",
        "trigger_avg_conf": ""
        if result.trigger_avg_conf is None
        else f"{result.trigger_avg_conf:.4f}",
        "trigger_window_scores": json.dumps(
            result.trigger_window_scores.get(result.trigger_class, [])
        )
        if result.trigger_class
        else "",
        "output_video_path": result.output_video_path or "",
        "jsonl_path": jsonl_path or "",
        "evidence_path": result.evidence_path or "",
    }


def _gt_label_from_vas_path(video: Path, vas_root: Path) -> str:
    """Ground truth from the VAS folder layout: the video's top-level subfolder
    under ``vas_root`` is OK when its name contains "OK" (e.g. "OKな動画"),
    otherwise every video under it is NG (e.g. "女性器", "女性肛門", "男性器")."""
    rel = video.relative_to(vas_root)
    subfolder = rel.parts[0]
    return "OK" if "ok" in subfolder.lower() else "NG"


def _max_trigger_column(cls: str) -> str:
    return f"max_trigger_{cls}"


def _window_scores_column(cls: str) -> str:
    return f"trigger_window_scores_{cls}"


def _window_avg_column(cls: str) -> str:
    return f"trigger_window_avg_{cls}"


class _VasEvalCsvWriter:
    """CSV columns: video_path, gt_label, pred_label, trigger_classes, evidence_path,
    plus per NG class: max_trigger_<CLASS> (highest sliding-window average that class
    ever reached in the video, 0 if never observed), trigger_window_scores_<CLASS>
    (the list of per-frame scores in the window at the moment that class first
    triggered, empty if it never triggered), and trigger_window_avg_<CLASS> (that
    window's average)."""

    def __init__(self, path: Path, ng_classes: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.ng_classes = ng_classes
        self._max_cols = [_max_trigger_column(c) for c in ng_classes]
        self._window_scores_cols = [_window_scores_column(c) for c in ng_classes]
        self._window_avg_cols = [_window_avg_column(c) for c in ng_classes]
        fieldnames = (
            "video_path",
            "gt_label",
            "pred_label",
            "trigger_classes",
            "evidence_path",
            *self._max_cols,
            *self._window_scores_cols,
            *self._window_avg_cols,
        )
        self._file = path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._file.flush()
        self.count = 0

    def append(
        self,
        video_path: str,
        gt_label: str,
        pred_label: str,
        trigger_classes: tuple[str, ...] = (),
        peak_window_avg: dict[str, float] | None = None,
        evidence_path: str | None = None,
        trigger_window_scores: dict[str, list[float]] | None = None,
        trigger_window_avg: dict[str, float] | None = None,
    ) -> None:
        peak_window_avg = peak_window_avg or {}
        trigger_window_scores = trigger_window_scores or {}
        trigger_window_avg = trigger_window_avg or {}
        row = {
            "video_path": video_path,
            "gt_label": gt_label,
            "pred_label": pred_label,
            "trigger_classes": "|".join(trigger_classes),
            "evidence_path": evidence_path or "",
        }
        for cls in self.ng_classes:
            row[_max_trigger_column(cls)] = f"{peak_window_avg.get(cls, 0.0):.4f}"
            row[_window_scores_column(cls)] = json.dumps(trigger_window_scores.get(cls, []))
            row[_window_avg_column(cls)] = (
                f"{trigger_window_avg[cls]:.4f}" if cls in trigger_window_avg else ""
            )
        self._writer.writerow(row)
        self._file.flush()
        self.count += 1

    def close(self) -> None:
        self._file.close()


def run_vas_eval(
    *,
    loaded: LoadedModel,
    ng_classes: set[str],
    vas_root: Path,
    csv_path: Path,
    windows: int,
    conf_thres: float,
    fps: float,
    frame_thres: float,
    class_conf_thres: dict[str, float] | None,
    quiet: bool,
) -> int:
    """Run inference on every video under ``vas_root`` and write video_path,
    gt_label, pred_label to ``csv_path``. Ground truth is derived from the
    top-level subfolder name (see ``_gt_label_from_vas_path``)."""
    if not vas_root.is_dir():
        print(f"VAS root not found: {vas_root}", file=sys.stderr)
        return 1

    videos = iter_video_paths(vas_root)
    if not videos:
        print(f"No videos found under {vas_root}", file=sys.stderr)
        return 1

    print(f"Evaluating {len(videos)} video(s) under {vas_root} ...", file=sys.stderr, flush=True)
    resolved_ng_classes = sorted(c for c in ng_classes if c in loaded.class_list)
    csv_writer = _VasEvalCsvWriter(csv_path, resolved_ng_classes)
    n_correct = 0
    try:
        for video in videos:
            gt_label = _gt_label_from_vas_path(video, vas_root)
            result = infer_video_sliding_window(
                video,
                loaded=loaded,
                ng_classes=ng_classes,
                windows=windows,
                conf_thres=conf_thres,
                fps=fps,
                frame_thres=frame_thres,
                class_conf_thres=class_conf_thres,
                collect_all_triggers=True,
                save_evidence=True,
                log_progress=not quiet,
            )
            pred_label = "NG" if result.is_ng else "OK"
            n_correct += int(pred_label == gt_label)
            csv_writer.append(
                str(video),
                gt_label,
                pred_label,
                result.trigger_classes,
                result.peak_window_avg,
                result.evidence_path,
                result.trigger_window_scores,
                result.trigger_window_avg,
            )
            trig = f" trigger={list(result.trigger_classes)}" if result.trigger_classes else ""
            print(f"{pred_label}\t(gt={gt_label})\t{video}{trig}")
    finally:
        csv_writer.close()

    print(
        f"Wrote {csv_writer.count} row(s) to {csv_writer.path} "
        f"(accuracy={n_correct}/{csv_writer.count})",
        file=sys.stderr,
    )
    return 0


class _IncrementalCsvWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=_CSV_FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()
        self.count = 0

    def append(self, result: VideoModerationResult, *, jsonl_path: str | None = None) -> None:
        self._writer.writerow(_result_csv_row(result, jsonl_path=jsonl_path))
        self._file.flush()
        self.count += 1

    def close(self) -> None:
        self._file.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sliding-window NSFW video moderation (DINOv3 ViT multilabel classifier)."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Video file or directory (recursive for directories); ignored when --vas-eval is set",
    )
    parser.add_argument(
        "--vas-eval",
        action="store_true",
        help=(
            "Run inference on every video under --vas-root and write video_path, gt_label, "
            "pred_label, trigger_classes to --csv-output (default: outputs/vas_eval.csv). "
            "Ground truth is OK when the video's top-level subfolder name contains 'OK', "
            "NG otherwise."
        ),
    )
    parser.add_argument(
        "--vas-root",
        type=Path,
        default=REPO_ROOT / "VAS",
        help="Root folder containing per-class video subfolders (used with --vas-eval)",
    )
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG, help="Training config (.yml)")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Checkpoint (.pth)")
    parser.add_argument("--device", default=None, help="e.g. cuda:0 or cpu")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Model per-frame active threshold (overrides config); does not affect the window rule",
    )
    parser.add_argument(
        "--ng-classes",
        nargs="*",
        default=list(DEFAULT_NG_CLASSES),
        help="Class names whose presence makes a video NG (must match model class_list)",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=4,
        help="Consecutive sampled frames required in the window (default: 4)",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.5,
        help="Default window-average threshold for NG (per-class overridable)",
    )
    parser.add_argument(
        "--frame-thres",
        type=float,
        default=0.1,
        help="Per-frame min probability to count a class as detected (default: 0.1)",
    )
    parser.add_argument("--fps", type=float, default=5.0, help="Frames to sample per second")
    parser.add_argument(
        "--class-conf-thres",
        nargs="*",
        default=[],
        metavar="CLASS=THRES",
        help="Per-class window-average overrides, e.g. FEMALE_GENITALIA=0.6 EXPLICIT=0.5",
    )
    parser.add_argument(
        "--csv-output", type=Path, default=None, help="Write evaluation CSV to this path"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write a single JSONL summary here (directory mode writes per-video JSONL under outputs/jsonl/)",
    )
    parser.add_argument(
        "--log-frames", action="store_true", help="Include per-frame observations in JSON output"
    )
    parser.add_argument(
        "--video-out",
        type=Path,
        default=None,
        help="Directory for annotated output videos (HUD overlay). Directory mode uses outputs/sliding_videos/",
    )
    parser.add_argument("--no-evidence", action="store_true", help="Do not save trigger evidence frames")
    parser.add_argument("--quiet", action="store_true", help="Disable per-video progress logs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    class_conf_thres: dict[str, float] = {}
    for item in args.class_conf_thres or []:
        cls, sep, val = item.partition("=")
        if not sep:
            print(f"warning: ignoring malformed --class-conf-thres {item!r}", file=sys.stderr)
            continue
        class_conf_thres[cls.strip()] = float(val)
    if class_conf_thres:
        print(f"Per-class thresholds: {class_conf_thres}", file=sys.stderr)

    ng_classes = set(args.ng_classes)

    if args.vas_eval:
        print("Loading DINOv3 model ...", file=sys.stderr, flush=True)
        loaded = load_model(
            config=args.config, weights=args.weights, threshold=args.threshold, device=args.device
        )
        print(f"class_list = {loaded.class_list}", file=sys.stderr)
        vas_root = args.vas_root if args.vas_root.is_absolute() else REPO_ROOT / args.vas_root
        csv_path = args.csv_output or (REPO_ROOT / "outputs" / "vas_eval.csv")
        csv_path = csv_path if csv_path.is_absolute() else REPO_ROOT / csv_path
        return run_vas_eval(
            loaded=loaded,
            ng_classes=ng_classes,
            vas_root=vas_root,
            csv_path=csv_path,
            windows=args.windows,
            conf_thres=args.conf_thres,
            fps=args.fps,
            frame_thres=args.frame_thres,
            class_conf_thres=class_conf_thres or None,
            quiet=args.quiet,
        )

    if args.source is None:
        print("error: --source is required unless --vas-eval is set", file=sys.stderr)
        return 2

    source_path = args.source if args.source.is_absolute() else REPO_ROOT / args.source
    if source_path.is_file():
        videos = [source_path]
        source_is_dir = False
    elif source_path.is_dir():
        videos = iter_video_paths(source_path)
        source_is_dir = True
        if not videos:
            print(f"No videos found under {source_path}", file=sys.stderr)
            return 1
    else:
        print(f"Source not found: {source_path}", file=sys.stderr)
        return 1

    print("Loading DINOv3 model ...", file=sys.stderr, flush=True)
    loaded = load_model(
        config=args.config, weights=args.weights, threshold=args.threshold, device=args.device
    )
    print(f"class_list = {loaded.class_list}", file=sys.stderr)

    video_out_dir = args.video_out  # None => no annotated video written
    jsonl_out_dir = DIR_BATCH_JSONL_OUT if (source_is_dir and args.output is None) else None

    csv_writer: _IncrementalCsvWriter | None = None
    if args.csv_output is not None:
        csv_path = args.csv_output if args.csv_output.is_absolute() else REPO_ROOT / args.csv_output
        csv_writer = _IncrementalCsvWriter(csv_path)
        print(f"Writing CSV incrementally to {csv_path}", file=sys.stderr)

    results: list[VideoModerationResult] = []
    try:
        for video in videos:
            out_video = None
            if video_out_dir is not None:
                video_out_dir.mkdir(parents=True, exist_ok=True)
                out_video = video_out_dir / f"{video.stem}_pred.mp4"

            result = infer_video_sliding_window(
                video,
                loaded=loaded,
                ng_classes=ng_classes,
                windows=args.windows,
                conf_thres=args.conf_thres,
                fps=args.fps,
                frame_thres=args.frame_thres,
                class_conf_thres=class_conf_thres or None,
                collect_observations=args.log_frames,
                output_video_path=str(out_video) if out_video is not None else None,
                save_evidence=not args.no_evidence,
                log_progress=not args.quiet,
            )
            results.append(result)

            jsonl_path: str | None = None
            if jsonl_out_dir is not None:
                jsonl_out_dir.mkdir(parents=True, exist_ok=True)
                jp = jsonl_out_dir / f"{video.stem}.jsonl"
                _write_result_jsonl(jp, result)
                jsonl_path = str(jp)
            if csv_writer is not None:
                csv_writer.append(result, jsonl_path=jsonl_path)

            status = "NG" if result.is_ng else "OK"
            extra = ""
            if result.is_ng:
                extra = (
                    f" @ {result.trigger_at_s:.2f}s class={result.trigger_class} "
                    f"avg={result.trigger_avg_conf:.4f}"
                )
            print(f"{status}\t{video}{extra}")
    finally:
        if csv_writer is not None:
            csv_writer.close()
            print(f"Wrote {csv_writer.count} row(s) to {csv_writer.path}", file=sys.stderr)

    if args.output is not None and not source_is_dir:
        out_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(_result_to_jsonable(row), ensure_ascii=False, indent=4) + "\n")
        print(f"Wrote {len(results)} result(s) to {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
