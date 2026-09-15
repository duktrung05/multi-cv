#!/usr/bin/env python3
"""Classify every image in a folder via the `/classify` API and stream results to a CSV.

Output CSV matches ``example.annotations.csv`` (columns ``image`` + ``body_parts`` as a
Label-Studio ``{"choices": [...]}`` JSON blob) plus an extra ``orig_path`` column holding
the source image path.

Rows are flushed to disk the moment each image is processed (no buffering until the end),
so a long run can be interrupted without losing finished work and progress is tail-able.

The work is wrapped in :class:`FolderClassifier` so it can be reused across folders or run
in parallel (threads/processes). Concurrent writers to the *same* CSV are serialised with a
lock; pass a shared ``threading.Lock`` when fanning out threads onto one output file.

Example
-------
    python tools/dataset/classify_folder.py \
        --folder /data/dump/genitals \
        --api http://10.0.65.25:8005/classify \
        --categories ANAL FEMALE_GENITALS MALE_GENITALS \
        --out data/reas_imagefolder_3cls/annotations.csv \
        --max-images 500
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

import requests

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from constants import DEFAULT_LABEL_MAP as _RAW_TO_CONVERTED_LABEL_MAP

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
CSV_FIELDS = ["image", "body_parts", "orig_path"]

# One lock per output path so independent FolderClassifier instances writing to the same
# file (e.g. several threads) never interleave rows. Guarded by _LOCKS_GUARD.
_LOCKS: dict = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: str) -> threading.Lock:
    key = os.path.abspath(path)
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock


def _guess_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext in (".tif", ".tiff"):
        return "image/tiff"
    if ext == ".bmp":
        return "image/bmp"
    return "application/octet-stream"


@dataclass
class FolderClassifier:
    """Classify images under ``folder`` and append annotation rows to ``out_csv``.

    Parameters
    ----------
    folder:
        Directory to scan for images.
    api_url:
        Full URL of the ``/classify`` endpoint.
    categories:
        Category names sent to the API (repeated ``-F categories=...`` form fields).
    out_csv:
        Destination CSV. Created (with header) if missing; otherwise appended to.
    max_images:
        Cap on how many images to process (``None`` = no cap).
    recursive:
        Recurse into subdirectories when ``True``; otherwise only the top level.
    choices_source:
        Which API field becomes the ``choices`` list: ``"evaluated"`` (the list the API
        returned) or ``"category"`` (just the single winning category).
    image_key:
        ``"basename"`` writes just the filename to the ``image`` column; ``"relpath"``
        writes the path relative to ``folder``. (``orig_path`` always holds the full path.)
    skip_existing:
        Skip images whose ``orig_path`` is already present in ``out_csv``, so re-runs resume
        instead of duplicating work.
    default_label:
        Label always added to every image's ``choices`` if the API result did not already
        include it (``None`` = disabled). Useful when an entire folder is known to share a
        label (e.g. all images in ``raw_data/FEMALE_GENITALS`` get ``FEMALE_GENITALS``).
    force_labels:
        If set, assign exactly these labels to *every* image and **skip the ``/classify``
        API entirely** (no network calls). ``label_map`` / ``default_label`` are still applied
        afterwards. Use when the folder's labels are already known and no inference is needed.
    label_map:
        Optional mapping from API label name -> training class name, applied to every label
        the API returns before they are written. Labels absent from the map are kept as-is.
        Mapping two API labels to the same class name de-duplicates them.
    timeout:
        Per-request timeout in seconds.
    lock:
        Optional shared lock serialising writes; defaults to a per-output-path lock.
    """

    folder: str
    api_url: str = "http://10.0.65.25:8005/classify"
    categories: Sequence[str] = field(default_factory=lambda: ["MALE_GENITALIA"])
    out_csv: str = "annotations.csv"
    max_images: Optional[int] = None
    recursive: bool = True
    choices_source: str = "categories"
    image_key: str = "basename"
    skip_existing: bool = True
    default_label: Optional[str] = None
    force_labels: Optional[Sequence[str]] = None
    label_map: Optional[Mapping[str, str]] = None
    timeout: float = 60.0
    lock: Optional[threading.Lock] = None
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        self.folder = os.path.abspath(self.folder)
        self.out_csv = os.path.abspath(self.out_csv)
        if self.lock is None:
            self.lock = _lock_for(self.out_csv)
        self._processed = 0
        self._failed = 0

    # ----- discovery -----------------------------------------------------
    def _iter_images(self) -> Iterable[str]:
        if self.recursive:
            for dirpath, _dirs, files in os.walk(self.folder, followlinks=True):
                for fn in sorted(files):
                    if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                        yield os.path.join(dirpath, fn)
        else:
            for fn in sorted(os.listdir(self.folder)):
                full = os.path.join(self.folder, fn)
                if os.path.isfile(full) and os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                    yield full

    def _load_done(self) -> set:
        """Original paths already recorded in out_csv (for resume)."""
        done: set = set()
        if not (self.skip_existing and os.path.isfile(self.out_csv)):
            return done
        try:
            with open(self.out_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    p = (row.get("orig_path") or "").strip()
                    if p:
                        done.add(os.path.abspath(p))
        except (OSError, csv.Error):
            pass
        return done

    # ----- API -----------------------------------------------------------
    def classify(self, image_path: str) -> List[str]:
        """POST one image, return the list of labels for the ``choices`` field."""
        with open(image_path, "rb") as fh:
            files = {"file": (os.path.basename(image_path), fh, _guess_mime(image_path))}
            data = [("categories", c) for c in self.categories]
            resp = self.session.post(
                self.api_url,
                headers={"accept": "application/json"},
                files=files,
                data=data,
                timeout=self.timeout,
            )
        resp.raise_for_status()
        payload = resp.json()

        if self.choices_source == "categories":
            return resp.json()['categories']
        evaluated = payload.get("evaluated")
        if isinstance(evaluated, list):
            return [str(x) for x in evaluated if str(x).strip()]
        return []

    def _normalize(self, labels: List[str]) -> List[str]:
        """Map API labels to class names (label_map), preserving order, dropping duplicates."""
        if not self.label_map:
            return labels
        out: List[str] = []
        for lab in labels:
            mapped = self.label_map.get(lab, lab)
            if mapped not in out:
                out.append(mapped)
        return out

    def _with_default(self, choices: List[str]) -> List[str]:
        """Ensure ``default_label`` is present without reordering or duplicating."""
        if self.default_label and self.default_label not in choices:
            return choices + [self.default_label]
        return choices

    # ----- writing -------------------------------------------------------
    def _image_value(self, path: str) -> str:
        if self.image_key == "relpath":
            return os.path.relpath(path, self.folder).replace("\\", "/")
        return os.path.basename(path)

    def _append_row(self, image_path: str, choices: List[str]) -> None:
        body_parts = json.dumps({"choices": choices})
        row = {
            "image": self._image_value(image_path),
            "body_parts": body_parts,
            "orig_path": image_path,
        }
        assert self.lock is not None
        with self.lock:
            os.makedirs(os.path.dirname(self.out_csv) or ".", exist_ok=True)
            new_file = not os.path.isfile(self.out_csv) or os.path.getsize(self.out_csv) == 0
            with open(self.out_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())

    # ----- driver --------------------------------------------------------
    def run(self) -> dict:
        done = self._load_done()
        for path in self._iter_images():
            if self.max_images is not None and self._processed >= self.max_images:
                break
            if os.path.abspath(path) in done:
                continue
            try:
                labels = list(self.force_labels) if self.force_labels is not None else self.classify(path)
                choices = self._with_default(self._normalize(labels))
                self._append_row(path, choices)
                self._processed += 1
                print(f"[{self._processed}] {path} -> {choices}")
            except Exception as exc:  # noqa: BLE001 - keep going on per-image failures
                self._failed += 1
                print(f"[ERROR] {path}: {exc}")
        summary = {"processed": self._processed, "failed": self._failed, "out_csv": self.out_csv}
        print(f"[DONE] {summary}")
        return summary


# Default API-label -> training-class-name mapping for the 3-class genitals model.
# constants.DEFAULT_LABEL_MAP goes training-class-name -> API-label, so reverse it here.
DEFAULT_LABEL_MAP = {v: k for k, v in _RAW_TO_CONVERTED_LABEL_MAP.items()}


def _parse_label_map(pairs: Optional[Sequence[str]]) -> Optional[dict]:
    """Parse ``API=CLASS`` CLI tokens into a mapping. ``None`` -> ``None``."""
    if pairs is None:
        return None
    mapping: dict = {}
    for tok in pairs:
        if "=" not in tok:
            raise argparse.ArgumentTypeError(f"--label-map entry must be API=CLASS, got: {tok!r}")
        api, cls = tok.split("=", 1)
        api, cls = api.strip(), cls.strip()
        if not api or not cls:
            raise argparse.ArgumentTypeError(f"--label-map entry must be API=CLASS, got: {tok!r}")
        mapping[api] = cls
    return mapping


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", default="raw_data/FEMALE_GENITALS", help="Image folder to process")
    p.add_argument("--api", default="http://10.0.65.25:8005/classify", help="/classify endpoint URL")
    p.add_argument("--categories", nargs="+", default=["SEXY", "TEXT", "FEMALE", "MALE_GENITALIA", "FEMALE_GENITALIA", "ANUS", "ADULT", "FEMALE_FACE", "MANY_PEOPLE"], help="Categories to send")
    p.add_argument("--out", default="annotations.csv", help="Output CSV path")
    p.add_argument("--max-images", type=int, default=None, help="Max images to process")
    p.add_argument("--no-recursive", action="store_true", help="Do not recurse into subfolders")
    p.add_argument("--choices-source", choices=["evaluated", "categories"], default="categories",
                   help="API field used to populate choices")
    p.add_argument("--image-key", choices=["basename", "relpath"], default="basename",
                   help="What to write in the image column")
    p.add_argument("--no-skip-existing", action="store_true", help="Do not skip already-recorded images")
    p.add_argument("--default-label", default=None,
                   help="Label always included in every image's choices if not already returned by the API")
    p.add_argument("--force-label", nargs="+", default=None, metavar="LABEL",
                   help="Assign these labels to every image and SKIP the /classify API entirely "
                        "(e.g. --force-label FEMALE_GENITALS ANAL). No network calls are made.")
    p.add_argument("--label-map", nargs="*", default=None, metavar="API=CLASS",
                   help="Map API labels to class names, e.g. FEMALE_GENITALIA=FEMALE_GENITALS ANUS=ANAL. "
                        "Pass --label-map with no args to use the built-in genitals default map.")
    p.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout (seconds)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    # --label-map omitted -> None; bare --label-map -> built-in default; with pairs -> parsed.
    if args.label_map is None:
        label_map = None
    elif len(args.label_map) == 0:
        label_map = dict(DEFAULT_LABEL_MAP)
    else:
        label_map = _parse_label_map(args.label_map)
    FolderClassifier(
        folder=args.folder,
        api_url=args.api,
        categories=args.categories,
        out_csv=args.out,
        max_images=args.max_images,
        recursive=not args.no_recursive,
        choices_source=args.choices_source,
        image_key=args.image_key,
        skip_existing=not args.no_skip_existing,
        default_label=args.default_label,
        force_labels=args.force_label,
        label_map=label_map,
        timeout=args.timeout,
    ).run()


if __name__ == "__main__":
    main()
