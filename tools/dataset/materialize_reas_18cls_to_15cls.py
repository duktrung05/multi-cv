"""
Copy `data/reas_imagefolder_18cls` into a new ImageFolder tree with merged classes (no symlinks).

Merges:
  ADULT        <- NUDE, PORN
  PRIVATE_PART <- PRIVATE_PART, ANAL, PUBIC_HAIR (PUBIC_HAIR optional if absent)
  WOMAN        <- WOMAN, WOMANS_FACE

All other class folders are copied 1:1 with the same directory name.

Copies run in parallel (thread pool) because workload is I/O bound.

Splits: always `train/` and `valid/`; copies `test/` too when present under `--src`.
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# Match torchvision ImageFolder defaults for classification images.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}

# canonical_output_dir -> source physical folders under each split (train/valid)
MERGE_SOURCES: Dict[str, Sequence[str]] = {
    "ADULT": ("NUDE", "PORN"),
    "PRIVATE_PART": ("PRIVATE_PART", "ANAL", "PUBIC_HAIR"),
    "WOMAN": ("WOMAN", "WOMANS_FACE"),
}

# Folders that exist as-is on disk in 18cls (not produced only by merges).
PASSTHROUGH_18: Tuple[str, ...] = (
    "CHILD",
    "COMICS",
    "MALE",
    "MANY_PEOPLE",
    "NO_PEOPLE",
    "PEOPLE",
    "PERSONAL_DATA",
    "PORN_COMICS",
    "SEX_TOY",
    "SEXY",
    "TEXT",
    "WOMENS_UNDERWEAR",
)


def _default_workers() -> int:
    n = os.cpu_count() or 8
    return min(64, max(16, n * 4))


def _iter_images(folder: Path) -> Iterable[Path]:
    if not folder.is_dir():
        return
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _merged_pairs(dest_class: Path, sources: Sequence[Path]) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    dest_class.mkdir(parents=True, exist_ok=True)
    for src_root in sources:
        if not src_root.is_dir():
            continue
        tag = src_root.name
        for img in _iter_images(src_root):
            pairs.append((img, dest_class / f"{tag}__{img.name}"))
    return pairs


def _passthrough_pairs(src: Path, dst: Path) -> List[Tuple[Path, Path]]:
    if not src.is_dir():
        return []
    dst.mkdir(parents=True, exist_ok=True)
    return [(img, dst / img.name) for img in _iter_images(src)]


def _parallel_copy(pairs: List[Tuple[Path, Path]], workers: int) -> int:
    if not pairs:
        return 0
    if workers <= 1:
        for s, d in pairs:
            shutil.copy2(s, d)
        return len(pairs)

    def _one(t: Tuple[Path, Path]) -> None:
        shutil.copy2(t[0], t[1])

    chunksize = max(1, len(pairs) // (workers * 8))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(_one, pairs, chunksize=chunksize):
            pass
    return len(pairs)


def materialize_split(
    src_split: Path,
    dst_split: Path,
    dry_run: bool,
    split_label: str,
    verbose: bool,
    workers: int,
) -> Tuple[int, List[str]]:
    warnings: List[str] = []
    total = 0
    for canon, phys_names in MERGE_SOURCES.items():
        src_dirs = [src_split / p for p in phys_names]
        missing = [str(p) for p in src_dirs if not p.is_dir()]
        if len(missing) == len(src_dirs):
            warnings.append(f"{src_split}: no sources for merged class {canon} (tried {phys_names})")
        pairs = _merged_pairs(dst_split / canon, src_dirs)
        if dry_run:
            k = len(pairs)
        else:
            k = _parallel_copy(pairs, workers)
        total += k
        if verbose and k:
            print(f"  [{split_label}] {canon}: {k} files", flush=True)

    for name in PASSTHROUGH_18:
        s = src_split / name
        if not s.is_dir():
            warnings.append(f"missing passthrough folder: {s}")
            continue
        pairs = _passthrough_pairs(s, dst_split / name)
        if dry_run:
            k = len(pairs)
        else:
            k = _parallel_copy(pairs, workers)
        total += k
        if verbose and k:
            print(f"  [{split_label}] {name}: {k} files", flush=True)

    return total, warnings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src",
        type=Path,
        default=Path("data/reas_imagefolder_18cls"),
        help="Source root containing train/, valid/, and optionally test/",
    )
    ap.add_argument(
        "--dst",
        type=Path,
        default=Path("data/reas_imagefolder_15cls"),
        help="Output root (per-split dirs mirror source: train/, valid/, test/ when present).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Count copies only; do not write files.")
    ap.add_argument("--quiet", action="store_true", help="Only print summary lines (no per-class progress).")
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help=f"Parallel copy threads (default: {_default_workers()} from CPU count; use 1 for sequential).",
    )
    args = ap.parse_args()

    src: Path = args.src.resolve()
    dst: Path = args.dst.resolve()
    if not src.is_dir():
        raise SystemExit(f"Source root not found: {src}")

    workers = int(args.workers) if args.workers is not None else _default_workers()
    if workers < 1:
        raise SystemExit("--workers must be >= 1")

    verbose = not args.quiet
    if verbose and not args.dry_run:
        print(
            f"Materializing images (full copy, no symlinks) from\n  {src}\n  -> {dst}\n"
            f"Parallel workers: {workers} (I/O bound; tune with --workers if disk saturates).\n",
            flush=True,
        )

    grand_total = 0
    for split in ("train", "valid", "test"):
        sp = src / split
        if split == "test" and not sp.is_dir():
            if verbose:
                print("No test/ under source; skipping test split.", flush=True)
            continue
        if not sp.is_dir():
            raise SystemExit(f"Missing split directory: {sp}")
        dp = dst / split
        if not args.dry_run:
            dp.mkdir(parents=True, exist_ok=True)
        n, warns = materialize_split(
            sp,
            dp,
            dry_run=bool(args.dry_run),
            split_label=split,
            verbose=verbose,
            workers=workers,
        )
        for w in warns:
            print(f"WARN: {w}")
        print(f"{'(dry-run) ' if args.dry_run else ''}{split}: {n} images")
        grand_total += n

    print(f"Total: {grand_total} images -> {dst}")


if __name__ == "__main__":
    main()
