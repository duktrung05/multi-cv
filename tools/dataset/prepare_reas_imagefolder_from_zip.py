"""
Prepare ReaS ImageFolder classification dataset from a "zip of zips".

"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# Keep in sync with torchvision ImageFolder's allowed image extensions.
# (Notably, `.gif` is not included by default.)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


def default_jp_to_en_map() -> Dict[str, str]:
    # Note: class folder names must match the class_list you use in YAML for training.
    return {
        # People
        "女性の顔": "WOMANS_FACE",
        "女性顔": "WOMANS_FACE",
        "女性": "WOMAN",
        "男性": "MALE",
        "人(一部分)": "PEOPLE",
        "複数人": "MANY_PEOPLE",

        # Sexual content
        "女性器": "PRIVATE_PART",
        "男性器": "PRIVATE_PART",
        "女性の陰毛": "PRIVATE_PART",
        "女性器モザイク": "PRIVATE_PART",
        "女性器(モザイク)": "PRIVATE_PART",
        "男性器(モザイク)": "PRIVATE_PART",

        "肛門": "ANAL",
        "女性肛門": "ANAL",
        "女性ヌード": "NUDE",
        "下着水着": "WOMENS_UNDERWEAR",
        "下着/水着姿": "WOMENS_UNDERWEAR",
        "女性セクシー": "SEXY",

        "ポルノ": "PORN",
        "女性ポルノ": "PORN",
        "男性ポルノ": "PORN",
        "アダルトグッズ": "SEX_TOY",

        # Others
        "テキスト": "TEXT",
        "漫画": "COMICS",
        "ポルノ漫画": "PORN_COMICS",
        "個人情報": "PERSONAL_DATA",
        "子ども": "CHILD",
        "人が映っていない": "NO_PEOPLE",
    }


def safe_filename(name: str) -> str:
    """
    Keep filenames mostly intact, but normalize path separators and strip odd control chars.
    We do NOT romanize JP here because only folder labels need English for training.
    """
    name = name.replace("\\", "/")
    name = re.sub(r"[\x00-\x1f]", "_", name)
    name = name.strip().strip("/")
    return name


def iter_outer_inner_zip_members(zf: zipfile.ZipFile) -> Iterable[str]:
    for n in zf.namelist():
        n2 = safe_filename(n)
        if n2.lower().endswith(".zip") and not n2.endswith("/"):
            yield n2


def choose_label_from_inner(zip_member_path: str) -> str:
    # outer member example: "2026AIモデル教育用画像/男性器.zip" -> "男性器"
    stem = Path(zip_member_path).name
    if stem.lower().endswith(".zip"):
        stem = stem[: -len(".zip")]
    return stem


def unique_dest_path(dest_dir: Path, filename: str) -> Path:
    """
    Avoid overwriting when different source images share same filename.
    Strategy: if exists, append _{n} before suffix.
    """
    p = dest_dir / filename
    if not p.exists():
        return p
    base = p.stem
    suf = p.suffix
    for i in range(1, 10_000_000):
        cand = dest_dir / f"{base}_{i}{suf}"
        if not cand.exists():
            return cand
    raise RuntimeError(f"Too many filename collisions under {dest_dir} for {filename}")


def is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def parse_args():
    p = argparse.ArgumentParser(
        description="Unzip a 'zip of zips' and prepare ImageFolder dataset with English labels."
    )
    p.add_argument(
        "--outer-zip",
        type=str,
        required=True,
        help="Path to the outer zip (contains multiple inner zips).",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default="data/reas_imagefolder",
        help="Output dataset root. Will create <output_root>/{train,valid,test}/<class>/..",
    )
    p.add_argument(
        "--split",
        type=str,
        default="train",
        help="(Legacy) Single split folder name under output_root when --make-splits is off (default: train).",
    )
    p.add_argument(
        "--make-splits",
        action="store_true",
        help="Create train/valid/test splits (default ratios 8/1/1).",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train ratio when --make-splits is on (default: 0.8).",
    )
    p.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Valid ratio when --make-splits is on (default: 0.1).",
    )
    p.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test ratio when --make-splits is on (default: 0.1).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting (default: 42).",
    )
    p.add_argument(
        "--valid-name",
        type=str,
        default="valid",
        help="Name of validation split folder (default: valid).",
    )
    p.add_argument(
        "--labels-map",
        type=str,
        default="",
        help="Optional path to a JSON mapping dict {jp_label: en_label}. If omitted, uses built-in mapping.",
    )
    p.add_argument(
        "--mapped-source",
        type=str,
        default="auto",
        choices=["outer", "inner", "auto"],
        help=(
            "When --label-mode=mapped, choose where the JP label comes from. "
            "'outer': use zip filename stem (e.g. 女性肛門.zip). "
            "'inner': use first path component inside zip (e.g. 女性顔/xxx.jpg). "
            "'auto': prefer inner when it exists in labels-map, else fall back to outer. "
            "Default: auto (fixes mispacked zips)."
        ),
    )
    p.add_argument(
        "--label-mode",
        type=str,
        default="mapped",
        choices=["mapped", "topic"],
        help=(
            "Labeling mode. "
            "'mapped': map JP topic -> canonical EN ReaS classes (default). "
            "'topic': keep (normalized) JP topic names as class folders (for topic-level training)."
        ),
    )
    p.add_argument(
        "--topic-map",
        type=str,
        default="",
        help=(
            "Optional JSON mapping dict {jp_topic: normalized_topic}. "
            "Only used when --label-mode=topic. If omitted, uses built-in normalization "
            "(e.g. merges 女性顔 -> 女性の顔 to get 24 topics)."
        ),
    )
    p.add_argument(
        "--topic-lang",
        type=str,
        default="jp",
        choices=["jp", "en"],
        help=(
            "When --label-mode=topic, choose class folder language. "
            "'jp' keeps (normalized) JP topic names. "
            "'en' maps topics to English folder names for easier tracking."
        ),
    )
    p.add_argument(
        "--topic-en-map",
        type=str,
        default="",
        help=(
            "Optional JSON mapping dict {jp_topic(normalized): en_topic}. "
            "Only used when --label-mode=topic --topic-lang=en. If omitted, uses built-in mapping."
        ),
    )
    p.add_argument(
        "--topic-source",
        type=str,
        default="auto",
        choices=["outer", "inner", "auto"],
        help=(
            "When --label-mode=topic, choose where topic comes from. "
            "'outer': use zip filename (e.g. 女性肛門.zip). "
            "'inner': use first path component inside zip (e.g. 女性顔/xxx.jpg). "
            "'auto': prefer inner when it looks like a known topic, else outer. "
            "Default: auto (fixes mispacked zips while keeping full coverage)."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned actions, do not write files.",
    )
    p.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="If >0, stop after extracting N images for each class (useful for quick tests).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files instead of auto-renaming on collisions.",
    )
    return p.parse_args()


def load_label_map(path: str) -> Dict[str, str]:
    if not path:
        return default_jp_to_en_map()
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if not isinstance(m, dict):
        raise ValueError(f"labels-map must be a JSON object/dict, got: {type(m)}")
    return {str(k): str(v) for k, v in m.items()}


def default_topic_normalize_map() -> Dict[str, str]:
    """
    Normalize/merge topic names when training per-topic.

    The source directory contains 25 topic zips, but some are synonymous variants.
    We normalize them so the resulting ImageFolder has 24 topics by default.
    """
    return {
        # Optional merge to reduce topics from 25 -> 24.
        # Keep mosaic variant under the same topic for topic-level training.
        "女性器モザイク": "女性器",
    }


def load_topic_map(path: str) -> Dict[str, str]:
    if not path:
        return default_topic_normalize_map()
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if not isinstance(m, dict):
        raise ValueError(f"topic-map must be a JSON object/dict, got: {type(m)}")
    return {str(k): str(v) for k, v in m.items()}


def default_topic_en_map() -> Dict[str, str]:
    """
    English names for topic-level training. Keys are normalized JP topic names.
    """
    return {
        "アダルトグッズ": "SEX_TOY",
        "テキスト": "TEXT",
        "ポルノ": "PORN",
        "ポルノ漫画": "PORN_COMICS",
        "下着水着": "WOMENS_UNDERWEAR",
        "人が映っていない": "NO_PEOPLE",
        "人(一部分)": "PEOPLE_PART",
        "個人情報": "PERSONAL_DATA",
        "女性": "WOMAN",
        "女性セクシー": "SEXY_WOMAN",
        "女性ヌード": "FEMALE_NUDE",
        "女性の陰毛": "FEMALE_PUBIC_HAIR",
        "女性ポルノ": "FEMALE_PORN",
        "女性器": "FEMALE_GENITALS",
        "女性肛門": "FEMALE_ANAL",
        "女性顔": "WOMANS_FACE",
        "子ども": "CHILD",
        "漫画": "COMICS",
        "男性": "MALE",
        "男性ポルノ": "MALE_PORN",
        "男性器": "MALE_GENITALS",
        "男性器(モザイク)": "MALE_GENITALS_MOSAIC",
        "肛門": "ANAL",
        "複数人": "MANY_PEOPLE",
    }


def load_topic_en_map(path: str) -> Dict[str, str]:
    if not path:
        return default_topic_en_map()
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if not isinstance(m, dict):
        raise ValueError(f"topic-en-map must be a JSON object/dict, got: {type(m)}")
    return {str(k): str(v) for k, v in m.items()}


def main():
    args = parse_args()
    outer_zip = Path(args.outer_zip)
    out_root = Path(args.output_root)
    split = args.split

    label_mode = str(args.label_mode)
    jp_to_en = load_label_map(args.labels_map) if label_mode == "mapped" else {}
    mapped_source = str(args.mapped_source)
    topic_norm = load_topic_map(args.topic_map) if label_mode == "topic" else {}
    topic_lang = str(args.topic_lang)
    topic_en = (
        load_topic_en_map(args.topic_en_map) if (label_mode == "topic" and topic_lang == "en") else {}
    )
    topic_source = str(args.topic_source)

    if args.make_splits:
        split_names = ("train", args.valid_name, "test")
        split_roots = {s: (out_root / s) for s in split_names}
        manifest_paths = {s: (out_root / f"manifest_{s}.csv") for s in split_names}
    else:
        split_names = (split,)
        split_roots = {split: (out_root / split)}
        manifest_paths = {split: (out_root / f"manifest_{split}.csv")}

    if label_mode == "mapped":
        labels_map_out_path = out_root / "labels_map.json"
    else:
        labels_map_out_path = out_root / ("topic_map_en.json" if topic_lang == "en" else "topic_map.json")

    print(f"Outer archive/dir: {outer_zip}")
    print(f"Label mode: {label_mode}")
    if label_mode == "mapped":
        print(f"Mapped label source: {mapped_source}")
    if label_mode == "topic":
        print(f"Topic folder names: {topic_lang}")
        print(f"Topic source: {topic_source}")
    if args.make_splits:
        print(
            f"Output root: {out_root} (splits=train/{args.valid_name}/test ratios="
            f"{args.train_ratio:g}/{args.valid_ratio:g}/{args.test_ratio:g}, seed={args.seed})"
        )
    else:
        print(f"Output root: {out_root} (split='{split}')")
    print(f"Dry run: {args.dry_run}")
    print(f"Limit per class: {args.limit_per_class if args.limit_per_class > 0 else 'no limit'}")

    if not outer_zip.exists():
        raise FileNotFoundError(f"Outer input not found: {outer_zip}")
    if not outer_zip.is_file() and not outer_zip.is_dir():
        raise ValueError(f"Outer input must be a file or directory: {outer_zip}")

    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        for s in split_names:
            split_roots[s].mkdir(parents=True, exist_ok=True)

        with open(labels_map_out_path, "w", encoding="utf-8") as f:
            if label_mode == "mapped":
                json.dump(jp_to_en, f, ensure_ascii=False, indent=2)
            else:
                payload = {"normalize_map": topic_norm}
                if topic_lang == "en":
                    payload["en_map"] = topic_en
                json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Wrote label map: {labels_map_out_path}")

    total_images = 0
    unknown_labels = set()
    per_jp_label_counts = Counter()
    per_label_counts_by_split: Dict[str, Counter] = {s: Counter() for s in split_names}

    manifest_files: Dict[str, object] = {}
    manifest_writers: Dict[str, csv.writer] = {}
    if not args.dry_run:
        for s in split_names:
            mf = open(manifest_paths[s], "w", newline="", encoding="utf-8")
            manifest_files[s] = mf
            w = csv.writer(mf)
            manifest_writers[s] = w
            w.writerow(
                [
                    "outer_zip",
                    "inner_zip_member",
                    "inner_member",
                    "outer_jp_label",
                    "jp_label",
                    "label",
                    "split",
                    "output_path",
                ]
            )

    try:
        # Validate ratios when splitting.
        if args.make_splits:
            total_ratio = args.train_ratio + args.valid_ratio + args.test_ratio
            if total_ratio <= 0:
                raise ValueError("Sum of ratios must be > 0")
            # Normalize small floating errors.
            args.train_ratio = args.train_ratio / total_ratio
            args.valid_ratio = args.valid_ratio / total_ratio
            args.test_ratio = args.test_ratio / total_ratio

        # Determine the list of inner zip files.
        single_inner_zip_mode = False
        if outer_zip.is_dir():
            inner_members = sorted([p for p in outer_zip.rglob("*.zip") if p.is_file()])
            if not inner_members:
                raise RuntimeError(f"No inner .zip files found under directory {outer_zip}")
            print(f"Found {len(inner_members)} inner zip(s) under directory.")
        else:
            with zipfile.ZipFile(outer_zip, "r") as outer:
                inner_members = list(iter_outer_inner_zip_members(outer))
                if inner_members:
                    print(f"Found {len(inner_members)} inner zip(s) inside archive.")
                else:
                    # Treat the input itself as a single inner zip (one class).
                    single_inner_zip_mode = True
                    inner_members = [outer_zip]
                    print("Input looks like a single class zip (no nested inner zips).")

        # First pass: collect all image entries per label, so we can split per-class.
        per_label_items: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)

        def _norm_topic(x: str) -> str:
            return topic_norm.get(x, x)

        def _choose_mapped_jp_label(outer_jp: str, inner_member: str) -> str:
            inner_parts = Path(inner_member).parts
            inner_jp = inner_parts[0] if len(inner_parts) > 1 else ""
            if mapped_source == "outer":
                return outer_jp
            if mapped_source == "inner":
                return inner_jp or outer_jp
            # auto
            if inner_jp and inner_jp in jp_to_en:
                return inner_jp
            return outer_jp

        def _topic_to_label(jp_topic_raw: str) -> str | None:
            jp_topic = _norm_topic(jp_topic_raw)
            if topic_lang == "en":
                return topic_en.get(jp_topic)
            return jp_topic

        def _choose_topic(outer_jp: str, inner_member: str) -> str:
            inner_parts = Path(inner_member).parts
            inner_jp = inner_parts[0] if len(inner_parts) > 1 else ""
            if topic_source == "outer":
                return outer_jp
            if topic_source == "inner":
                return inner_jp or outer_jp
            # auto
            if inner_jp and _topic_to_label(inner_jp) is not None:
                return inner_jp
            return outer_jp
        if single_inner_zip_mode:
            inner_zip_path = Path(inner_members[0])
            inner_zip_id = str(inner_zip_path)
            outer_jp_label = choose_label_from_inner(inner_zip_id)
            per_jp_label_counts[outer_jp_label] += 1
            label = "__topic_per_image__" if label_mode == "topic" else None

            with zipfile.ZipFile(inner_zip_path, "r") as inner:
                inner_names = [safe_filename(n) for n in inner.namelist()]
                for inner_name in inner_names:
                    if inner_name.endswith("/"):
                        continue
                    if not is_image_file(inner_name):
                        continue
                    if label_mode == "mapped":
                        jp_label = _choose_mapped_jp_label(outer_jp_label, inner_name)
                        label = jp_to_en.get(jp_label)
                        if label is None:
                            unknown_labels.add(jp_label)
                            print(
                                f"[WARN] Unknown JP label (no mapping): '{jp_label}' "
                                f"(zip: {inner_zip_path}, member: {inner_name})"
                            )
                            continue
                        per_label_items[label].append((inner_zip_id, inner_name, jp_label))
                    else:
                        jp_topic = _choose_topic(outer_jp_label, inner_name)
                        out_label = _topic_to_label(jp_topic)
                        if out_label is None:
                            unknown_labels.add(_norm_topic(jp_topic))
                            continue
                        per_label_items[out_label].append((inner_zip_id, inner_name, jp_topic))
        elif outer_zip.is_dir():
            for inner_zip_path in inner_members:
                inner_zip_id = str(inner_zip_path)
                outer_jp_label = choose_label_from_inner(inner_zip_id)
                per_jp_label_counts[outer_jp_label] += 1
                label = "__topic_per_image__" if label_mode == "topic" else None

                with zipfile.ZipFile(inner_zip_path, "r") as inner:
                    inner_names = [safe_filename(n) for n in inner.namelist()]
                    for inner_name in inner_names:
                        if inner_name.endswith("/"):
                            continue
                        if not is_image_file(inner_name):
                            continue
                        if label_mode == "mapped":
                            jp_label = _choose_mapped_jp_label(outer_jp_label, inner_name)
                            label = jp_to_en.get(jp_label)
                            if label is None:
                                unknown_labels.add(jp_label)
                                print(
                                    f"[WARN] Unknown JP label (no mapping): '{jp_label}' "
                                    f"(zip: {inner_zip_path}, member: {inner_name})"
                                )
                                continue
                            per_label_items[label].append((inner_zip_id, inner_name, jp_label))
                        else:
                            jp_topic = _choose_topic(outer_jp_label, inner_name)
                            out_label = _topic_to_label(jp_topic)
                            if out_label is None:
                                unknown_labels.add(_norm_topic(jp_topic))
                                continue
                            per_label_items[out_label].append((inner_zip_id, inner_name, jp_topic))
        else:
            with zipfile.ZipFile(outer_zip, "r") as outer:
                for inner_zip_member in inner_members:
                    inner_zip_id = inner_zip_member
                    outer_jp_label = choose_label_from_inner(inner_zip_member)
                    per_jp_label_counts[outer_jp_label] += 1
                    label = "__topic_per_image__" if label_mode == "topic" else None

                    inner_bytes = outer.read(inner_zip_member)
                    with zipfile.ZipFile(io.BytesIO(inner_bytes), "r") as inner:
                        inner_names = [safe_filename(n) for n in inner.namelist()]
                        for inner_name in inner_names:
                            if inner_name.endswith("/"):
                                continue
                            if not is_image_file(inner_name):
                                continue
                            if label_mode == "mapped":
                                jp_label = _choose_mapped_jp_label(outer_jp_label, inner_name)
                                label = jp_to_en.get(jp_label)
                                if label is None:
                                    unknown_labels.add(jp_label)
                                    print(
                                        f"[WARN] Unknown JP label (no mapping): '{jp_label}' "
                                        f"(member: {inner_zip_member}, inner: {inner_name})"
                                    )
                                    continue
                                per_label_items[label].append((inner_zip_id, inner_name, jp_label))
                            else:
                                jp_topic = _choose_topic(outer_jp_label, inner_name)
                                out_label = _topic_to_label(jp_topic)
                                if out_label is None:
                                    unknown_labels.add(_norm_topic(jp_topic))
                                    continue
                                per_label_items[out_label].append((inner_zip_id, inner_name, jp_topic))

        # Second pass: split per label, then actually extract.
        for label, items in per_label_items.items():
            if not items:
                continue

            rng = random.Random(args.seed + (hash(label) % 1_000_000_007))
            rng.shuffle(items)

            if args.limit_per_class > 0:
                items = items[: args.limit_per_class]

            if args.make_splits:
                n = len(items)
                n_train = int(n * args.train_ratio)
                n_valid = int(n * args.valid_ratio)
                # Put remainder into test to keep total exact.
                n_test = n - n_train - n_valid

                # Avoid empty splits when there are enough samples.
                if n >= 3:
                    if n_train == 0:
                        n_train = 1
                        n_test = max(0, n - n_train - n_valid)
                    if n_valid == 0:
                        n_valid = 1
                        n_test = max(0, n - n_train - n_valid)
                    if n_test == 0:
                        n_test = 1
                        # Borrow from train first, then valid.
                        if n_train > 1:
                            n_train -= 1
                        elif n_valid > 1:
                            n_valid -= 1

                split_slices = {
                    "train": items[:n_train],
                    args.valid_name: items[n_train : n_train + n_valid],
                    "test": items[n_train + n_valid :],
                }
            else:
                split_slices = {split: items}

            # Extract for each split slice.
            for split_name, split_items in split_slices.items():
                if not split_items:
                    continue

                # Group by inner zip member to avoid re-reading the same inner zip repeatedly.
                by_inner_zip: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
                # (inner_member_name, jp_label)
                for inner_zip_id, inner_member, jp_label in split_items:
                    by_inner_zip[inner_zip_id].append((inner_member, jp_label))

                if single_inner_zip_mode or outer_zip.is_dir():
                    for inner_zip_id, members in by_inner_zip.items():
                        inner_zip_path = Path(inner_zip_id)
                        with zipfile.ZipFile(inner_zip_path, "r") as inner:
                            for inner_member, jp_label in members:
                                filename = Path(inner_member).name
                                dest_dir = split_roots[split_name] / label
                                dest_path = dest_dir / filename

                                if not args.dry_run:
                                    dest_dir.mkdir(parents=True, exist_ok=True)
                                    if dest_path.exists() and not args.overwrite:
                                        dest_path = unique_dest_path(dest_dir, filename)

                                    with inner.open(inner_member, "r") as src, open(dest_path, "wb") as dst:
                                        dst.write(src.read())

                                total_images += 1
                                per_label_counts_by_split[split_name][label] += 1

                                w = manifest_writers.get(split_name)
                                if w is not None:
                                    w.writerow(
                                        [
                                            str(outer_zip),
                                            str(inner_zip_path),
                                            inner_member,
                                            choose_label_from_inner(str(inner_zip_path)),
                                            jp_label,
                                            label,
                                            split_name,
                                            str(dest_path),
                                        ]
                                    )
                else:
                    with zipfile.ZipFile(outer_zip, "r") as outer:
                        for inner_zip_member, members in by_inner_zip.items():
                            inner_bytes = outer.read(inner_zip_member)
                            with zipfile.ZipFile(io.BytesIO(inner_bytes), "r") as inner:
                                for inner_member, jp_label in members:
                                    filename = Path(inner_member).name
                                    dest_dir = split_roots[split_name] / label
                                    dest_path = dest_dir / filename

                                    if not args.dry_run:
                                        dest_dir.mkdir(parents=True, exist_ok=True)
                                        if dest_path.exists() and not args.overwrite:
                                            dest_path = unique_dest_path(dest_dir, filename)

                                        with inner.open(inner_member, "r") as src, open(dest_path, "wb") as dst:
                                            dst.write(src.read())

                                    total_images += 1
                                    per_label_counts_by_split[split_name][label] += 1

                                    w = manifest_writers.get(split_name)
                                    if w is not None:
                                        w.writerow(
                                            [
                                                str(outer_zip),
                                                inner_zip_member,
                                                inner_member,
                                                choose_label_from_inner(inner_zip_member),
                                                jp_label,
                                                label,
                                                split_name,
                                                str(dest_path),
                                            ]
                                        )

        print("")
        print("Done.")
        print(f"Total extracted images: {total_images}")
        for s in split_names:
            print(f"Per label counts ({s}):")
            for k, v in per_label_counts_by_split[s].most_common():
                print(f"  {k}: {v}")
        if unknown_labels:
            print("")
            print("[WARN] Unmapped JP labels (skipped):")
            for x in sorted(unknown_labels):
                print(f"  - {x}")
    finally:
        for s in list(manifest_files.keys()):
            try:
                manifest_files[s].close()
            except Exception:
                pass
        if not args.dry_run:
            for s in split_names:
                print(f"Wrote manifest: {manifest_paths[s]}")


if __name__ == "__main__":
    main()

