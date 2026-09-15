"""Strict CSV contract. Empty JSON list means inspected and defect-free."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def read_annotations(root, annotations_path, class_list):
    root = Path(root).resolve()
    if not class_list or len(set(class_list)) != len(class_list):
        raise ValueError("class_list must contain unique defect names")
    samples, seen = [], set()
    with open(annotations_path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"image", "labels"} <= set(reader.fieldnames or []):
            raise ValueError("CSV requires image,labels columns")
        for row_number, row in enumerate(reader, 2):
            try:
                relative = Path(row["image"])
                path = (root / relative).resolve()
                if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
                    raise ValueError(f"Image must exist inside dataset root: {relative}")
                if path in seen:
                    raise ValueError(f"Duplicate image: {relative}")
                labels = json.loads(row["labels"])
                if not isinstance(labels, list) or any(not isinstance(x, str) for x in labels):
                    raise ValueError("labels must be a JSON list of strings; use [] for defect-free")
                unknown = set(labels) - set(class_list)
                if unknown:
                    raise ValueError(f"Unknown labels: {sorted(unknown)}")
                if len(set(labels)) != len(labels):
                    raise ValueError("Duplicate labels")
                samples.append((str(path), [float(name in labels) for name in class_list]))
                seen.add(path)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{annotations_path}:{row_number}: {exc}") from exc
    if not samples:
        raise ValueError(f"No annotated images in {annotations_path}")
    return samples
