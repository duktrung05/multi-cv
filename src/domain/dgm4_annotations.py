"""Validate DGM4 annotations without importing ML or detection dependencies."""
import csv
import json
from pathlib import Path

CLASS_NAMES = ("REAL", "AI_EDITED")


def read_manifest(root):
    root = Path(root).resolve()
    if (root / "INCOMPLETE.txt").exists():
        raise ValueError("DGM4 dataset build is incomplete")
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if report.get("complete") is not True:
        raise ValueError("DGM4 report must confirm a complete dataset")
    with (root / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    seen, owners = set(), {}
    for row in rows:
        split, label = row["split"], row["label"]
        if split not in {"train", "val", "test"} or label not in CLASS_NAMES:
            raise ValueError("Invalid DGM4 split or label")
        path = (root / row["image"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or unsafe image path: {row['image']}")
        relative = path.relative_to(root)
        if relative.parts[:2] != (split, label.lower()):
            raise ValueError(f"Folder/manifest label mismatch: {row['image']}")
        if path in seen:
            raise ValueError(f"Duplicate manifest path: {row['image']}")
        seen.add(path)
        for key in ("source_id", "source_image", "pixel_sha256"):
            value = row.get(key)
            if not value:
                raise ValueError(f"Missing provenance: {key}")
            identity = (key, value)
            if identity in owners and owners[identity] != split:
                raise ValueError(f"Cross-split leakage: {key}={value}")
            owners[identity] = split
        method = row.get("method")
        fake_cls = row.get("fake_cls", "").split("&")
        if label == "REAL" and (method != "original" or fake_cls != ["orig"]):
            raise ValueError("REAL must come from an original image")
        if label == "AI_EDITED" and (method not in {"simswap", "StyleCLIP"}
                or not set(fake_cls) & {"face_swap", "face_attribute"}
                or not json.loads(row.get("fake_image_box", "[]"))):
            raise ValueError("AI_EDITED must have an annotated face manipulation")
    if len(rows) != report.get("total"):
        raise ValueError("Manifest count differs from build report")
    return rows

