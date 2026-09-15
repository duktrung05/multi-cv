"""Validate image annotations and detect train/validation/test leakage."""
import argparse
import hashlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.domain.inspection import DEFECT_LABELS
from src.domain.inspection_annotations import read_annotations


def validate(root, class_list):
    from PIL import Image
    root = Path(root)
    seen, report = {}, {}
    for split in ("train", "valid", "test"):
        samples = read_annotations(root / split, root / f"{split}.csv", class_list)
        counts = dict.fromkeys(class_list, 0)
        normal = 0
        for path, target in samples:
            with Image.open(path) as image:
                image.verify()
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if digest in seen:
                raise ValueError(f"Duplicate image content: {path} and {seen[digest]}")
            seen[digest] = path
            normal += int(not any(target))
            for label, active in zip(class_list, target):
                counts[label] += int(active)
        report[split] = {"images": len(samples), "defect_free": normal, "labels": counts,
                         "missing_labels": [k for k, n in counts.items() if n == 0]}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/product_inspection")
    parser.add_argument("--labels", nargs="+", default=list(DEFECT_LABELS))
    args = parser.parse_args()
    import json
    print(json.dumps(validate(args.root, args.labels), indent=2))


if __name__ == "__main__":
    main()
