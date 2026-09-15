"""Re-read all selected images, verify provenance, and write a reviewable audit."""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PIL import Image, ImageOps
from src.domain.dgm4_annotations import read_manifest


def render_candidates(root, groups, output):
    if not groups:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    columns = max(map(len, groups))
    fig, axes = plt.subplots(len(groups), columns, figsize=(5 * columns, 3 * len(groups)), squeeze=False)
    for index, group in enumerate(groups):
        for j, ax in enumerate(axes[index]):
            ax.axis("off")
            if j < len(group):
                with Image.open(Path(root) / group[j]) as source:
                    ax.imshow(ImageOps.exif_transpose(source).convert("RGB"))
                ax.set_title(f"Group {index+1}: {group[j]}", fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(output) / "near_duplicate_candidates.png", dpi=100)
    plt.close(fig)


def audit(root, output, expected=(8000, 1000, 1000)):
    root, output = Path(root), Path(output)
    rows = read_manifest(root)
    counts = Counter(row["split"] for row in rows)
    if counts != dict(zip(("train", "val", "test"), expected)):
        raise ValueError(f"Unexpected split sizes: {counts}")
    output.mkdir(parents=True, exist_ok=True)
    pixels, dhashes = {}, defaultdict(list)
    errors, duplicates = [], []
    for i, row in enumerate(rows):
        try:
            with Image.open(root / row["image"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                if digest != row["pixel_sha256"]:
                    errors.append({"image": row["image"], "error": "Pixel hash differs from manifest"})
                if image.size != (int(row["width"]), int(row["height"])):
                    errors.append({"image": row["image"], "error": "Dimensions differ from manifest"})
                if digest in pixels:
                    duplicates.append([pixels[digest], row["image"]])
                pixels[digest] = row["image"]
                gray = list(image.convert("L").resize((9, 8)).getdata())
                bits = ''.join('1' if gray[y*9+x] > gray[y*9+x+1] else '0'
                               for y in range(8) for x in range(8))
                dhashes[bits].append(row)
        except (OSError, ValueError) as exc:
            errors.append({"image": row["image"], "error": str(exc)})
        if (i + 1) % 1000 == 0:
            print(f"Audited {i + 1}/{len(rows)}", flush=True)
    suspicious = [[r["image"] for r in group] for group in dhashes.values()
                  if len({r["split"] for r in group}) > 1]
    report = {
        "complete": not errors and not duplicates,
        "ready_for_training": not errors and not duplicates and not suspicious,
        "total": len(rows),
        "counts": dict(Counter(r["split"] + '/' + r["label"] for r in rows)),
        "methods": dict(Counter(r["split"] + '/' + r["method"] for r in rows)),
        "sources": dict(Counter(r["split"] + '/' + r["source_image"].split('/')[2] for r in rows)),
        "errors": errors, "duplicate_pixels": duplicates,
        "cross_split_identical_dhash_groups": suspicious,
        "near_duplicate_limit": "Exact 64-bit dHash candidates only; neither proof of duplication nor exhaustive near-duplicate detection.",
        "manifest_sha256": hashlib.sha256((root / 'manifest.csv').read_bytes()).hexdigest(),
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # A deterministic sample across splits, labels, and manipulation methods.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    samples = []
    for split in ("train", "val", "test"):
        for method in ("original", "simswap", "StyleCLIP"):
            samples.extend([r for r in rows if r["split"] == split and r["method"] == method][:2])
    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    for ax, row in zip(axes.flat, samples):
        with Image.open(root / row["image"]) as source:
            ax.imshow(ImageOps.exif_transpose(source).convert("RGB"))
        ax.set_title(f"{row['split']} / {row['label']}\n{row['method']} / id {row['source_id']}", fontsize=8)
        ax.axis("off")
    for ax in list(axes.flat)[len(samples):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "samples.png", dpi=130)
    plt.close(fig)
    render_candidates(root, suspicious, output)
    print(json.dumps({k: v for k, v in report.items() if k != "cross_split_identical_dhash_groups"}, indent=2))
    print("Cross-split dHash candidate groups:", len(suspicious))
    if errors or duplicates:
        raise ValueError("Dataset audit failed; inspect audit.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/dgm4_binary")
    parser.add_argument("--output", default="outputs/dgm4_binary/audit")
    args = parser.parse_args()
    audit(args.root, args.output)
