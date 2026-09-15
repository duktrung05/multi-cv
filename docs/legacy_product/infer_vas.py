#!/usr/bin/env python3
"""Run a trained VAS multilabel classifier on one or more images.

Loads the model architecture + ``class_list`` from a training config (same
config used by ``train_vas.py``), loads a checkpoint's ``model`` state dict
on top of it, then runs sigmoid multilabel inference (see
``src/infrastructure/ml/multilabel_predict.py``) and prints the classes
whose probability is >= threshold.

Usage
-----
    python infer_vas.py --image path/to/image.jpg
    python infer_vas.py --image img1.jpg img2.jpg --threshold 0.5
    python infer_vas.py --image img.jpg \
        --config configs/deimv2/vas_dinov3_vit_s_multilabel_all.yml \
        --weights outputs/vas_multilabel_dinov3_vit_s_all/summary/multilabel_classification_20260701_094609/best.pth

    # Batch mode: predict every row of a CSV (columns: image/orig_path, body_parts
    # ground-truth same as data/reas_imagefolder_sub4cls/test.csv), write a
    # `pred_results` column, and print macro-F1 against the ground truth.
    python infer_vas.py --csv data/reas_imagefolder_sub4cls/test.csv --output test_pred.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLBACKEND", "Agg")

import copy

import torch
from PIL import Image

from train_ias import initialize_training_context
from src.infrastructure.ml.multilabel_predict import predict_multilabel
from engine.data.dataset.reas_multilabel_imagefolder import _parse_labels_cell
from engine.data.transforms.container import Compose

DEFAULT_CONFIG = "configs/deimv2/ias_dinov3_vit_s_multilabel_all.yml"
DEFAULT_WEIGHTS = (
    # "outputs/ias_multilabel_dinov3_vit_s_all/summary/multilabel_classification_20260708_073955/best.pth"
    # "outputs/ias_multilabel_dinov3_vit_s_all/summary/multilabel_classification_20260728_100108/best.pth"
    # "outputs/ias_multilabel_dinov3_vit_s_all/summary/multilabel_classification_20260728_120905/best.pth"
    "outputs/ias_multilabel_dinov3_vit_s_all/summary/multilabel_classification_20260729_061317/best.pth"
)


def _rp(repo_root: str, path: str) -> str:
    p = (path or "").strip()
    return path if os.path.isabs(p) else os.path.join(repo_root, p)


def _build_val_transform(cfg) -> Compose:
    """Build the exact same eval-time transform Compose used by val_dataloader.

    Reusing the registered Compose (instead of hand-rolling a resize/normalize
    pair) means inference automatically stays in sync with whatever ops the
    training config uses (e.g. Letterbox vs plain Resize).
    """
    tspec = (((cfg.yaml_cfg.get("val_dataloader") or {}).get("dataset") or {}).get("transforms")) or {}
    ops = copy.deepcopy(tspec.get("ops") or [])
    return Compose(ops=ops, policy=tspec.get("policy"))


def _val_output_size(cfg) -> list[int]:
    """The [H, W] the val transform pipeline resizes/letterboxes images to."""
    ops = (((cfg.yaml_cfg.get("val_dataloader") or {}).get("dataset") or {}).get("transforms") or {}).get(
        "ops", []
    )
    size = [384, 384]
    for op in ops:
        if op.get("type") in ("Letterbox", "Resize") and op.get("size"):
            size = list(op["size"])
    return size


def load_image_tensor(image_path: str, transform: Compose) -> torch.Tensor:
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return transform(img)


class LoadedModel:
    """Bundle of everything needed to run inference, reused by the CLI and the predict API."""

    def __init__(self, model, class_list, threshold, transform, size, device, repo_root):
        self.model = model
        self.class_list = class_list
        self.threshold = threshold
        self.transform = transform
        self.size = size
        self.device = device
        self.repo_root = repo_root

    def predict(self, image_path: str) -> tuple[list[float], list[str]]:
        image_path = _rp(self.repo_root, image_path)
        tensor = load_image_tensor(image_path, self.transform).unsqueeze(0).to(self.device)
        probs, active = predict_multilabel(self.model, tensor, class_names=self.class_list, threshold=self.threshold)
        return probs[0].tolist(), active[0]


def load_model(
    config: str = DEFAULT_CONFIG,
    weights: str = DEFAULT_WEIGHTS,
    threshold: float | None = None,
    device: str | None = None,
) -> LoadedModel:
    repo_root, cfg = initialize_training_context(argparse.Namespace(config=config, update=None, local_rank=None))

    weights_path = _rp(repo_root, weights)
    if not os.path.isfile(weights_path):
        raise SystemExit(f"weights not found: {weights_path}")

    resolved_device = torch.device(
        device or cfg.yaml_cfg.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    model = cfg.model
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(resolved_device).eval()

    class_list = list(
        (cfg.yaml_cfg.get("train_dataloader") or {}).get("dataset", {}).get("class_list")
        or cfg.yaml_cfg.get("class_list")
        or []
    )
    resolved_threshold = float(threshold if threshold is not None else cfg.yaml_cfg.get("multilabel_threshold", 0.35))
    transform = _build_val_transform(cfg)
    size = _val_output_size(cfg)

    return LoadedModel(model, class_list, resolved_threshold, transform, size, resolved_device, repo_root)


def _resolve_csv_image_path(row, repo_root: str, image_column: str) -> str:
    orig_path = str(row.get("orig_path") or "").strip()
    if orig_path and os.path.isfile(orig_path):
        return orig_path
    image = str(row.get(image_column) or "").strip()
    return _rp(repo_root, os.path.join("raw_data", image))


def _macro_f1(pred_bin: torch.Tensor, target: torch.Tensor) -> float:
    tp = (pred_bin * target).sum(dim=0)
    fp = (pred_bin * (1 - target)).sum(dim=0)
    fn = ((1 - pred_bin) * target).sum(dim=0)
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-12)
    return float(f1.mean().item())


def run_csv_predict(
    loaded: LoadedModel,
    csv_path: str,
    output_path: str,
    *,
    image_column: str = "image",
    labels_column: str = "body_parts",
    labels_format: str = "label_studio_choices",
) -> None:
    import pandas as pd

    csv_path = _rp(loaded.repo_root, csv_path)
    df = pd.read_csv(csv_path)

    class_list = loaded.class_list
    pred_results: list[str] = []
    pred_rows: list[list[float]] = []
    target_rows: list[list[float]] = []
    have_targets = labels_column in df.columns

    for i, row in df.iterrows():
        image_path = _resolve_csv_image_path(row, loaded.repo_root, image_column)
        if not os.path.isfile(image_path):
            print(f"[skip] not found: {image_path}")
            pred_results.append("")
            continue

        probs, _ = loaded.predict(image_path)
        pred_results.append(json.dumps({name: round(p, 4) for name, p in zip(class_list, probs)}))
        pred_rows.append([1.0 if p >= loaded.threshold else 0.0 for p in probs])

        if have_targets:
            active = _parse_labels_cell(str(row.get(labels_column) or ""), labels_format)
            target_rows.append([1.0 if name in active else 0.0 for name in class_list])

        if (i + 1) % 20 == 0 or (i + 1) == len(df):
            print(f"  predicted {i + 1}/{len(df)}")

    df["pred_results"] = pred_results
    df.to_csv(output_path, index=False)
    print(f"\nWrote predictions to {output_path}")

    if have_targets and target_rows:
        macro_f1 = _macro_f1(torch.tensor(pred_rows), torch.tensor(target_rows))
        print(f"macro_f1 = {macro_f1:.4f}  (n={len(target_rows)}, threshold={loaded.threshold})")
    else:
        print(f"macro_f1 skipped: no '{labels_column}' ground-truth column found in {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VAS multilabel inference on one or more images")
    parser.add_argument("--image", nargs="+", help="Path(s) to image file(s)")
    parser.add_argument("--csv", help="CSV with columns image/orig_path (+ body_parts ground truth) to batch-predict")
    parser.add_argument("--output", help="Output CSV path for --csv mode (default: <csv>_pred.csv)")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--threshold", type=float, default=None, help="Overrides multilabel_threshold in config")
    parser.add_argument("--device", default=None, help="e.g. cuda:0 or cpu (defaults to config device)")
    args = parser.parse_args()
    if not args.image and not args.csv:
        raise SystemExit("must pass --image or --csv")

    loaded = load_model(config=args.config, weights=args.weights, threshold=args.threshold, device=args.device)
    class_list = loaded.class_list
    threshold = loaded.threshold

    if args.csv:
        csv_path = _rp(loaded.repo_root, args.csv)
        output_path = args.output or f"{os.path.splitext(csv_path)[0]}_pred.csv"
        run_csv_predict(loaded, csv_path, output_path)
        return

    for image_path in args.image:
        resolved_path = _rp(loaded.repo_root, image_path)
        if not os.path.isfile(resolved_path):
            print(f"[skip] not found: {resolved_path}")
            continue

        probs, detected = loaded.predict(resolved_path)

        print(f"\n{resolved_path}")
        print(f"  detected classes (threshold={threshold}): {detected or '(none)'}")
        ranked = sorted(zip(class_list, probs), key=lambda x: x[1], reverse=True)
        for name, p in ranked:
            marker = "*" if p >= threshold else " "
            print(f"  {marker} {name:<20s} {p:.4f}")


if __name__ == "__main__":
    main()
