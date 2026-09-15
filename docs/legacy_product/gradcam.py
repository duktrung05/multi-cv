#!/usr/bin/env python3
"""Grad-CAM heatmaps for the IAS multilabel classifier.

The model concatenates pooled features from all three backbone scales
(c2=1:8, c3=1:16, c4=1:32; see ``engine/backbone/dinov3_adapter.py``) before the
final head, so a class logit depends on all three jointly. This hooks all three in
a single forward/backward pass, builds a CAM per scale, upsamples each to a common
size, and averages them into one combined heatmap (individual per-scale heatmaps
are still available for comparison). Useful for checking *where* the model looks
when it fires MALE_SEXUAL vs FEMALE/FEMALE_GENITALIA on the same image.

Usage
-----
    python gradcam.py --image test_data/cant_detect/アナル.jpg \
        --classes MALE_SEXUAL FEMALE FEMALE_GENITALIA \
        --out-dir outputs/gradcam

    # Single scale instead of the combined heatmap:
    python gradcam.py --image img.jpg --classes MALE_SEXUAL --scale c4

    # Compare a full checkpoint against another (e.g. before/after retraining):
    python gradcam.py --image img.jpg --classes MALE_SEXUAL FEMALE \
        --weights outputs/ias_multilabel_dinov3_vit_s_all/summary/<run>/best.pth
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from infer_vas import DEFAULT_CONFIG, DEFAULT_WEIGHTS, load_image_tensor, load_model


_SCALE_POS = {"c2": 0, "c3": 1, "c4": 2}
ALL_SCALES = ("c2", "c3", "c4")


def _compute_gradcam_multiscale(
    model, tensor: torch.Tensor, target_idx: int, scales: tuple[str, ...] = ALL_SCALES
) -> tuple[dict[str, np.ndarray], float]:
    """One forward/backward pass; returns a per-scale CAM (each individually normalized to [0, 1])."""
    positions = [_SCALE_POS[s] for s in scales]
    activations: dict[int, torch.Tensor] = {}
    gradients: dict[int, torch.Tensor] = {}

    def fwd_hook(_module, _input, output):
        for pos in positions:
            act = output[pos]
            activations[pos] = act
            act.register_hook(lambda grad, pos=pos: gradients.__setitem__(pos, grad))

    handle = model.backbone.register_forward_hook(fwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        score = logits[0, target_idx]
        score.backward()
    finally:
        handle.remove()

    cams: dict[str, np.ndarray] = {}
    for s, pos in zip(scales, positions):
        act = activations[pos][0]  # [C, H, W]
        grad = gradients[pos][0]  # [C, H, W]
        weights = grad.mean(dim=(1, 2))  # [C]
        cam = F.relu((weights[:, None, None] * act).sum(0))  # [H, W]
        cam = cam / cam.max().clamp_min(1e-8)
        cams[s] = cam.detach().cpu().numpy()
    return cams, float(torch.sigmoid(score).item())


def _resize_cam(cam: np.ndarray, size: list[int]) -> np.ndarray:
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((size[1], size[0]), Image.BILINEAR)
    return np.array(cam_img, dtype=np.float32) / 255.0


def _combine_cams(cams: dict[str, np.ndarray], size: list[int]) -> np.ndarray:
    """Upsample each per-scale CAM to a common size and average (each scale contributes equally)."""
    stacked = np.stack([_resize_cam(cam, size) for cam in cams.values()], axis=0)
    combined = stacked.mean(axis=0)
    return combined / max(combined.max(), 1e-8)


def _compute_gradcam(
    model, tensor: torch.Tensor, target_idx: int, scale: str
) -> tuple[np.ndarray, float]:
    """Single-scale or combined ('combined' = mean of c2/c3/c4) CAM for one target class."""
    scales = ALL_SCALES if scale == "combined" else (scale,)
    cams, prob = _compute_gradcam_multiscale(model, tensor, target_idx, scales)
    cam = _combine_cams(cams, list(next(iter(cams.values())).shape)) if scale == "combined" else cams[scale]
    return cam, prob


def _overlay(image_path: str, cam: np.ndarray, size: list[int], alpha: float = 0.45) -> Image.Image:
    img = Image.open(image_path).convert("RGB").resize((size[1], size[0]))
    heatmap_norm = _resize_cam(cam, size) if cam.shape != tuple(size) else cam
    heatmap = np.uint8(cm.jet(heatmap_norm)[:, :, :3] * 255)
    blended = np.uint8((1 - alpha) * np.array(img) + alpha * heatmap)
    return Image.fromarray(blended)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM heatmaps for the IAS multilabel classifier")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--classes", nargs="+", required=True, help="Target class name(s), e.g. MALE_SEXUAL FEMALE")
    parser.add_argument(
        "--scale",
        default="combined",
        choices=["combined", "c2", "c3", "c4"],
        help="'combined' averages c2+c3+c4 (matches what the head actually sees); "
        "or pick a single scale (c4=coarsest/most semantic, c2=finest)",
    )
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--out-dir", default="outputs/gradcam")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    loaded = load_model(config=args.config, weights=args.weights, device=args.device)
    model = loaded.model
    class_list = loaded.class_list

    missing = [c for c in args.classes if c not in class_list]
    if missing:
        raise SystemExit(f"unknown class(es) {missing}; available: {class_list}")

    image_path = args.image if os.path.isabs(args.image) else os.path.join(loaded.repo_root, args.image)
    tensor = load_image_tensor(image_path, loaded.transform).unsqueeze(0).to(loaded.device)
    tensor.requires_grad_(False)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]

    for cls in args.classes:
        idx = class_list.index(cls)
        cam, prob = _compute_gradcam(model, tensor, idx, args.scale)
        overlay = _overlay(image_path, cam, loaded.size)
        out_path = os.path.join(args.out_dir, f"{stem}__{cls}__p{prob:.2f}__{args.scale}.png")
        overlay.save(out_path)
        print(f"{cls:20s} prob={prob:.4f} scale={args.scale} -> {out_path}")


if __name__ == "__main__":
    main()
