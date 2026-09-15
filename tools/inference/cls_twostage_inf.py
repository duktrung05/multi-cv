"""
Two-stage ReaS classification: coarse classifier + NSFW second stage (.pt or .engine) for NSFW_EXPLICIT.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from engine.core import YAMLConfig  # noqa: E402
from engine.misc.nsfw_engine_refine import DETAIL_CLASS_NAMES, NSFW_REFINE_INPUT_SIZE, build_nsfw_refiner  # noqa: E402


def _load_classifier_state_dict(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "ema" in ckpt and isinstance(ckpt["ema"], dict):
        return ckpt["ema"].get("module", ckpt["ema"])
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def main():
    parser = argparse.ArgumentParser(description="ReaS coarse + NSFW.engine detail inference")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True, help="checkpoint .pth (model or ema)")
    parser.add_argument(
        "--nsfw-engine",
        type=str,
        required=True,
        help="ckpts/nsfw.pt (PyTorch) or ckpts/nsfw.engine (TensorRT)",
    )
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    args = parser.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(_REPO_ROOT, args.config)
    cfg = YAMLConfig(cfg_path)

    resume = args.resume if os.path.isabs(args.resume) else os.path.join(_REPO_ROOT, args.resume)
    nsfw_engine = (
        args.nsfw_engine
        if os.path.isabs(args.nsfw_engine)
        else os.path.join(_REPO_ROOT, args.nsfw_engine)
    )
    inp = args.input if os.path.isabs(args.input) else os.path.join(_REPO_ROOT, args.input)

    state = _load_classifier_state_dict(resume)
    cfg.model.load_state_dict(state, strict=True)
    model = cfg.model.to(args.device).eval()

    ds_cfg = (((cfg.yaml_cfg or {}).get("val_dataloader") or {}).get("dataset") or {})
    class_list = list(ds_cfg.get("class_list") or [])
    if "NSFW_EXPLICIT" not in class_list:
        raise ValueError("val_dataloader.dataset.class_list must contain NSFW_EXPLICIT")

    merged_idx = class_list.index("NSFW_EXPLICIT")

    h, w = NSFW_REFINE_INPUT_SIZE
    tfm = transforms.Compose(
        [
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    im_pil = Image.open(inp).convert("RGB")
    batch = tfm(im_pil).unsqueeze(0).to(args.device)

    with torch.no_grad():
        logits = model(batch)
        pred = int(logits.argmax(dim=-1).item())

    coarse_name = class_list[pred]
    print(f"coarse_idx={pred} coarse={coarse_name}")

    if coarse_name == "NSFW_EXPLICIT":
        refiner = build_nsfw_refiner(nsfw_engine, torch.device(args.device))
        probs = refiner.forward_probs(batch)
        di = int(refiner.detail_indices_from_probs(probs)[0].item())
        refiner.synchronize()
        print(f"detail_idx={di} detail={DETAIL_CLASS_NAMES[di]}")


if __name__ == "__main__":
    main()
