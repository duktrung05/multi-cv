"""Shared preprocessing and strict checkpoint loading for the DGM4 baseline."""
import hashlib
import math
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageOps
from torchvision import transforms

from engine.backbone.dgm4_classifier import DINOv3BinaryClassifier
from engine.data.dataset.dgm4_binary import CLASS_NAMES
from src.shared.exceptions import ModelLoadError

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = "configs/dgm4_binary.yml"
DEFAULT_WEIGHTS = "ckpts/dgm4_binary.pth"
MODEL_SPEC_KEYS = ("backbone", "feature_pooling", "image_size", "mean", "std")


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path=DEFAULT_CONFIG, overrides=None):
    from engine.core.yaml_utils import merge_dict, parse_cli
    cfg = yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))
    merge_dict(cfg, parse_cli(overrides))
    if cfg.get("task_domain") != "dgm4_binary" or cfg.get("class_list") != list(CLASS_NAMES) or cfg.get("num_classes") != 2:
        raise ValueError("DGM4 requires the exact class order [REAL, AI_EDITED]")
    if cfg.get("backbone") != "dinov3_vits16" or cfg.get("feature_pooling") != "cls_and_mean_patch" or cfg.get("freeze_backbone") is not True:
        raise ValueError("This baseline supports frozen DINOv3 ViT-S/16 with CLS + mean patch pooling")
    if not isinstance(cfg["image_size"], int) or cfg["image_size"] < 32 or cfg["image_size"] % 16:
        raise ValueError("image_size must be a multiple of 16 and at least 32")
    if len(cfg["mean"]) != 3 or len(cfg["std"]) != 3 or any(float(s) <= 0 for s in cfg["std"]):
        raise ValueError("Expected RGB mean and positive std")
    if not 0 < float(cfg.get("threshold", .5)) < 1:
        raise ValueError("threshold must be between 0 and 1")
    return cfg


def make_transform(cfg):
    return transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"]), antialias=True),
        transforms.ToTensor(), transforms.Normalize(cfg["mean"], cfg["std"]),
    ])


def resolve_device(device):
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result = torch.device(device)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; install the CUDA PyTorch build or use device=cpu")
    return result


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_spec(cfg):
    return {key: cfg[key] for key in MODEL_SPEC_KEYS}


def validate_checkpoint(state, cfg, allow_smoke=False):
    if not isinstance(state, dict) or state.get("task_domain") != "dgm4_binary" or state.get("class_list") != list(CLASS_NAMES):
        raise ModelLoadError("Checkpoint domain/label order must be dgm4_binary [REAL, AI_EDITED]")
    if state.get("model_spec") != model_spec(cfg):
        raise ModelLoadError("Checkpoint architecture/preprocessing differs from config")
    if not allow_smoke and (state.get("smoke_only") or not state.get("pretrained_sha256")):
        raise ModelLoadError("Synthetic/unverified checkpoint cannot be used by the application")


def result_from_scores(scores, threshold=.5):
    if len(scores) != 2 or any(not math.isfinite(float(p)) or not 0 <= p <= 1 for p in scores):
        raise ValueError("Expected two finite softmax scores")
    if not math.isclose(sum(scores), 1.0, abs_tol=1e-5) or not 0 < threshold < 1:
        raise ValueError("Invalid softmax scores or threshold")
    index = int(scores[1] >= threshold)
    return {"label": CLASS_NAMES[index], "score": float(scores[index]),
            "scores": dict(zip(CLASS_NAMES, map(float, scores))),
            "ai_edited_threshold": threshold, "task_domain": "dgm4_binary"}


class DGM4Predictor:
    def __init__(self, model, cfg, device):
        self.model = model.to(device).eval()
        self.device, self.transform = device, make_transform(cfg)
        self.threshold = float(cfg.get("threshold", .5))
        self.class_names = list(CLASS_NAMES)

    @classmethod
    def load(cls, config=DEFAULT_CONFIG, checkpoint=DEFAULT_WEIGHTS, device="auto"):
        cfg = load_config(config)
        path = resolve_path(checkpoint)
        if not path.is_file():
            raise ModelLoadError(f"Missing trained DGM4 checkpoint: {path}. Train and select best.pth first.")
        state = torch.load(path, map_location="cpu", weights_only=True)
        validate_checkpoint(state, cfg)
        model = DINOv3BinaryClassifier()
        model.load_state_dict(state["model"], strict=True)
        return cls(model, cfg, resolve_device(device))

    def predict(self, image_path):
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            if logits.shape != (1, 2):
                raise ValueError("Model must return [1, 2] logits")
            scores = logits.float().softmax(dim=-1)[0].cpu().tolist()
        return result_from_scores(scores, self.threshold)
