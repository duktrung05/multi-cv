"""
NSFW second-stage: TensorRT (.engine) or PyTorch (.pt / TorchScript) multi-label -> ReaS detail labels.

Classifier uses dataset imgsz (currently 384); NSFW torch path resizes normalized tensors to `input_size`.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

NSFW_REFINE_INPUT_SIZE: Tuple[int, int] = (384, 384)

# Label order must match exported NSFW model (`nsfw.pt` / `nsfw.engine`).
NSFW_ENGINE_LABELS: List[str] = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]

_I_FEMALE_GENITALIA_EXPOSED = 4
_I_MALE_GENITALIA_EXPOSED = 14

_VOTE_BUCKET: tuple[int, ...] = (
    0,
    2,
    1,
    1,
    0,
    1,
    0,
    2,
    2,
    2,
    2,
    1,
    2,
    1,
    0,
    0,
    2,
    2,
)

DETAIL_CLASS_NAMES: tuple[str, ...] = ("PRIVATE_PART", "NUDE", "SEDUCTIVE", "PORN")

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def _import_trt_inference():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from tools.inference.trt_inf import TRTInference

    return TRTInference


def detail_indices_from_probs(
    probs: torch.Tensor,
    tau_porn: float = 0.35,
    tau_low_vote: float = 0.08,
) -> torch.Tensor:
    if probs.dim() != 2 or probs.shape[-1] != len(NSFW_ENGINE_LABELS):
        raise ValueError(f"Expected probs [B,{len(NSFW_ENGINE_LABELS)}], got {tuple(probs.shape)}")
    device = probs.device
    B = probs.shape[0]
    out = torch.empty((B,), dtype=torch.long, device=device)
    for b in range(B):
        p = probs[b]
        dual = float(p[_I_FEMALE_GENITALIA_EXPOSED].item() + p[_I_MALE_GENITALIA_EXPOSED].item())
        if dual >= tau_porn:
            out[b] = 3
            continue
        v = p.new_zeros(3)
        for i, bucket in enumerate(_VOTE_BUCKET):
            if bucket < 3:
                v[bucket] = v[bucket] + p[i]
        best = int(v.argmax().item())
        if float(v.max().item()) < tau_low_vote:
            best = 2
        out[b] = best
    return out


def _forward_logits(model: Union[torch.nn.Module, torch.jit.ScriptModule], x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, dict):
        for k in ("logits", "pred_logits", "scores", "output"):
            if k in out and torch.is_tensor(out[k]):
                return out[k]
        raise RuntimeError(f"NSFW checkpoint dict has no tensor logits key; keys={list(out.keys())}")
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def _load_nsfw_torch_model(path: str, device: torch.device) -> Union[torch.nn.Module, torch.jit.ScriptModule]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    model = None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        ckpt = None

    if isinstance(ckpt, nn.Module):
        model = ckpt
    elif isinstance(ckpt, dict):
        for k in ("model", "net", "module"):
            v = ckpt.get(k)
            if isinstance(v, nn.Module):
                model = v
                break

    if model is None:
        try:
            model = torch.jit.load(path, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(
                "Could not load nsfw.pt as nn.Module dict or TorchScript. "
                "Expected torch.save(model), {'model': nn.Module}, or torch.jit.save(...)."
            ) from exc

    model.eval()
    model.to(device)
    return model


class NsfwTrtRefiner:
    """TensorRT .engine (optional)."""

    def __init__(
        self,
        engine_path: str,
        device: torch.device,
        tau_porn: float = 0.35,
        tau_low_vote: float = 0.08,
    ) -> None:
        TRTInference = _import_trt_inference()
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"NSFW TensorRT engine not found: {engine_path}")
        self._device = device
        self._tau_porn = tau_porn
        self._tau_low_vote = tau_low_vote
        self._trt = TRTInference(engine_path, device=str(device), max_batch_size=64)
        self._input_names = self._trt.input_names
        self._output_names = self._trt.output_names
        if not self._input_names or not self._output_names:
            raise RuntimeError("TensorRT engine missing input/output tensors")

        in_binding = self._trt.bindings[self._input_names[0]]
        self._in_shape = tuple(in_binding.shape)

        mean = _IMAGENET_MEAN.to(device).float().view(1, 3, 1, 1)
        std = _IMAGENET_STD.to(device).float().view(1, 3, 1, 1)
        self._mean = mean
        self._std = std

    def _preprocess_batch(self, imgs_bc_hw: torch.Tensor) -> torch.Tensor:
        x = imgs_bc_hw * self._std + self._mean
        x = x.clamp(0.0, 1.0)
        _, _, h, w = self._in_shape
        if h <= 0 or w <= 0:
            raise RuntimeError(f"Bad TRT input spatial dims: {self._in_shape}")
        x = F.interpolate(x, size=(int(h), int(w)), mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std
        return x.contiguous().float()

    def forward_probs(self, imgs_bc_hw: torch.Tensor) -> torch.Tensor:
        blob_in = self._preprocess_batch(imgs_bc_hw)
        name = self._input_names[0]
        raw = self._trt({name: blob_in})[self._output_names[0]]
        raw = raw.reshape(raw.shape[0], -1)
        if raw.shape[-1] != len(NSFW_ENGINE_LABELS):
            raise RuntimeError(
                f"NSFW output dim {raw.shape[-1]} != {len(NSFW_ENGINE_LABELS)}"
            )
        return torch.sigmoid(raw)

    def detail_indices_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        return detail_indices_from_probs(
            probs, tau_porn=self._tau_porn, tau_low_vote=self._tau_low_vote
        )

    def synchronize(self):
        self._trt.synchronize()


class NsfwTorchRefiner:
    """PyTorch weights (.pt): full module, {'model': Module}, or TorchScript."""

    def __init__(
        self,
        weights_path: str,
        device: torch.device,
        input_size: Tuple[int, int] = NSFW_REFINE_INPUT_SIZE,
        tau_porn: float = 0.35,
        tau_low_vote: float = 0.08,
    ) -> None:
        self._device = device
        self._h, self._w = int(input_size[0]), int(input_size[1])
        self._tau_porn = tau_porn
        self._tau_low_vote = tau_low_vote
        self._model = _load_nsfw_torch_model(weights_path, device)

        mean = _IMAGENET_MEAN.to(device).float().view(1, 3, 1, 1)
        std = _IMAGENET_STD.to(device).float().view(1, 3, 1, 1)
        self._mean = mean
        self._std = std

    def _preprocess_batch(self, imgs_bc_hw: torch.Tensor) -> torch.Tensor:
        x = imgs_bc_hw * self._std + self._mean
        x = x.clamp(0.0, 1.0)
        x = F.interpolate(x, size=(self._h, self._w), mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std
        return x.contiguous().float()

    def forward_probs(self, imgs_bc_hw: torch.Tensor) -> torch.Tensor:
        x = self._preprocess_batch(imgs_bc_hw)
        with torch.no_grad():
            logits = _forward_logits(self._model, x)
        logits = logits.reshape(logits.shape[0], -1).float()
        if logits.shape[-1] != len(NSFW_ENGINE_LABELS):
            raise RuntimeError(
                f"NSFW .pt output dim {logits.shape[-1]} != {len(NSFW_ENGINE_LABELS)}"
            )
        return torch.sigmoid(logits)

    def detail_indices_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        return detail_indices_from_probs(
            probs, tau_porn=self._tau_porn, tau_low_vote=self._tau_low_vote
        )

    def synchronize(self):
        if self._device.type == "cuda":
            torch.cuda.synchronize()


def _is_ultralytics_detection_ckpt(path: str) -> bool:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    if isinstance(ckpt, dict):
        m = ckpt.get("model")
        return m is not None and m.__class__.__name__ == "DetectionModel"
    return False


class NsfwUltralyticsRefiner:
    """
    Ultralytics YOLO DetectionModel checkpoint (18 NSFW classes): aggregate detection scores per class.
    """

    def __init__(
        self,
        weights_path: str,
        device: torch.device,
        input_size: Tuple[int, int] = NSFW_REFINE_INPUT_SIZE,
        tau_porn: float = 0.35,
        tau_low_vote: float = 0.08,
    ) -> None:
        from ultralytics import YOLO

        self._device = device
        self._h = int(input_size[0])
        self._tau_porn = tau_porn
        self._tau_low_vote = tau_low_vote
        self._yolo = YOLO(weights_path)
        self._yolo.to(device)

        mean = _IMAGENET_MEAN.to(device).float().view(1, 3, 1, 1)
        std = _IMAGENET_STD.to(device).float().view(1, 3, 1, 1)
        self._mean = mean
        self._std = std

    def forward_probs(self, imgs_bc_hw: torch.Tensor) -> torch.Tensor:
        device = imgs_bc_hw.device
        B = imgs_bc_hw.shape[0]
        x01 = (imgs_bc_hw * self._std + self._mean).clamp(0.0, 1.0)
        ims = []
        for i in range(B):
            im = (x01[i] * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            ims.append(im)
        dev_arg = str(device) if device.type == "cuda" else "cpu"
        results = self._yolo.predict(ims, imgsz=self._h, verbose=False, device=dev_arg)
        probs = torch.zeros(B, len(NSFW_ENGINE_LABELS), device=device, dtype=torch.float32)
        for bi, r in enumerate(results):
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            confs = boxes.conf
            clss = boxes.cls
            if confs is None or clss is None:
                continue
            for sj, cj in zip(confs, clss):
                j = int(cj.item())
                if 0 <= j < len(NSFW_ENGINE_LABELS):
                    s = float(sj.item())
                    probs[bi, j] = torch.maximum(
                        probs[bi, j], torch.tensor(s, device=device, dtype=torch.float32)
                    )
        return probs

    def detail_indices_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        return detail_indices_from_probs(
            probs, tau_porn=self._tau_porn, tau_low_vote=self._tau_low_vote
        )

    def synchronize(self):
        if self._device.type == "cuda":
            torch.cuda.synchronize()


def build_nsfw_refiner(
    path: str,
    device: torch.device,
    *,
    tau_porn: float = 0.35,
    tau_low_vote: float = 0.08,
    nsfw_input_size: Tuple[int, int] = NSFW_REFINE_INPUT_SIZE,
) -> Union[NsfwTrtRefiner, NsfwTorchRefiner, NsfwUltralyticsRefiner]:
    if path.lower().endswith(".engine"):
        return NsfwTrtRefiner(path, device, tau_porn=tau_porn, tau_low_vote=tau_low_vote)
    if _is_ultralytics_detection_ckpt(path):
        return NsfwUltralyticsRefiner(
            path, device, input_size=nsfw_input_size, tau_porn=tau_porn, tau_low_vote=tau_low_vote
        )
    return NsfwTorchRefiner(
        path, device, input_size=nsfw_input_size, tau_porn=tau_porn, tau_low_vote=tau_low_vote
    )


# Back-compat alias
NsfwEngineRefiner = NsfwTrtRefiner
