"""Train/eval loop for multi-label (sigmoid) classification."""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from ..misc import MetricLogger, SmoothedValue, dist_utils, reduce_dict
from ..misc.tqdm import TQDM


def _unpack_batch(batch):
    imgs, label_vec = batch[0], batch[1]
    return imgs, label_vec


def _unpack_preds(preds):
    if isinstance(preds, (tuple, list)):
        return preds[0]
    return preds


def _macro_f1_multilabel(pred_bin: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred_bin.float()
    tgt = target.float()
    tp = (pred * tgt).sum(dim=0)
    fp = (pred * (1 - tgt)).sum(dim=0)
    fn = ((1 - pred) * tgt).sum(dim=0)
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-12)
    return float(f1.mean().item())


@torch.no_grad()
def _threshold() -> float:
    return 0.5


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    dataloader,
    optimizer,
    ema,
    epoch,
    device,
    lr_warmup_scheduler=None,
    max_norm: float = 0.0,
):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", SmoothedValue(window_size=1))

    nb = len(dataloader) if hasattr(dataloader, "__len__") else None
    if dist_utils.is_main_process():
        header = ("\n" + "%11s" * 5) % ("Epoch", "GPU_mem", "loss", "ml_f1", "imgsz")
        TQDM.write(header)
        pbar = TQDM(enumerate(dataloader), total=nb, desc="", unit="batch")
    else:
        pbar = enumerate(dataloader)

    for _i, batch in pbar:
        imgs, label_vec = _unpack_batch(batch)
        imgs = imgs.to(device)
        label_vec = label_vec.to(device)

        preds = model(imgs)
        label_logits = _unpack_preds(preds)
        loss: torch.Tensor = criterion(label_logits, label_vec, epoch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss; check images, targets and checkpoint")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        prob = torch.sigmoid(label_logits)
        f1 = _macro_f1_multilabel(prob >= _threshold(), label_vec)

        metric_logger.update(loss=loss.item(), ml_f1=f1, lr=optimizer.param_groups[0]["lr"])

        if dist_utils.is_main_process():
            try:
                imgsz = int(imgs.shape[-1])
                pbar.set_description(
                    ("%11s" + "%11.4g" * 4)
                    % ("", float(loss.item()), f1, imgsz)
                )
            except Exception:
                pass

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    dataloader,
    device,
    threshold: float = 0.5,
    **kwargs,
):
    del kwargs
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", SmoothedValue(window_size=1))
    metric_logger.add_meter("ml_acc", SmoothedValue(window_size=1))

    n_images = 0
    class_names: list[str] = []
    ds = dataloader.dataset
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    class_names = list(getattr(ds, "classes", []) or [])
    nc = len(class_names)
    tp = torch.zeros(nc, dtype=torch.float64)
    fp = torch.zeros(nc, dtype=torch.float64)
    fn = torch.zeros(nc, dtype=torch.float64)

    nb = len(dataloader) if hasattr(dataloader, "__len__") else None
    if dist_utils.is_main_process():
        header = ("\n" + "%11s" * 5) % ("Valid", "loss", "ml_f1", "ml_acc", "imgsz")
        TQDM.write(header)
        pbar = TQDM(enumerate(dataloader), total=nb, desc="", unit="batch")
    else:
        pbar = enumerate(dataloader)

    for _i, batch in pbar:
        imgs, label_vec = _unpack_batch(batch)
        imgs = imgs.to(device)
        label_vec = label_vec.to(device)

        preds = model(imgs)
        label_logits = _unpack_preds(preds)
        loss = criterion(label_logits, label_vec)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss; checkpoint will not be saved")

        prob = torch.sigmoid(label_logits)
        pred_bin = prob >= threshold
        tgt = label_vec > 0.5
        if nc:
            pred_f = pred_bin.float()
            tgt_f = tgt.float()
            tp += (pred_f * tgt_f).sum(dim=0).cpu().double()
            fp += (pred_f * (1 - tgt_f)).sum(dim=0).cpu().double()
            fn += ((1 - pred_f) * tgt_f).sum(dim=0).cpu().double()
        exact = (pred_bin == tgt).all(dim=-1).float().mean()

        n_images += int(imgs.shape[0])
        metric_logger.update(loss=loss.item(), ml_acc=float(exact.item()))

        if dist_utils.is_main_process():
            try:
                imgsz = int(imgs.shape[-1])
                running_f1 = float(((2 * tp) / (2 * tp + fp + fn + 1e-12)).mean().item()) if nc else 0.0
                pbar.set_description(
                    ("%11s" + "%11.4g" * 4)
                    % ("", float(loss.item()), running_f1, float(exact.item()), imgsz)
                )
            except Exception:
                pass

    metric_logger.synchronize_between_processes()
    if dist_utils.is_dist_available_and_initialized():
        t = torch.stack([tp, fp, fn]).to(device)
        torch.distributed.all_reduce(t)
        tp, fp, fn = t.cpu()[0], t.cpu()[1], t.cpu()[2]

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    per_class_f1 = (2 * tp) / (2 * tp + fp + fn + 1e-12)
    macro_f1 = float(per_class_f1.mean().item()) if nc else 0.0
    stats["per_class_precision"] = dict(zip(class_names, (tp / (tp + fp + 1e-12)).tolist()))
    stats["per_class_recall"] = dict(zip(class_names, (tp / (tp + fn + 1e-12)).tolist()))
    stats["per_class_f1"] = dict(zip(class_names, per_class_f1.tolist()))
    stats["macro_f1"] = macro_f1
    stats["ml_f1"] = macro_f1
    stats["acc"] = stats.get("ml_acc", 0.0)
    stats["n_eval_images"] = n_images
    if class_names:
        stats["class_names"] = class_names
    stats["multilabel_threshold"] = threshold
    stats["inference"] = "sigmoid_per_class"

    return stats


def _per_class_f1_at_threshold(
    model: nn.Module,
    dataloader,
    device,
    threshold: float,
) -> dict[str, float]:
    """One-pass per-class F1 (for test export)."""
    model.eval()
    leaf = dataloader.dataset
    while hasattr(leaf, "dataset"):
        leaf = leaf.dataset
    names = list(getattr(leaf, "classes", []) or [])
    nc = len(names)
    tp = torch.zeros(nc, dtype=torch.float64)
    fp = torch.zeros(nc, dtype=torch.float64)
    fn = torch.zeros(nc, dtype=torch.float64)
    with torch.no_grad():
        for batch in dataloader:
            imgs, label_vec = _unpack_batch(batch)
            imgs = imgs.to(device)
            label_vec = label_vec.to(device)
            logits = _unpack_preds(model(imgs))
            pred = (torch.sigmoid(logits) >= threshold).float()
            tgt = (label_vec > 0.5).float()
            tp += (pred * tgt).sum(dim=0).cpu().double()
            fp += (pred * (1 - tgt)).sum(dim=0).cpu().double()
            fn += ((1 - pred) * tgt).sum(dim=0).cpu().double()
    out: dict[str, float] = {}
    for i, name in enumerate(names):
        f1 = (2 * tp[i]) / (2 * tp[i] + fp[i] + fn[i] + 1e-12)
        out[name] = float(f1.item())
    return out
