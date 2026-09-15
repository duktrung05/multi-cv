"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn

from ..misc import (MetricLogger, SmoothedValue, reduce_dict)
from ..misc.nsfw_engine_refine import build_nsfw_refiner
from ..misc import dist_utils
from ..misc.tqdm import TQDM


def _leaf_dataset(dataloader):
    ds = dataloader.dataset
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return ds


def _gpu_mem_gb(device) -> float:
    try:
        if device is None:
            return 0.0
        if hasattr(device, "type") and device.type == "cuda" and torch.cuda.is_available():
            return float(torch.cuda.memory_reserved(device)) / (1024**3)
    except Exception:
        pass
    return 0.0


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    dataloader,
    optimizer,
    ema,
    epoch,
    device,
    lr_warmup_scheduler=None,
):
    """
    """
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', SmoothedValue(window_size=1))
    metric_logger.add_meter('loss', SmoothedValue(window_size=1))

    # Running confusion matrix for macro-F1 (train-time signal).
    confmat = None

    nb = len(dataloader) if hasattr(dataloader, "__len__") else None
    if dist_utils.is_main_process():
        # Ultralytics-like header: fixed-width columns
        header = ("\n" + "%11s" * 6) % ("Epoch", "GPU_mem", "loss", "acc", "f1", "imgsz")
        TQDM.write(header)
        pbar = TQDM(enumerate(dataloader), total=nb, desc="", unit="batch")
    else:
        pbar = enumerate(dataloader)

    for i, batch in pbar:
        if len(batch) == 3:
            imgs, labels, _detail = batch
        else:
            imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)

        preds = model(imgs)
        # If NaN happens, training will never recover (acc will stay 0).
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            raise FloatingPointError(f"Non-finite preds at epoch={epoch}.")
        loss: torch.Tensor = criterion(preds, labels, epoch)
        if not torch.isfinite(loss).all():
            raise FloatingPointError(
                f"Non-finite loss at epoch={epoch}: {loss.detach().item()}"
            )

        acc = (preds.argmax(dim=-1) == labels).sum() / preds.shape[0]

        # Update running confusion matrix (rows=true, cols=pred)
        pred_idx = preds.argmax(dim=-1)
        k = int(preds.shape[-1])
        if confmat is None:
            confmat = torch.zeros((k, k), dtype=torch.int64, device=device)
        confmat.index_put_(
            (labels.view(-1), pred_idx.view(-1)),
            torch.ones_like(labels.view(-1), dtype=torch.int64),
            accumulate=True,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None:
            ema.update(model)

        # Step LR warmup per-iteration.
        # Without this, warmup only applies once at scheduler init, making LR too small.
        if lr_warmup_scheduler is not None:
            try:
                if not hasattr(lr_warmup_scheduler, "finished") or not lr_warmup_scheduler.finished():
                    lr_warmup_scheduler.step()
            except TypeError:
                # Some schedulers may not support finished(); just step.
                lr_warmup_scheduler.step()

        reduced_values = {k: v.item() for k, v in reduce_dict({'loss': loss, 'acc': acc}).items()}
        metric_logger.update(**reduced_values)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if dist_utils.is_main_process():
            try:
                loss_v = float(reduced_values.get("loss", 0.0))
                acc_v = float(reduced_values.get("acc", 0.0))
                imgsz = int(getattr(imgs, "shape", [0, 0, 0, 0])[-1])
                mem = _gpu_mem_gb(device)
                # Running macro-F1 (rank0 local view; final macro_f1 is computed from reduced confmat at epoch end)
                f1_v = 0.0
                try:
                    cm = confmat.to(torch.float32)
                    tp = torch.diag(cm)
                    fp = cm.sum(dim=0) - tp
                    fn = cm.sum(dim=1) - tp
                    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-12)
                    f1_v = float(f1.mean().item())
                except Exception:
                    f1_v = 0.0
                # Match Ultralytics-style compact row
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * 4)
                    % (f"{epoch}", f"{mem:.3g}G", loss_v, acc_v, f1_v, imgsz)
                )
            except Exception:
                pass

    metric_logger.synchronize_between_processes()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # Reduce confusion matrix across ranks and compute macro_f1.
    if confmat is not None:
        confmat = reduce_dict({"confmat": confmat})["confmat"].to(torch.float32)
        tp = torch.diag(confmat)
        fp = confmat.sum(dim=0) - tp
        fn = confmat.sum(dim=1) - tp
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-12)
        stats["macro_f1"] = float(f1.mean().item())
    return stats



@torch.no_grad()
def evaluate(
    model,
    criterion,
    dataloader,
    device,
    nsfw_engine_path: str | None = None,
):
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('acc', SmoothedValue(window_size=1, fmt='{global_avg:.4f}'))
    # metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('acc', SmoothedValue(window_size=1))
    metric_logger.add_meter('loss', SmoothedValue(window_size=1))

    # For macro F1: accumulate confusion matrix on-device.
    confmat = None

    nsfw_refiner = None
    merged_idx = None
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if nsfw_engine_path:
        ds = _leaf_dataset(dataloader)
        class_names = list(getattr(ds, "classes", []) or [])
        if "NSFW_EXPLICIT" not in class_names:
            raise ValueError(
                "nsfw_engine_path is set but dataset.classes has no 'NSFW_EXPLICIT'; "
                "use class_merge + NSFW_EXPLICIT in YAML."
            )
        merged_idx = int(class_names.index("NSFW_EXPLICIT"))
        nsfw_refiner = build_nsfw_refiner(nsfw_engine_path, dev)

    detail_correct = 0
    detail_total = 0

    nb = len(dataloader) if hasattr(dataloader, "__len__") else None
    if dist_utils.is_main_process():
        header = ("\n" + "%11s" * 5) % ("Valid", "loss", "acc", "f1", "imgsz")
        TQDM.write(header)
        pbar = TQDM(enumerate(dataloader), total=nb, desc="", unit="batch")
    else:
        pbar = enumerate(dataloader)

    for i, batch in pbar:
        if len(batch) == 3:
            imgs, labels, detail_tgt = batch
            detail_tgt = detail_tgt.to(device)
        else:
            imgs, labels = batch
            detail_tgt = None
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)

        pred_idx = preds.argmax(dim=-1)
        if nsfw_refiner is not None and merged_idx is not None and detail_tgt is not None:
            probs = nsfw_refiner.forward_probs(imgs)
            d_pred = nsfw_refiner.detail_indices_from_probs(probs)
            refine_ok = (pred_idx == merged_idx) & (detail_tgt >= 0)
            if refine_ok.any():
                detail_total += int(refine_ok.sum().item())
                detail_correct += int((d_pred[refine_ok] == detail_tgt[refine_ok]).sum().item())
            nsfw_refiner.synchronize()

        acc = (pred_idx == labels).sum() / preds.shape[0]
        loss = criterion(preds, labels)

        # Update confusion matrix
        k = int(preds.shape[-1])
        if confmat is None:
            confmat = torch.zeros((k, k), dtype=torch.int64, device=device)
        # rows=true, cols=pred
        confmat.index_put_(
            (labels.view(-1), pred_idx.view(-1)),
            torch.ones_like(labels.view(-1), dtype=torch.int64),
            accumulate=True,
        )

        dict_reduced = reduce_dict({'acc': acc, 'loss': loss})
        reduced_values = {k: v.item() for k, v in dict_reduced.items()}
        metric_logger.update(**reduced_values)

        if dist_utils.is_main_process():
            try:
                loss_v = float(reduced_values.get("loss", 0.0))
                acc_v = float(reduced_values.get("acc", 0.0))
                imgsz = int(getattr(imgs, "shape", [0, 0, 0, 0])[-1])
                # Running macro-F1 (rank0 local view)
                f1_v = 0.0
                try:
                    cm = confmat.to(torch.float32) if confmat is not None else None
                    if cm is not None:
                        tp = torch.diag(cm)
                        fp = cm.sum(dim=0) - tp
                        fn = cm.sum(dim=1) - tp
                        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-12)
                        f1_v = float(f1.mean().item())
                except Exception:
                    f1_v = 0.0
                pbar.set_description(("%11s" + "%11.4g" * 4) % ("", loss_v, acc_v, f1_v, imgsz))
            except Exception:
                pass

    metric_logger.synchronize_between_processes()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # Compute macro F1 from reduced confusion matrix (sum across ranks).
    if confmat is not None:
        confmat = reduce_dict({"confmat": confmat})["confmat"]
        confmat = confmat.to(torch.float32)
        tp = torch.diag(confmat)
        fp = confmat.sum(dim=0) - tp
        fn = confmat.sum(dim=1) - tp
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-12)
        stats["macro_f1"] = float(f1.mean().item())
        # rows=true, cols=pred (same as index_put_ above); train.py maps to Ultralytics ConfusionMatrix.
        stats["confmat"] = confmat.cpu().numpy()

    leaf_ds = _leaf_dataset(dataloader)
    classes_list = list(getattr(leaf_ds, "classes", []) or [])
    if classes_list:
        stats["class_names"] = classes_list

    if nsfw_refiner is not None and merged_idx is not None:
        dc = torch.tensor([detail_correct], dtype=torch.float32, device=dev)
        dt = torch.tensor([detail_total], dtype=torch.float32, device=dev)
        dc = reduce_dict({"dc": dc}, avg=False)["dc"].item()
        dt = reduce_dict({"dt": dt}, avg=False)["dt"].item()
        stats["nsfw_detail_acc"] = float(dc / dt) if dt > 0 else float("nan")
        stats["nsfw_detail_correct"] = int(dc)
        stats["nsfw_detail_total"] = int(dt)

    return stats
