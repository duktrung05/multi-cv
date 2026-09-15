"""Frozen-feature extraction, linear-head training, and held-out evaluation."""
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from engine.backbone.dgm4_classifier import DINOv3BinaryClassifier
from engine.data.dataset.dgm4_binary import CLASS_NAMES, DGM4BinaryDataset
from src.infrastructure.ml.dgm4 import (load_config, make_transform, model_spec,
    resolve_device, resolve_path, sha256_file, validate_checkpoint)


def binary_metrics(targets, probabilities, threshold=.5):
    truth = targets.long().cpu()
    pred = (probabilities.cpu() >= threshold).long()
    cm = torch.bincount(truth * 2 + pred, minlength=4).reshape(2, 2)
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        tp = int(cm[i, i])
        fp, fn = int(cm[:, i].sum()) - tp, int(cm[i].sum()) - tp
        per_class[name] = {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                           "f1": 2 * tp / max(2 * tp + fp + fn, 1), "support": int(cm[i].sum())}
    from scipy.stats import rankdata
    p = probabilities.cpu().numpy()
    positive = truth.numpy() == 1
    npos, nneg = int(positive.sum()), int((~positive).sum())
    auc = float((rankdata(p)[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else None
    return {"n_images": len(truth), "accuracy": float((truth == pred).float().mean()),
            "macro_f1": sum(m["f1"] for m in per_class.values()) / 2,
            "false_positive_rate": int(cm[0, 1]) / nneg if nneg else None,
            "false_negative_rate": int(cm[1, 0]) / npos if npos else None,
            "roc_auc": auc, "threshold": threshold, "class_order": list(CLASS_NAMES),
            "confusion_matrix_rows_true_cols_pred": cm.tolist(), "per_class": per_class}


def dataset_signature(dataset):
    # Include current file metadata as well as the prepared pixel identity.
    items = []
    for row in dataset.rows:
        stat = (dataset.root / row["image"]).stat()
        items.append((row["image"], row["label"], row["pixel_sha256"], stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def extract(model, dataset, cfg, device, pretrained_hash, split):
    identity = {"version": 1, "dataset": dataset_signature(dataset),
                "model_spec": model_spec(cfg), "pretrained": pretrained_hash}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = resolve_path(cfg["cache_dir"]) / f"{split}_{key}.pt"
    if cache.is_file():
        state = torch.load(cache, map_location="cpu", weights_only=True)
        if state.get("identity") == identity and len(state["features"]) == len(dataset):
            print(f"Using cached {split} features: {len(dataset)}", flush=True)
            return state["features"], state["targets"]
    loader = DataLoader(dataset, batch_size=cfg["extract_batch_size"], shuffle=False,
                        num_workers=cfg.get("num_workers", 0), pin_memory=device.type == "cuda")
    features, labels = [], []
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        for step, (images, targets) in enumerate(loader):
            feat = model.extract_features(images.to(device)).float().cpu()
            if not torch.isfinite(feat).all():
                raise FloatingPointError("Non-finite DINOv3 features")
            features.append(feat)
            labels.append(targets)
            if step % 25 == 0 or step + 1 == len(loader):
                print(f"Extract {split}: {min((step+1)*cfg['extract_batch_size'], len(dataset))}/{len(dataset)}, elapsed={time.perf_counter()-start:.1f}s", flush=True)
    x, y = torch.cat(features), torch.cat(labels)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp")
    torch.save({"identity": identity, "features": x, "targets": y}, temporary)
    temporary.replace(cache)
    return x, y


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def evaluate_checkpoint(cfg, checkpoint, split="test", output=None):
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test")
    state = torch.load(resolve_path(checkpoint), map_location="cpu", weights_only=True)
    validate_checkpoint(state, cfg)
    device = resolve_device(cfg["device"])
    model = DINOv3BinaryClassifier().to(device)
    model.load_state_dict(state["model"], strict=True)
    dataset = DGM4BinaryDataset(resolve_path(cfg["dataset_root"]), split, make_transform(cfg))
    if state.get("manifest_sha256") != sha256_file(dataset.root / "manifest.csv"):
        raise ValueError("Evaluation dataset manifest differs from the training dataset")
    x, y = extract(model, dataset, cfg, device, state["pretrained_sha256"], split)
    model.eval()
    with torch.no_grad():
        logits = model.head(x.to(device)).cpu()
        probabilities = logits.softmax(-1)[:, 1]
    report = binary_metrics(y, probabilities, cfg["threshold"])
    report.update(split=split, checkpoint=str(resolve_path(checkpoint)),
                  checkpoint_sha256=sha256_file(resolve_path(checkpoint)),
                  manifest_sha256=state["manifest_sha256"], loss=float(F.cross_entropy(logits, y)))
    report["by_method"] = {}
    pred = (probabilities >= cfg["threshold"]).long()
    for method in sorted({row["method"] for row in dataset.rows}):
        mask = torch.tensor([r["method"] == method for r in dataset.rows])
        report["by_method"][method] = {"n_images": int(mask.sum()), "correct_rate": float((pred[mask] == y[mask]).float().mean())}
    output = Path(output) if output else resolve_path(checkpoint).parent / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{split}_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    records = [{"image": row["image"], "source_id": row["source_id"], "method": row["method"],
                "true_label": CLASS_NAMES[int(y[i])], "pred_label": CLASS_NAMES[int(pred[i])],
                "ai_edited_score": float(probabilities[i]), "correct": bool(y[i] == pred[i])}
               for i, row in enumerate(dataset.rows)]
    for name, subset in (("predictions", records), ("errors", [r for r in records if not r["correct"]])):
        with (output / f"{split}_{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(subset)
    print(json.dumps(report, indent=2), flush=True)
    return report


def train_head(model, train_data, val_data, cfg, output, provenance, resume_state=None):
    """All updates are confined to the head; validation selects the checkpoint."""
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=.5)
    start, best, stale = 0, -1., 0
    generator = torch.Generator().manual_seed(cfg["seed"])
    history = []
    if resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        start, best, stale = resume_state["epoch"] + 1, resume_state["best_metric"], resume_state["stale_epochs"]
        generator.set_state(resume_state["loader_rng"])
        history = resume_state["history"]
    train_loader = DataLoader(TensorDataset(*train_data), batch_size=cfg["head_batch_size"], shuffle=True, generator=generator)
    val_x, val_y = val_data
    for epoch in range(start, cfg["epochs"]):
        model.train()
        loss_sum, n = 0., 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model.head(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite head training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            n += len(y)
        model.eval()
        with torch.no_grad():
            logits = model.head(val_x.to(device)).cpu()
        metric = binary_metrics(val_y, logits.softmax(-1)[:, 1], cfg["threshold"])
        metric.update(epoch=epoch, train_loss=loss_sum/n, val_loss=float(F.cross_entropy(logits, val_y)), lr=optimizer.param_groups[0]["lr"])
        history.append(metric)
        scheduler.step(metric["macro_f1"])
        improved = metric["macro_f1"] > best
        best, stale = (metric["macro_f1"], 0) if improved else (best, stale + 1)
        state = {**provenance, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                 "task_domain": "dgm4_binary", "class_list": list(CLASS_NAMES), "model_spec": model_spec(cfg),
                 "config": cfg, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "epoch": epoch, "best_metric": best, "stale_epochs": stale,
                 "loader_rng": generator.get_state(), "history": history}
        save_checkpoint(output / "last.pth", state)
        if improved:
            save_checkpoint(output / "best.pth", state)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Epoch {epoch+1}/{cfg['epochs']}: loss={loss_sum/n:.4f}, val_macro_f1={metric['macro_f1']:.4f}, val_acc={metric['accuracy']:.4f}", flush=True)
        if cfg["early_stopping_patience"] > 0 and stale >= cfg["early_stopping_patience"]:
            break
    return history


def run(config_path, overrides=None):
    cfg = load_config(config_path, overrides)
    for key in ("extract_batch_size", "head_batch_size", "epochs", "num_threads"):
        if int(cfg[key]) < 1:
            raise ValueError(f"{key} must be positive")
    torch.set_num_threads(cfg["num_threads"])
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    print(f"DGM4 device: {device}; frozen DINOv3; labels={list(CLASS_NAMES)}", flush=True)
    if cfg.get("test_only"):
        if not cfg.get("resume"):
            raise ValueError("test_only requires resume=path/to/best.pth")
        return evaluate_checkpoint(cfg, cfg["resume"], cfg.get("eval_split", "val"))
    pretrained = resolve_path(cfg["pretrained"])
    if not pretrained.is_file():
        raise FileNotFoundError(f"Missing DINOv3 pretrained: {pretrained}. Random initialization is forbidden for training.")
    if not cfg.get("limit_per_class"):
        audit_path = resolve_path(cfg["audit_report"])
        if not audit_path.is_file():
            raise ValueError("Run tools/dataset/validate_dgm4.py before full training")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit.get("ready_for_training") or audit.get("manifest_sha256") != sha256_file(resolve_path(cfg["dataset_root"]) / "manifest.csv"):
            raise ValueError("Dataset audit is unresolved or stale; resolve cross-split near-duplicates and re-audit before full training")
    output = resolve_path(cfg["output_dir"])
    if output.exists() and any(output.iterdir()) and not cfg.get("resume"):
        raise FileExistsError("Training output is not empty; choose a new output_dir or resume an existing run")
    model = DINOv3BinaryClassifier().to(device)
    model.backbone.load_state_dict(torch.load(pretrained, map_location="cpu", weights_only=True), strict=True)
    pretrained_hash = sha256_file(pretrained)
    transform = make_transform(cfg)
    datasets = {s: DGM4BinaryDataset(resolve_path(cfg["dataset_root"]), s, transform, cfg.get("limit_per_class")) for s in ("train", "val")}
    provenance = {"pretrained_sha256": pretrained_hash, "smoke_only": False,
                  "manifest_sha256": sha256_file(resolve_path(cfg["dataset_root"]) / "manifest.csv"),
                  "dataset_signatures": {s: dataset_signature(ds) for s, ds in datasets.items()}}
    resume_state = None
    if cfg.get("resume"):
        resume_path = resolve_path(cfg["resume"])
        if resume_path.parent.resolve() != output.resolve():
            raise ValueError("Resume in the same output_dir to preserve best.pth")
        if not (output / "best.pth").is_file():
            raise ValueError("Resume requires the run's best.pth")
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=True)
        validate_checkpoint(resume_state, cfg)
        if any(resume_state.get(k) != v for k, v in provenance.items()):
            raise ValueError("Resume provenance differs from current data or pretrained")
        model.load_state_dict(resume_state["model"], strict=True)
    output.mkdir(parents=True, exist_ok=True)
    import yaml
    (output / "resolved_config.yml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    cached = {s: extract(model, ds, cfg, device, pretrained_hash, s) for s, ds in datasets.items()}
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}", flush=True)
    return train_head(model, cached["train"], cached["val"], cfg, output, provenance, resume_state)
