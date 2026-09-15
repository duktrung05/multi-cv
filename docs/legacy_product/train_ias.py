"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import pprint

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS

debug=False


def _print_classification_confusion_matrix(solver) -> None:
    """Print Ultralytics ConfusionMatrix for the last validation pass (classification only).

    Ultralytics classify layout: ``matrix[pred, true]`` (see ``ConfusionMatrix.plot``: x=True, y=Predicted).
    ``clas_engine.evaluate`` accumulates rows=true, cols=pred, so we transpose when assigning.

    Imports are deferred here so ``python train.py`` / ``--help`` start without loading ultralytics (slow).
    """
    import numpy as np
    import torch
    from ultralytics.utils.metrics import ConfusionMatrix
    from ultralytics.utils.plotting import plot_images

    if not dist_utils.is_main_process():
        return
    stats = getattr(solver, "_last_val_stats", None)
    if not stats or "confmat" not in stats:
        return

    cm_true_pred = np.asarray(stats["confmat"], dtype=np.float64)
    nc = int(cm_true_pred.shape[0])
    raw = list(stats.get("class_names") or [])
    names: dict[int, str] = {i: str(raw[i]) if i < len(raw) else str(i) for i in range(nc)}

    cm = ConfusionMatrix(names=names, task="classify")
    cm.matrix = cm_true_pred.T.copy()

    print("\n========== ConfusionMatrix (ultralytics.utils.metrics.ConfusionMatrix) ==========")
    print("Labels (class index -> name):")
    for i in range(nc):
        print(f"  {i}: {names[i]}")
    print(
        "\nMatrix layout: rows = Predicted, columns = True (same as Ultralytics val plots).\n"
        "Console rows from ConfusionMatrix.print() (no axis names in that helper):\n"
    )
    cm.print()
    print("===========================================================================\n")

    out_dir = str((solver.cfg.yaml_cfg or {}).get("output_dir") or "./outputs")
    os.makedirs(out_dir, exist_ok=True)
    try:
        n = min(16, max(8, nc))
        images = torch.rand(n, 3, 224, 224)
        plot_labels = {
            "cls": np.array([i % nc for i in range(n)], dtype=np.int64),
            "batch_idx": np.arange(n, dtype=np.int64),
        }
        fname = os.path.join(out_dir, "label.jpg")
        plot_images(labels=plot_labels, images=images, fname=fname, names=names)
        print(f"[train] wrote class-name grid (plot_images): {fname}\n")
    except Exception as e:
        print(f"[train][WARN] label.jpg (plot_images) failed: {e}\n")

    try:
        cm.plot(normalize=False, save_dir=out_dir)
        print(f"[train] wrote confusion matrix: {os.path.join(out_dir, 'confusion_matrix.png')}\n")
    except Exception as e:
        print(f"[train][WARN] confusion_matrix.png (ConfusionMatrix.plot) failed: {e}\n")


if debug:
    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

def initialize_training_context(args):
    """Load config, distributed runtime, device, and derived ``num_classes`` (bootstrap shared with ``test/test_ias.py``).

    Default config trains the product inspection multi-label model.
    """
    repo_root = os.path.dirname(os.path.abspath(__file__))

    # Default: product inspection with explicit CSV annotations.
    default_config = os.path.join(repo_root, 'configs', 'product_inspection.yml')
    fixed_config = args.config or default_config
    if not os.path.isabs(fixed_config):
        fixed_config = os.path.join(repo_root, fixed_config)

    update_dict = yaml_utils.parse_cli(args.update)
    cfg = YAMLConfig(fixed_config, **update_dict)
    cfg.yaml_cfg = cfg.yaml_cfg or {}

    def _resolve_opt_path(key: str) -> str:
        p = str((cfg.yaml_cfg or {}).get(key) or "").strip()
        if p and not os.path.isabs(p):
            p = os.path.join(repo_root, p)
        cfg.yaml_cfg[key] = p
        return p

    _resolve_opt_path("resume")
    _resolve_opt_path("tuning")
    _resolve_opt_path("nsfw_engine_path")

    assert not all([cfg.yaml_cfg.get("tuning"), cfg.yaml_cfg.get("resume")]), \
        'Only support from_scrach or resume at one time'

    dist_utils.setup_distributed(
        int(cfg.yaml_cfg.get("print_rank", 0)),
        str(cfg.yaml_cfg.get("print_method", "builtin")),
        seed=cfg.yaml_cfg.get("seed"),
    )

    cfg.yaml_cfg['output_dir'] = str((cfg.yaml_cfg or {}).get("output_dir") or "./outputs")
    cfg.output_dir = cfg.yaml_cfg['output_dir']
    cfg.yaml_cfg['summary_dir'] = str((cfg.yaml_cfg or {}).get("summary_dir") or "./outputs/summary")
    cfg.summary_dir = cfg.yaml_cfg['summary_dir']
    cfg.resume = cfg.yaml_cfg.get("resume")
    cfg.tuning = cfg.yaml_cfg.get("tuning")

    gpu_device = str((cfg.yaml_cfg or {}).get("device") or "").strip()
    if gpu_device.lower() == "cpu":
        cfg.yaml_cfg["device"] = "cpu"
        print("[Device] training on cpu")
    else:
        # normalize device string
        if gpu_device.isdigit():
            gpu_device = f"cuda:{gpu_device}"
        # If user passes '', fallback to cuda:0 if possible
        if gpu_device == "":
            gpu_device = "cuda:0"
        try:
            import torch
            if gpu_device.startswith("cuda") and not torch.cuda.is_available():
                print(f"[Device] CUDA requested ({gpu_device}) but torch says cuda not available; fallback to cpu")
                cfg.yaml_cfg["device"] = "cpu"
            else:
                cfg.yaml_cfg["device"] = gpu_device
                device_id = gpu_device.split(":")[-1] if ":" in gpu_device else ""
                print(f"[Device] training on {gpu_device}{' (id '+device_id+')' if device_id != '' else ''}")
        except Exception:
            cfg.yaml_cfg["device"] = "cpu"
            print("[Device] training on cpu (device check failed)")

    # Keep `BaseConfig.device` consistent with yaml config.
    cfg.device = cfg.yaml_cfg.get("device", cfg.device)

    if cfg.yaml_cfg.get("task") == "multilabel_classification":
        names = cfg.yaml_cfg.get("class_list") or []
        model_cfg = cfg.yaml_cfg.get(cfg.yaml_cfg.get("model"), {})
        if not names or len(set(names)) != len(names):
            raise ValueError("Multi-label training requires an explicit unique class_list")
        if len(names) != cfg.yaml_cfg.get("num_classes") or len(names) != model_cfg.get("num_classes"):
            raise ValueError("num_classes must match the explicit label order")
        for split in ("train_dataloader", "val_dataloader"):
            ds = cfg.yaml_cfg[split]["dataset"]
            if ds["class_list"] != names:
                raise ValueError(f"{split} label order differs from model")
            for key in ("root", "annotations_path"):
                if key in ds and not os.path.isabs(ds[key]):
                    ds[key] = os.path.join(repo_root, ds[key])
        if cfg.yaml_cfg.get("task_domain") == "product_inspection" and (cfg.resume or cfg.tuning):
            import torch
            state = torch.load(cfg.resume or cfg.tuning, map_location="cpu", weights_only=True)
            if state.get("class_list") != names or state.get("task_domain") != "product_inspection":
                raise ValueError("Product checkpoint domain/label order does not match config")
        pretrained = model_cfg.get("backbone", {}).get("weights_path")
        if pretrained and not os.path.isabs(pretrained):
            pretrained = os.path.join(repo_root, pretrained)
            model_cfg["backbone"]["weights_path"] = pretrained
        if not (cfg.resume or cfg.tuning) and (not pretrained or not os.path.isfile(pretrained)):
            raise FileNotFoundError(f"DINOv3 pretrained weights required: {pretrained}")
        return repo_root, cfg

    # ---------------------------
    # Num classes = only from train folders (from config)
    # ---------------------------
    def _resolve_root(p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.join(repo_root, p)

    ycfg = cfg.yaml_cfg or {}
    task = str(ycfg.get("task") or "classification")
    train_dataset_cfg = (((ycfg.get('train_dataloader') or {}).get('dataset') or {}))

    if task == "multilabel_classification":
        class_list = list(train_dataset_cfg.get("class_list") or ycfg.get("class_list") or [])
        if not class_list:
            raise ValueError("multilabel_classification requires class_list in dataset YAML")
        num_classes = len(class_list)

        def _update_multilabel_nums(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "num_classes":
                        obj[k] = num_classes
                    else:
                        _update_multilabel_nums(v)
            elif isinstance(obj, list):
                for item in obj:
                    _update_multilabel_nums(item)

        _update_multilabel_nums(cfg.yaml_cfg)
        ycfg["num_classes"] = num_classes
        if cfg.resume:
            if "HGNetv2" in cfg.yaml_cfg:
                cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        return repo_root, cfg

    train_root_cfg = str(train_dataset_cfg.get('root', ''))
    classes_root_cfg = train_dataset_cfg.get('classes_root', None)

    class_roots = []
    if classes_root_cfg is None:
        class_roots = [_resolve_root(train_root_cfg)]
    elif isinstance(classes_root_cfg, str):
        class_roots = [_resolve_root(classes_root_cfg)]
    elif isinstance(classes_root_cfg, list):
        class_roots = [_resolve_root(str(x)) for x in classes_root_cfg]
    else:
        class_roots = [_resolve_root(train_root_cfg)]

    class_set = set()
    for r in class_roots:
        if not os.path.isdir(r):
            continue
        for d in os.listdir(r):
            if d.startswith('.'):
                continue
            if os.path.isdir(os.path.join(r, d)):
                class_set.add(d)

    yaml_class_list = train_dataset_cfg.get("class_list")
    yaml_class_merge = train_dataset_cfg.get("class_merge") or {}

    if yaml_class_list:
        merge_map = {str(k): [str(x) for x in v] for k, v in yaml_class_merge.items()}

        def _canonical_present(canonical: str) -> bool:
            if canonical in class_set:
                return True
            if canonical in merge_map:
                return any(p in class_set for p in merge_map[canonical])
            return False

        num_classes = sum(1 for c in yaml_class_list if _canonical_present(str(c)))
    else:
        num_classes = len(sorted(class_set))

    if num_classes <= 0:
        raise ValueError(f"No class folders found under class_roots={class_roots} (from config)")

    def _update_num_classes(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == 'num_classes':
                    obj[k] = num_classes
                else:
                    _update_num_classes(v)
        elif isinstance(obj, list):
            for item in obj:
                _update_num_classes(item)

    _update_num_classes(cfg.yaml_cfg)

    if cfg.resume:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    return repo_root, cfg


def main(args, ) -> None:
    """main
    """
    repo_root, cfg = initialize_training_context(args)

    # ---------------------------
    # Pretty config print + full resolved config dump
    # ---------------------------
    if dist_utils.is_main_process():
        y = cfg.yaml_cfg or {}
        train_ds = (((y.get("train_dataloader") or {}).get("dataset") or {}))
        val_ds = (((y.get("val_dataloader") or {}).get("dataset") or {}))
        summary = {
            "task": y.get("task"),
            "device": y.get("device"),
            "output_dir": y.get("output_dir"),
            "summary_dir": y.get("summary_dir"),
            "epoches": y.get("epoches"),
            "num_classes": y.get("num_classes"),
            "model": y.get("model"),
            "criterion": {
                "type": y.get("criterion"),
                **(y.get(str(y.get("criterion", ""))) or {}),
            },
            "lr_scheduler": y.get("lr_scheduler"),
            "train_dataset": {
                "type": train_ds.get("type"),
                "root": train_ds.get("root"),
                "classes_root": train_ds.get("classes_root"),
                "hard_classes": train_ds.get("hard_classes"),
                "transforms_type": (train_ds.get("transforms") or {}).get("type"),
                "hard_transforms_type": (train_ds.get("hard_transforms") or {}).get("type"),
            },
            "val_dataset": {
                "type": val_ds.get("type"),
                "root": val_ds.get("root"),
                "classes_root": val_ds.get("classes_root"),
                "drop_unknown": val_ds.get("drop_unknown"),
                "transforms_type": (val_ds.get("transforms") or {}).get("type"),
            },
        }

        print("\n========== CONFIG (summary) ==========")
        pprint.pprint(summary, width=120, compact=False, sort_dicts=False)
        print("======================================\n")

        # Write the full resolved yaml config to output_dir for reproducibility.
        try:
            import yaml

            out_dir = str(y.get("output_dir") or "./outputs")
            os.makedirs(out_dir, exist_ok=True)
            resolved_path = os.path.join(out_dir, "resolved_config.yml")
            with open(resolved_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(y, f, sort_keys=False, allow_unicode=True)
            print(f"[Config] wrote full resolved config to: {resolved_path}")
        except Exception as e:
            print(f"[Config][WARN] failed to write resolved_config.yml: {e}")

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if bool((cfg.yaml_cfg or {}).get("test_only", False)):
        solver.val()
    else:
        solver.fit()

    task = str(cfg.yaml_cfg.get("task") or "")
    if task == "classification":
        _print_classification_confusion_matrix(solver)
    elif task == "multilabel_classification" and dist_utils.is_main_process():
        stats = getattr(solver, "_last_val_stats", None) or {}
        print(
            "\n[train_ias] multilabel eval: inference=sigmoid_per_class "
            f"threshold={stats.get('multilabel_threshold', 0.5)} "
            f"macro_f1={stats.get('macro_f1', 0):.4f}\n"
        )

    dist_utils.cleanup()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='configs/product_inspection.yml',
        help='15cls softmax: reas_dinov3_vit_s_15cls.yml | multilabel sigmoid 15cls: reas_dinov3_vit_s_multilabel_15cls.yml',
    )
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')
    parser.add_argument('--local-rank', type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    main(args)
