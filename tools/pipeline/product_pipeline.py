"""End-to-end ML pipeline for product defect inspection.

Stages (fail-fast, in order):
  1. validate : tools/dataset/validate_product_dataset.py --root <data_root>
  2. train     : python train.py -c <config> [extra -u overrides]
  3. eval-valid: python train.py -u test_only=true resume=<best.pth>
  4. eval-test : same as (3) with val_dataloader.dataset.root/annotations_path -> test split
  5. promote   : copy best.pth -> ckpts/product_inspection.pth (or --promote-to)
  6. smoke     : optional infer_product.py on --sample-images, or synthetic
                tools/benchmark/smoke_product.py via --synthetic-smoke

Examples (PowerShell, run from repo root):
  python tools/pipeline/product_pipeline.py --help
  python tools/pipeline/product_pipeline.py --dry-run
  python tools/pipeline/product_pipeline.py
  python tools/pipeline/product_pipeline.py --skip-train --best-ckpt outputs/product_inspection/summary/<run>/best.pth
  python tools/pipeline/product_pipeline.py --device cpu --no-test-eval --no-promote
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/product_inspection.yml"
DEFAULT_PROMOTE_TO = "ckpts/product_inspection.pth"


def log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


def run(cmd: list, dry_run: bool = False) -> None:
    log("+ " + " ".join(str(c) for c in cmd))
    if dry_run:
        return
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(str(c) for c in cmd)}")
    log(f"done in {dt:.1f}s")


def find_best_ckpt(summary_dir: str) -> str:
    cands = sorted(
        glob.glob(os.path.join(REPO_ROOT, summary_dir, "**", "best.pth"), recursive=True),
        key=os.path.getmtime,
    )
    if not cands:
        raise FileNotFoundError(f"no best.pth under {summary_dir} - did training write a checkpoint?")
    newest = max(cands, key=os.path.getmtime)
    log(f"using best checkpoint: {newest}")
    return newest


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", default=DEFAULT_CONFIG, help="training config (default: %(default)s)")
    p.add_argument("--data-root", default="data/product_inspection", help="dataset root for validation")
    p.add_argument("--labels", nargs="+", default=None, help="override defect labels for validation")
    p.add_argument("--device", default=None, help="override device, e.g. 'cpu' or 'cuda:0' (passed as device=...)")
    p.add_argument("--train-overrides", nargs="*", default=[], help="extra train.py -u overrides, e.g. epoches=5")
    p.add_argument("--summary-dir", default=None, help="override summary dir glob base (default: from config)")
    p.add_argument("--best-ckpt", default=None, help="skip searching: use this checkpoint for eval/promote")
    p.add_argument("--promote-to", default=DEFAULT_PROMOTE_TO, help="where to copy best.pth (default: %(default)s)")
    p.add_argument("--sample-images", nargs="*", default=[], help="run infer_product.py on these after promote")
    p.add_argument("--synthetic-smoke", action="store_true", help="also run tools/benchmark/smoke_product.py")
    p.add_argument("--skip-validate", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-valid-eval", action="store_true")
    p.add_argument("--no-test-eval", action="store_true", help="skip test-split eval (test.csv may not exist)")
    p.add_argument("--no-promote", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print commands without executing")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    py = sys.executable or "python"
    summary_base = args.summary_dir or "outputs/product_inspection/summary"

    report = {"stages": {}, "config": args.config}

    # 1. validate
    if not args.skip_validate:
        cmd = [py, "tools/dataset/validate_product_dataset.py", "--root", args.data_root]
        if args.labels:
            cmd += ["--labels", *args.labels]
        run(cmd, args.dry_run)
        report["stages"]["validate"] = "ok"
    else:
        log("skip: validate")

    # 2. train
    if not args.skip_train:
        cmd = [py, "train.py", "-c", args.config]
        updates = list(args.train_overrides)
        if args.device:
            updates.append(f"device={args.device}")
        if updates:
            cmd += ["-u", *updates]
        run(cmd, args.dry_run)
        report["stages"]["train"] = "ok"
    else:
        log("skip: train")

    # locate best checkpoint for downstream stages
    best = args.best_ckpt
    need_best = not (args.skip_valid_eval and args.no_test_eval and args.no_promote)
    if need_best and not best and not args.dry_run:
        best = find_best_ckpt(summary_base)
    elif need_best and not best and args.dry_run:
        best = f"<{summary_base}/<run>/best.pth>"
    if best:
        report["best_ckpt"] = best

    # 3. eval valid
    if not args.skip_valid_eval:
        run([py, "train.py", "-c", args.config, "-u", "test_only=true", f"resume={best}"], args.dry_run)
        report["stages"]["eval_valid"] = "ok"
    else:
        log("skip: eval-valid")

    # 4. eval test (override val dataloader to test split)
    if not args.no_test_eval:
        test_root = os.path.join(args.data_root, "test")
        test_csv = os.path.join(args.data_root, "test.csv")
        if not args.dry_run and not (os.path.isdir(os.path.join(REPO_ROOT, test_root)) and os.path.isfile(os.path.join(REPO_ROOT, test_csv))):
            log(f"test split not found ({test_root}, {test_csv}) - skipping test eval")
            report["stages"]["eval_test"] = "skipped:no-test-split"
        else:
            run(
                [
                    py, "train.py", "-c", args.config, "-u",
                    "test_only=true", f"resume={best}",
                    f"val_dataloader.dataset.root={test_root}",
                    f"val_dataloader.dataset.annotations_path={test_csv}",
                ],
                args.dry_run,
            )
            report["stages"]["eval_test"] = "ok"
    else:
        log("skip: eval-test")

    # 5. promote
    if not args.no_promote:
        if args.dry_run:
            log(f"would copy {best} -> {args.promote_to}")
        else:
            dest = REPO_ROOT / args.promote_to
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / best if not os.path.isabs(best) else best, dest)
            log(f"promoted {best} -> {dest}")
        report["stages"]["promote"] = "ok"
    else:
        log("skip: promote")

    # 6a. sample inference
    if args.sample_images:
        weights = args.best_ckpt or best
        if not args.no_promote and not args.best_ckpt:
            weights = args.promote_to
        run(
            [py, "infer_product.py", "--image", *args.sample_images,
             "--config", args.config, "--weights", weights],
            args.dry_run,
        )
        report["stages"]["sample_infer"] = "ok"

    # 6b. synthetic smoke
    if args.synthetic_smoke:
        run([py, "tools/benchmark/smoke_product.py"], args.dry_run)
        report["stages"]["synthetic_smoke"] = "ok"

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
