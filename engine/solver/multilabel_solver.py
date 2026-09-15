"""Solver for ``task: multilabel_classification`` (BCE + sigmoid inference)."""

from __future__ import annotations

import os
import time
import datetime
from pathlib import Path

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..misc import dist_utils
from ..misc.plotting import MetricsPlotter
from ..misc.tqdm import TQDM
from ._solver import BaseSolver
from .multilabel_engine import evaluate, train_one_epoch


class MultilabelClasSolver(BaseSolver):

    def val(self):
        self.eval()
        module = self.ema.module if self.ema else self.model
        ycfg = self.cfg.yaml_cfg or {}
        thr = float(ycfg.get("multilabel_threshold", 0.5))
        test_stats = evaluate(
            module,
            self.criterion,
            self.val_dataloader,
            self.device,
            threshold=thr,
        )
        self._last_val_stats = test_stats
        if dist_utils.is_main_process():
            msg = "eval " + " ".join(
                f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in test_stats.items()
                if k not in ("class_names",) and not str(k).startswith("latency_")
            )
            TQDM.write(msg + "\n")
        return test_stats

    def fit(self):
        self.train()
        args = self.cfg

        try:
            if hasattr(self.criterion, "init_from_dataset"):
                self.criterion.init_from_dataset(self.train_dataloader.dataset)
        except Exception as e:
            if dist_utils.is_main_process():
                TQDM.write(f"[WARN] criterion.init_from_dataset() failed: {e}")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        weights_dir = Path(self.writer.log_dir) if self.writer is not None else output_dir
        weights_dir.mkdir(parents=True, exist_ok=True)

        best_metric = float("-inf")
        metrics_plotter = (
            MetricsPlotter(output_dir=str(weights_dir), acc_label="Macro F1")
            if dist_utils.is_main_process()
            else None
        )
        ycfg = self.cfg.yaml_cfg or {}
        esp = int(ycfg.get("early_stopping_patience", 0) or 0)
        early_stopper = None
        if esp > 0:
            from ultralytics.utils.torch_utils import EarlyStopping

            early_stopper = EarlyStopping(patience=esp)

        start_time = time.time()
        start_epoch = self.last_epoch + 1
        epochs_iter = range(start_epoch, args.epoches)
        bar = None
        if dist_utils.is_main_process():
            bar = TQDM(
                epochs_iter,
                desc="Progress",
                total=(args.epoches - start_epoch),
                leave=False,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {postfix}",
            )

        for epoch in (bar if bar is not None else epochs_iter):
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.ema,
                epoch=epoch,
                device=self.device,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                max_norm=float(ycfg.get("clip_max_norm", 0.0) or 0.0),
            )
            self.last_epoch += 1

            module = self.ema.module if self.ema else self.model
            test_stats = evaluate(
                module,
                self.criterion,
                self.val_dataloader,
                self.device,
                threshold=float(ycfg.get("multilabel_threshold", 0.5)),
            )
            self._last_val_stats = test_stats

            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                monitor = float(test_stats.get("macro_f1", 0.0))
                self.lr_scheduler.step(monitor)
            else:
                self.lr_scheduler.step()

            module_to_save = dist_utils.de_parallel(module)
            weights_state = module_to_save.state_dict()
            checkpoint = {
                "model": weights_state,
                "class_list": list(ycfg.get("class_list") or []),
                "task_domain": ycfg.get("task_domain"),
            }
            dist_utils.save_on_master(checkpoint, weights_dir / "last.pth")

            if self.writer and dist_utils.is_main_process():
                self.writer.add_scalar("Train/lr", float(train_stats.get("lr", 0.0)), epoch)
                self.writer.add_scalar("Train/loss", float(train_stats.get("loss", 0.0)), epoch)
                self.writer.add_scalar("Train/ml_f1", float(train_stats.get("ml_f1", 0.0)), epoch)
                self.writer.add_scalar("Valid/ml_f1", float(test_stats.get("ml_f1", 0.0)), epoch)
                self.writer.add_scalar("Valid/loss", float(test_stats.get("loss", 0.0)), epoch)
                self.writer.add_scalar("Valid/acc", float(test_stats.get("acc", 0.0)), epoch)

            if metrics_plotter is not None:
                metrics_plotter.update(
                    epoch=epoch,
                    train_loss=float(train_stats.get("loss", 0.0)),
                    val_loss=float(test_stats.get("loss", 0.0)),
                    train_acc=float(train_stats.get("ml_f1", 0.0)),
                    val_acc=float(test_stats.get("macro_f1", test_stats.get("ml_f1", 0.0))),
                )
            if bar is not None:
                bar.set_postfix(
                    lr=f"{float(train_stats.get('lr', 0.0)):.3g}",
                    loss=f"{float(train_stats.get('loss', 0.0)):.2f}",
                    f1=f"{float(test_stats.get('ml_f1', 0.0)):.2f}",
                )

            cur_metric = float(test_stats.get("macro_f1", float("-inf")))
            if cur_metric > best_metric:
                best_metric = cur_metric
                if dist_utils.is_main_process():
                    TQDM.write(
                        f"\n+{'-'*40}+\n| NEW BEST epoch={epoch} macro_f1={best_metric:.4f} |\n+{'-'*40}+\n"
                    )
                dist_utils.save_on_master(checkpoint, weights_dir / "best.pth")

            if early_stopper is not None and early_stopper(epoch, cur_metric):
                if dist_utils.is_main_process():
                    TQDM.write(f"[EarlyStopping] stopped at epoch={epoch}\n")
                break

        _ = time.time() - start_time
