"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import time
import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..misc import dist_utils
from ..misc.tqdm import TQDM
from ._solver import BaseSolver
from .clas_engine import train_one_epoch, evaluate


class ClasSolver(BaseSolver):

    def val(self):
        """Run validation / test-only evaluation (optionally with NSFW TensorRT refine)."""
        self.eval()
        module = self.ema.module if self.ema else self.model
        ycfg = self.cfg.yaml_cfg or {}
        nsfw_path = str(ycfg.get("nsfw_engine_path") or "").strip()
        if nsfw_path and not os.path.isabs(nsfw_path):
            nsfw_path = os.path.abspath(nsfw_path)
        use_ref = bool(ycfg.get("nsfw_refine")) and bool(nsfw_path)
        ep = nsfw_path if use_ref else None
        test_stats = evaluate(
            module,
            self.criterion,
            self.val_dataloader,
            self.device,
            nsfw_engine_path=ep,
        )
        self._last_val_stats = test_stats
        if dist_utils.is_main_process():
            skip_keys = frozenset({"confmat", "class_names"})
            msg = "eval " + " ".join(
                f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in test_stats.items()
                if k not in skip_keys
            )
            TQDM.write(msg + "\n")
        return test_stats

    def fit(self, ):
        self.train()
        args = self.cfg

        # Auto-initialize criterion (e.g., class weights) from training dataset, if supported.
        try:
            if hasattr(self.criterion, "init_from_dataset"):
                self.criterion.init_from_dataset(self.train_dataloader.dataset)
        except Exception as e:
            if dist_utils.is_main_process():
                TQDM.write(f"[WARN] criterion.init_from_dataset() failed: {e}")

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        # Save lightweight weights next to TensorBoard event files for this run.
        weights_dir = Path(self.writer.log_dir) if self.writer is not None else output_dir
        weights_dir.mkdir(parents=True, exist_ok=True)

        # Track best metric (prefer macro_f1 when available).
        best_metric = float("-inf")
        save_checkpoints = getattr(args, 'save_checkpoints', False)

        start_time = time.time()
        start_epoch = self.last_epoch + 1
        epochs_iter = range(start_epoch, args.epoches)
        bar = None
        if dist_utils.is_main_process():
            bar = TQDM(
                epochs_iter,
                desc='Progress',
                total=(args.epoches - start_epoch),
                leave=False,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {postfix}'
            )

        for epoch in (bar if bar is not None else epochs_iter):

            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(self.model,
                                        self.criterion,
                                        self.train_dataloader,
                                        self.optimizer,
                                        self.ema,
                                        epoch=epoch,
                                        device=self.device,
                                        lr_warmup_scheduler=self.lr_warmup_scheduler)
            self.last_epoch += 1

            module = self.ema.module if self.ema else self.model
            test_stats = evaluate(module, self.criterion, self.val_dataloader, self.device)
            self._last_val_stats = test_stats

            # Step LR scheduler (epoch-based vs plateau).
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                monitor = float(test_stats.get("macro_f1", test_stats.get("acc", 0.0)))
                self.lr_scheduler.step(monitor)
            else:
                self.lr_scheduler.step()

            # Lightweight weights-only checkpoint for eval/deploy.
            module_to_save = dist_utils.de_parallel(module)
            weights_state = module_to_save.state_dict()
            dist_utils.save_on_master({'model': weights_state}, weights_dir / 'last.pth')

            # TensorBoard scalars (lr/acc/loss) per epoch.
            # Classification previously only logged config as TEXT, so Scalars tab was empty.
            if self.writer and dist_utils.is_main_process():
                self.writer.add_scalar('Train/lr', float(train_stats.get('lr', 0.0)), epoch)
                self.writer.add_scalar('Train/loss', float(train_stats.get('loss', 0.0)), epoch)
                self.writer.add_scalar('Valid/acc', float(test_stats.get('acc', 0.0)), epoch)
                self.writer.add_scalar('Valid/loss', float(test_stats.get('loss', 0.0)), epoch)
                if "macro_f1" in test_stats:
                    self.writer.add_scalar('Valid/macro_f1', float(test_stats.get('macro_f1', 0.0)), epoch)

            if bar is not None:
                bar.set_postfix(
                    lr=f"{float(train_stats.get('lr', 0.0)):.3g}",
                    loss=f"{float(test_stats.get('loss', 0.0)):.2f}",
                    acc=f"{float(test_stats.get('acc', 0.0)):.2f}",
                    f1=f"{float(test_stats.get('macro_f1', 0.0)):.2f}" if "macro_f1" in test_stats else "-",
                )

            cur_metric = float(test_stats.get("macro_f1", test_stats.get("acc", float("-inf"))))
            if cur_metric > best_metric:
                best_metric = cur_metric
                if dist_utils.is_main_process():
                    # Single boxed line for "new best" events.
                    if "macro_f1" in test_stats:
                        msg = f"NEW BEST epoch={epoch} macro_f1={best_metric:.4f} acc={float(test_stats.get('acc', 0.0)):.4f}"
                    else:
                        msg = f"NEW BEST epoch={epoch} acc={best_metric:.4f}"
                    pad = 2
                    inner_width = max(10, len(msg) + pad * 2)
                    box_border = "+" + "-" * (inner_width + 2) + "+"
                    content = "| " + msg.ljust(inner_width) + " |"
                    box_msg = f"\n{box_border}\n{content}\n{box_border}\n"
                    if bar is not None:
                        bar.write(box_msg)
                        bar.refresh()
                    else:
                        TQDM.write(box_msg)
                dist_utils.save_on_master({'model': weights_state}, weights_dir / 'best.pth')

            # Optionally save full training checkpoint for resume (big files).
            if save_checkpoints:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every N epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(epoch), checkpoint_path)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        _ = total_time_str  # no console prints; rely on TQDM/TensorBoard
