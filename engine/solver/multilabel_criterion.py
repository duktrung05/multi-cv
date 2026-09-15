"""BCE multi-label loss (sigmoid at inference)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core import register


@register()
class MultiLabelBCELoss(nn.Module):
    """``BCEWithLogitsLoss`` on ``[B, C]`` vs multi-hot ``[B, C]``."""

    def __init__(
        self,
        pos_weight: list[float] | None = None,
        class_weight: list[float] | None = None,
        label_smoothing: float = 0.0,
        auto_pos_weight: bool = False,
        pos_weight_power: float = 1.0,
        max_pos_weight: float = 10.0,
        eps: float = 1e-6,
        auto_exclusion_pairs: bool = False,
        exclusion_weight: float = 1.0,
        exclusion_max_rate: float = 0.08,
        exclusion_min_count: int = 100,
        subset_f1_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.auto_pos_weight = bool(auto_pos_weight)
        self.pos_weight_power = float(pos_weight_power)
        self.max_pos_weight = float(max_pos_weight)
        self.eps = float(eps)
        self.label_smoothing = float(label_smoothing)
        # Auto-discovered pairs of classes that (almost) never co-occur in the
        # training labels (e.g. MALE_SEXUAL vs FEMALE*). Plain per-class BCE has
        # no notion of "these two logits shouldn't both fire" - it only learns
        # this indirectly and weakly, especially for rare classes. This penalty
        # adds an explicit gradient against joint activation for such pairs.
        self.auto_exclusion_pairs = bool(auto_exclusion_pairs)
        self.exclusion_weight = float(exclusion_weight)
        self.exclusion_max_rate = float(exclusion_max_rate)
        self.exclusion_min_count = int(exclusion_min_count)
        self._exclusion_pairs: list[tuple[int, int]] = []
        # Plain BCE is averaged over every (image, class) cell independently, so it
        # optimizes mean per-class accuracy - a model can nail that while still
        # getting many images' full label set wrong (missing one class, adding a
        # spurious one). This adds a per-*image* soft-F1 term: soft TP/FP/FN are
        # aggregated across classes within each sample, so the gradient rewards
        # matching the whole set for an image, not just each class on average.
        self.subset_f1_weight = float(subset_f1_weight)
        if pos_weight is not None:
            self.register_buffer(
                "_pos_weight",
                torch.tensor([float(x) for x in pos_weight], dtype=torch.float32),
                persistent=False,
            )
        else:
            self.register_buffer("_pos_weight", None, persistent=False)
        # Unlike `pos_weight` (which only reweights positive examples for a class,
        # trading precision/recall on imbalanced classes), `class_weight` scales the
        # whole loss term for a class, so the model is pushed harder to get that
        # class right on both positive and negative examples.
        if class_weight is not None:
            self.register_buffer(
                "_class_weight",
                torch.tensor([float(x) for x in class_weight], dtype=torch.float32),
                persistent=False,
            )
        else:
            self.register_buffer("_class_weight", None, persistent=False)

    def forward(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        epoch: int | None = None,
    ) -> torch.Tensor:
        del epoch
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        y = labels.float()
        if self.label_smoothing > 0:
            y = y * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        pw = self._pos_weight
        if pw is not None:
            pw = pw.to(preds.device)
        cw = self._class_weight
        if cw is None:
            if pw is not None:
                loss = nn.functional.binary_cross_entropy_with_logits(preds, y, pos_weight=pw)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(preds, y)
        else:
            bce = nn.functional.binary_cross_entropy_with_logits(
                preds, y, pos_weight=pw, reduction="none"
            )
            loss = (bce * cw.to(preds.device)).mean()
        return loss + self._exclusion_penalty(preds) + self._subset_f1_loss(preds, y)

    def _exclusion_penalty(self, preds: torch.Tensor) -> torch.Tensor:
        if not self._exclusion_pairs:
            return preds.new_zeros(())
        probs = torch.sigmoid(preds)
        i_idx = [p[0] for p in self._exclusion_pairs]
        j_idx = [p[1] for p in self._exclusion_pairs]
        joint = probs[:, i_idx] * probs[:, j_idx]
        return self.exclusion_weight * joint.mean()

    def _subset_f1_loss(self, preds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.subset_f1_weight <= 0:
            return preds.new_zeros(())
        probs = torch.sigmoid(preds)
        y_hard = (y > 0.5).float()  # undo label smoothing for the set-membership target
        has_pos = y_hard.sum(dim=1) > 0
        tp = (probs * y_hard).sum(dim=1)
        fp = (probs * (1 - y_hard)).sum(dim=1)
        fn = ((1 - probs) * y_hard).sum(dim=1)
        soft_f1 = (2 * tp) / (2 * tp + fp + fn).clamp_min(self.eps)
        # Samples with an empty label set have no well-defined F1 (0/0); for those,
        # just penalize any positive mass directly instead of skipping them.
        per_sample_loss = torch.where(has_pos, 1.0 - soft_f1, probs.mean(dim=1))
        return self.subset_f1_weight * per_sample_loss.mean()

    @torch.no_grad()
    def init_from_dataset(self, dataset) -> None:
        if not (self.auto_pos_weight or self.auto_exclusion_pairs):
            return
        leaf = dataset
        while hasattr(leaf, "dataset"):
            leaf = leaf.dataset
        targets = getattr(leaf, "targets", None)
        if not targets:
            return
        try:
            t = torch.stack(
                [
                    x if isinstance(x, torch.Tensor)
                    else torch.tensor(x, dtype=torch.float32)
                    for x in targets
                ]
            )
        except Exception:
            return
        pos = t.sum(dim=0).float()
        if self.auto_pos_weight:
            neg = (t.shape[0] - pos).clamp_min(self.eps)
            w = (neg / pos.clamp_min(self.eps)).pow(self.pos_weight_power)
            w = w / w.mean().clamp_min(self.eps)
            w = torch.clamp(w, max=self.max_pos_weight)
            self.register_buffer("_pos_weight", w, persistent=False)
        if self.auto_exclusion_pairs:
            self._exclusion_pairs = self._find_exclusion_pairs(t, pos)

    def _find_exclusion_pairs(
        self, t: torch.Tensor, pos: torch.Tensor
    ) -> list[tuple[int, int]]:
        co = t.T @ t  # [C, C] co-occurrence counts
        num_classes = t.shape[1]
        pairs: list[tuple[int, int]] = []
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                if self._is_exclusion_pair(pos[i].item(), pos[j].item(), co[i, j].item()):
                    pairs.append((i, j))
        return pairs

    def _is_exclusion_pair(self, ni: float, nj: float, co_count: float) -> bool:
        if ni < self.exclusion_min_count or nj < self.exclusion_min_count:
            return False
        return co_count / min(ni, nj) <= self.exclusion_max_rate
