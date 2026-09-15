from __future__ import annotations

import torch
import torch.nn as nn

from ..core import register


@register()
class ClassificationCrossEntropy(nn.Module):
    """
    Classification criterion used by `ClasSolver`.

    `ClasEngine` calls:
      - train: loss = criterion(preds, labels, epoch)
      - eval:  loss = criterion(preds, labels)
    """

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        # PyTorch supports label_smoothing for CrossEntropyLoss.
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        epoch: int | None = None,
    ) -> torch.Tensor:
        del epoch  # epoch không ảnh hưởng tới CrossEntropyLoss
        return self.loss_fn(preds, labels)


@register()
class ClassificationWeightedCrossEntropy(nn.Module):
    """
    CrossEntropyLoss with (optional) per-class weights.

    Supports auto-computing weights from the train dataset via `init_from_dataset()`.
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        class_weights: list[float] | None = None,
        auto_weight: bool = True,
        weight_power: float = 1.0,
        max_weight: float = 10.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.auto_weight = bool(auto_weight)
        self.weight_power = float(weight_power)
        self.max_weight = float(max_weight)
        self.eps = float(eps)

        self.register_buffer("_weight", None, persistent=False)
        if class_weights is not None:
            w = torch.tensor([float(x) for x in class_weights], dtype=torch.float32)
            self._weight = w

    @torch.no_grad()
    def init_from_dataset(self, dataset) -> None:
        if not self.auto_weight:
            return
        targets = getattr(dataset, "targets", None)
        if targets is None:
            return
        if not isinstance(targets, (list, tuple)) or len(targets) == 0:
            return
        # Compute weights ~ inverse frequency^power, normalized to mean=1.
        t = torch.tensor(targets, dtype=torch.int64)
        num_classes = int(t.max().item()) + 1
        counts = torch.bincount(t, minlength=num_classes).float()
        inv = (counts + self.eps).pow(-self.weight_power)
        inv = inv / inv.mean().clamp_min(self.eps)
        inv = torch.clamp(inv, max=self.max_weight)
        self._weight = inv

    def forward(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        epoch: int | None = None,
    ) -> torch.Tensor:
        del epoch
        weight = self._weight
        if weight is not None and weight.device != preds.device:
            weight = weight.to(preds.device)
        loss_fn = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=self.label_smoothing,
        )
        return loss_fn(preds, labels)


@register()
class ClassificationFocalLoss(nn.Module):
    """
    Multi-class focal loss on logits, optionally with class weights (alpha).

    FL = alpha_y * (1 - p_y)^gamma * CE(logits, y)

    `init_from_dataset()` can auto-compute alpha from class frequency (same scheme as weighted CE).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: list[float] | None = None,
        auto_weight: bool = False,
        weight_power: float = 1.0,
        max_weight: float = 10.0,
        eps: float = 1e-12,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.auto_weight = bool(auto_weight)
        self.weight_power = float(weight_power)
        self.max_weight = float(max_weight)
        self.eps = float(eps)
        self.reduction = str(reduction)

        self.register_buffer("_alpha", None, persistent=False)
        if class_weights is not None:
            a = torch.tensor([float(x) for x in class_weights], dtype=torch.float32)
            self._alpha = a

    @torch.no_grad()
    def init_from_dataset(self, dataset) -> None:
        if not self.auto_weight:
            return
        targets = getattr(dataset, "targets", None)
        if targets is None:
            return
        if not isinstance(targets, (list, tuple)) or len(targets) == 0:
            return
        t = torch.tensor(targets, dtype=torch.int64)
        num_classes = int(t.max().item()) + 1
        counts = torch.bincount(t, minlength=num_classes).float()
        inv = (counts + self.eps).pow(-self.weight_power)
        inv = inv / inv.mean().clamp_min(self.eps)
        inv = torch.clamp(inv, max=self.max_weight)
        self._alpha = inv

    def forward(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        epoch: int | None = None,
    ) -> torch.Tensor:
        del epoch
        logp = torch.log_softmax(preds, dim=-1)
        p = logp.exp()
        y = labels.long()
        logp_y = logp.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
        p_y = p.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)

        focal = (1.0 - p_y).clamp_min(0.0).pow(self.gamma)
        ce = -logp_y
        loss = focal * ce

        alpha = self._alpha
        if alpha is not None:
            if alpha.device != preds.device:
                alpha = alpha.to(preds.device)
            loss = loss * alpha.gather(dim=0, index=y).to(loss.dtype)

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        # mean
        return loss.mean()

