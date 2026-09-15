"""Multi-label classifiers (sigmoid / BCE at train time)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core import register
from .classification_adapter import AttentionPool2d


class GlobalAvgPool2d(nn.Module):
    """Global average pooling over spatial locations: [B, C, H, W] -> [B, C].

    Unlike ``AttentionPool2d``, this has no learned parameters, so every
    spatial location contributes equally to the pooled feature and Grad-CAM
    heatmaps computed against the pre-pool feature map directly reflect what
    the classifier head weighs - the standard setup CAM/Grad-CAM assume.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3))


def _build_feature_head(
    backbone: nn.Module,
    use_all_scales: bool,
    attn_hidden_dim: int,
    pooling: str = "attn",
) -> tuple[nn.ModuleList, int]:
    hidden_dim = getattr(backbone.convs[0], "out_channels", None)
    if hidden_dim is None:
        raise ValueError("backbone.convs[0].out_channels required")
    num_pools = 3 if use_all_scales else 1
    if pooling == "gap":
        pools = nn.ModuleList([GlobalAvgPool2d() for _ in range(num_pools)])
    elif pooling == "attn":
        pools = nn.ModuleList(
            [AttentionPool2d(hidden_dim, attn_hidden_dim) for _ in range(num_pools)]
        )
    else:
        raise ValueError(f"unknown pooling: {pooling!r} (expected 'attn' or 'gap')")
    in_features = hidden_dim * num_pools
    return pools, in_features


@register()
class DINOv3STAsMultiLabelClassifier(nn.Module):
    """DINOv3 + MLP → ``num_classes`` logits (multi-label, use sigmoid at inference)."""

    __inject__ = ["backbone"]

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        use_all_scales: bool = True,
        attn_hidden_dim: int = 128,
        mlp_hidden_dim: int | None = None,
        head_dropout: float = 0.0,
        pooling: str = "attn",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.use_all_scales = bool(use_all_scales)
        self.pooling = str(pooling)
        self.attn_pools, in_features = _build_feature_head(
            backbone, self.use_all_scales, attn_hidden_dim, self.pooling
        )
        if mlp_hidden_dim is None:
            mlp_hidden_dim = max(in_features // 2, self.num_classes * 2)
        self.head = nn.Sequential(
            nn.Linear(in_features, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(mlp_hidden_dim, self.num_classes),
        )

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        c2, c3, c4 = self.backbone(x)
        if self.use_all_scales:
            pooled = [
                self.attn_pools[0](c2),
                self.attn_pools[1](c3),
                self.attn_pools[2](c4),
            ]
            return torch.cat(pooled, dim=1)
        return self.attn_pools[0](c4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._features(x))
