from __future__ import annotations

import torch
import torch.nn as nn

from ..core import register


class AttentionPool2d(nn.Module):
    """
    Attention pooling over spatial locations.

    Input:  x of shape [B, C, H, W]
    Output: pooled of shape [B, C]
    """

    def __init__(self, channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).transpose(1, 2)  # [B, N, C]
        scores = self.score(x_flat).squeeze(-1)  # [B, N]
        attn = torch.softmax(scores, dim=-1)  # [B, N]
        pooled = (attn.unsqueeze(-1) * x_flat).sum(dim=1)  # [B, C]
        return pooled


@register()
class DINOv3STAsClassifier(nn.Module):
    """
    Image classifier on top of `DINOv3STAs` features.

    - Extracts (c2, c3, c4) from the backbone
    - Attention pooling on each scale
    - Concatenates pooled features and applies an MLP head
    """

    __inject__ = ["backbone"]

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        use_all_scales: bool = True,
        attn_hidden_dim: int = 128,
        mlp_hidden_dim: int | None = None,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.use_all_scales = bool(use_all_scales)

        hidden_dim = None
        if hasattr(backbone, "convs") and backbone.convs is not None and len(backbone.convs) > 0:
            # DINOv3STAs builds `self.convs` as projection layers: out_channels == hidden_dim
            hidden_dim = getattr(backbone.convs[0], "out_channels", None)

        if hidden_dim is None:
            raise ValueError(
                "Cannot infer classifier hidden dim from backbone. "
                "Expected backbone.convs[0].out_channels to exist."
            )

        in_features = hidden_dim * (3 if self.use_all_scales else 1)

        # One attention-pooling module per scale.
        num_pools = 3 if self.use_all_scales else 1
        self.attn_pools = nn.ModuleList(
            [AttentionPool2d(hidden_dim, attn_hidden_dim) for _ in range(num_pools)]
        )

        if mlp_hidden_dim is None:
            # Reasonable default: half of concat dim, at least a small multiple of classes.
            mlp_hidden_dim = max(in_features // 2, self.num_classes * 2)

        self.head = nn.Sequential(
            nn.Linear(in_features, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(mlp_hidden_dim, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c2, c3, c4 = self.backbone(x)
        if self.use_all_scales:
            pooled = [
                self.attn_pools[0](c2),
                self.attn_pools[1](c3),
                self.attn_pools[2](c4),
            ]
            feat = torch.cat(pooled, dim=1)  # [B, hidden_dim * 3]
        else:
            feat = self.attn_pools[0](c4)  # [B, hidden_dim]
        return self.head(feat)

