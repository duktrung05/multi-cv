"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DINOv3 (https://github.com/facebookresearch/dinov3)

Copyright (c) Meta Platforms, Inc. and affiliates.

This software may be used and distributed in accordance with
the terms of the DINOv3 License Agreement.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from functools import partial
from ..core import register
from .vit_tiny import VisionTransformer
from .dinov3 import DinoVisionTransformer


def _safe_torch_load_weights(path: str):
    """
    Avoid PyTorch FutureWarning about `weights_only=False` by explicitly
    requesting weights-only when supported. Fallback for older torch versions.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # `weights_only` arg not supported in this torch version.
        return torch.load(path, map_location="cpu")


class SpatialPriorModulev2(nn.Module):
    def __init__(self, inplanes=16):
        super().__init__()

        # 1/4
        self.stem = nn.Sequential(
            *[
                nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        # 1/8
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(2 * inplanes),
            ]
        )
        # 1/16
        self.conv3 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )
        # 1/32
        self.conv4 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)     # 1/8
        c3 = self.conv3(c2)     # 1/16
        c4 = self.conv4(c3)     # 1/32

        return c2, c3, c4


@register()
class DINOv3STAs(nn.Module):
    def __init__(
        self,
        name=None,
        weights_path=None,
        interaction_indexes=[],
        finetune=True,
        freeze_blocks=0,
        embed_dim=192,
        num_heads=3,
        patch_size=16,
        use_sta=True,
        conv_inplane=16,
        hidden_dim=None,
        drop_path_rate=0.,
        use_rope=True,
        use_abs_pos_embed=False,
        img_size=384,
    ):
        super(DINOv3STAs, self).__init__()
        # Simple selection rule:
        # - if `name` contains "dinov3" -> use DINOv3
        # - otherwise -> use VisionTransformer (ViT)
        use_dinov3 = name is not None and "dinov3" in str(name).lower()

        if use_dinov3:
            self.dinov3 = DinoVisionTransformer(name=name)
            if weights_path and os.path.exists(weights_path):
                self.dinov3.load_state_dict(_safe_torch_load_weights(weights_path))
                print(f'Loaded DINOv3 weights from {weights_path}')
        else:
            # ViT-style backbone
            self.dinov3 = VisionTransformer(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                return_layers=interaction_indexes,
                drop_path_rate=drop_path_rate,
                use_rope=use_rope,
                use_abs_pos_embed=use_abs_pos_embed,
            )
            if weights_path and os.path.exists(weights_path):
                self.dinov3._model.load_state_dict(_safe_torch_load_weights(weights_path))
                print(f'Loaded ViT weights from {weights_path}')
        embed_dim = self.dinov3.embed_dim
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size

        # Modules that are frozen (requires_grad=False) are also kept in eval() mode,
        # so stateful training-only ops (DropPath, Dropout) don't inject noise into
        # features that are supposed to be fixed. `nn.Module.train(mode=True)` recurses
        # into *all* children regardless of requires_grad, so a plain `.eval()` call
        # here would get silently undone the next time the outer model's `.train()` is
        # called (e.g. at the start of every training epoch) - `train()` is overridden
        # below to re-apply this after every such call.
        self._frozen_modules = []

        if not finetune:
            self.dinov3.requires_grad_(False)
            self._frozen_modules.append(self.dinov3)
        elif freeze_blocks > 0:
            # Partial freeze: keep patch embedding + first `freeze_blocks` transformer
            # blocks frozen (generic low-level features), finetune the rest.
            # `VisionTransformer` (vit_tiny.py, non-dinov3 path) nests these under
            # `_model` instead of exposing them directly like `DinoVisionTransformer`.
            freeze_target = getattr(self.dinov3, "_model", self.dinov3)
            freeze_target.patch_embed.requires_grad_(False)
            freeze_target.cls_token.requires_grad_(False)
            self._frozen_modules.append(freeze_target.patch_embed)
            if getattr(freeze_target, "storage_tokens", None) is not None:
                freeze_target.storage_tokens.requires_grad_(False)
            for blk in freeze_target.blocks[:freeze_blocks]:
                blk.requires_grad_(False)
                self._frozen_modules.append(blk)

        for m in self._frozen_modules:
            m.eval()

        # init the feature pyramid
        self.use_sta = use_sta
        if use_sta:
            self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        else:
            conv_inplane = 0

        # linear projection
        hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.convs = nn.ModuleList([
            nn.Conv2d(embed_dim + conv_inplane*2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        ])
        # norm
        self.norms = nn.ModuleList([
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim)
        ])

    def train(self, mode=True):
        super().train(mode)
        # Re-freeze after the recursive `.train()` call above, which unconditionally
        # sets `.training = True` on every submodule regardless of requires_grad.
        for m in self._frozen_modules:
            m.eval()
        return self

    def forward(self, x):
        # Code for matching with oss
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs, C, h, w = x.shape

        if len(self.interaction_indexes) > 0 and not isinstance(self.dinov3, VisionTransformer):
            all_layers = self.dinov3.get_intermediate_layers(
                x, n=self.interaction_indexes, return_class_token=True
            )
        else:
            all_layers = self.dinov3(x)

        if len(all_layers) == 1:    # repeat the same layer for all the three scales
            all_layers = [all_layers[0], all_layers[0], all_layers[0]]
        
        sem_feats = []
        num_scales = len(all_layers) - 2
        for i, sem_feat in enumerate(all_layers):
            feat, _ = sem_feat
            sem_feat = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()  # [B, D, H, W]
            resize_H, resize_W = int(H_c * 2**(num_scales-i)), int(W_c * 2**(num_scales-i))
            sem_feat = F.interpolate(sem_feat, size=[resize_H, resize_W], mode="bilinear", align_corners=False)
            sem_feats.append(sem_feat)

        # fusion
        fused_feats = []
        if self.use_sta:
            detail_feats = self.sta(x)
            for sem_feat, detail_feat in zip(sem_feats, detail_feats):
                fused_feats.append(torch.cat([sem_feat, detail_feat], dim=1))
        else:
            fused_feats = sem_feats

        c2 = self.norms[0](self.convs[0](fused_feats[0]))
        c3 = self.norms[1](self.convs[1](fused_feats[1]))
        c4 = self.norms[2](self.convs[2](fused_feats[2]))

        return c2, c3, c4