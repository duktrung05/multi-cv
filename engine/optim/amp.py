"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch

from ..core import register

__all__ = ["GradScaler"]


if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
    _AmpGradScaler = torch.amp.GradScaler

    @register()
    class GradScaler(_AmpGradScaler):
        """YAML-registered AMP scaler using ``torch.amp.GradScaler('cuda', ...)`` (avoids deprecated ``torch.cuda.amp.GradScaler``)."""

        def __init__(
            self,
            init_scale=2.0**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
            enabled=True,
        ):
            super().__init__(
                "cuda",
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
                enabled=enabled,
            )
else:  # pragma: no cover
    import torch.cuda.amp as cuda_amp

    GradScaler = register()(cuda_amp.grad_scaler.GradScaler)
