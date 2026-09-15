"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from ..core import register
from .muon import Muon as _Muon, MuSGD as _MuSGD


__all__ = [
    "AdamW",
    "SGD",
    "Adam",
    "Muon",
    "MuSGD",
    "MultiStepLR",
    "CosineAnnealingLR",
    "OneCycleLR",
    "LambdaLR",
    "ReduceLROnPlateau",
]



SGD = register()(optim.SGD)
Adam = register()(optim.Adam)
AdamW = register()(optim.AdamW)
Muon = register()(_Muon)
MuSGD = register()(_MuSGD)


MultiStepLR = register()(lr_scheduler.MultiStepLR)
CosineAnnealingLR = register()(lr_scheduler.CosineAnnealingLR)
OneCycleLR = register()(lr_scheduler.OneCycleLR)
LambdaLR = register()(lr_scheduler.LambdaLR)
ReduceLROnPlateau = register()(lr_scheduler.ReduceLROnPlateau)
