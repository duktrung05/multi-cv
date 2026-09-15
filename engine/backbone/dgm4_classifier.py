"""Linear probe on the existing DINOv3 implementation; backbone stays frozen."""
import torch
from torch import nn

from .dinov3 import DinoVisionTransformer


class DINOv3BinaryClassifier(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        self.backbone = backbone if backbone is not None else DinoVisionTransformer(name="dinov3_vits16")
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        # Concatenate the CLS token and mean patch token from the final block.
        self.head = nn.Linear(self.backbone.embed_dim * 2, 2)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def extract_features(self, images):
        features = self.backbone.forward_features(images)
        return torch.cat((features["x_norm_clstoken"],
                          features["x_norm_patchtokens"].mean(dim=1)), dim=1)

    def forward(self, images):
        return self.head(self.extract_features(images))
