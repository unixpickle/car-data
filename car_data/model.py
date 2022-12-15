from typing import Dict, Optional

import clip
import torch
import torch.nn as nn

from .losses import NUM_PRICE_BINS


def create_model(name: str, device: torch.device, download_root: Optional[str] = None):
    if name == "clip":
        return CLIPModel(device, download_root=download_root)
    else:
        raise ValueError(f"unknown model name: {name}")


class CLIPModel(nn.Module):
    def __init__(self, device: torch.device, download_root: Optional[str] = None):
        super().__init__()
        self.device = device
        self.clip, _ = clip.load("ViT-B/16", device=device, download_root=download_root)
        self.clip.float()
        self.price_layer = nn.Linear(512, NUM_PRICE_BINS, device=device)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.clip.encode_image(x)
        return dict(
            price_bin=self.price_layer(feats),
        )
