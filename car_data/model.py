from typing import Dict, Optional

import clip
import torch
import torch.nn as nn

from .losses import MEDIAN_PRICE_SCALE, NUM_PRICE_BINS


def create_model(name: str, device: torch.device, download_root: Optional[str] = None):
    if name == "clip":
        return CLIPModel(device, download_root=download_root)
    elif name == "mobilenetv2":
        return MobileNetV2Model(device, download_root=download_root)
    else:
        raise ValueError(f"unknown model name: {name}")


class CLIPModel(nn.Module):
    def __init__(self, device: torch.device, download_root: Optional[str] = None):
        super().__init__()
        self.device = device
        self.clip, _ = clip.load("ViT-B/16", device=device, download_root=download_root)
        self.clip.float()
        self.price_layer = nn.Linear(512, NUM_PRICE_BINS, device=device)
        self.median_layer = nn.Linear(512, 1, device=device)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.clip.encode_image(x)
        return dict(
            price_bin=self.price_layer(feats),
            price_median=self.median_layer(feats)[..., 0] * MEDIAN_PRICE_SCALE,
        )


class MobileNetV2Model(nn.Module):
    def __init__(self, device: torch.device, download_root: Optional[str] = None):
        super().__init__()
        if download_root is not None:
            backup_dir = torch.hub.get_dir()
            torch.hub.set_dir(download_root)
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True
        ).to(device)
        if download_root is not None:
            torch.hub.set_dir(backup_dir)
        self.model.classifier[1] = nn.Linear(1280, NUM_PRICE_BINS + 1, device=device)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        all_outputs = self.model(x)
        return dict(
            price_bin=all_outputs[..., :NUM_PRICE_BINS],
            price_median=all_outputs[..., NUM_PRICE_BINS] * MEDIAN_PRICE_SCALE,
        )
