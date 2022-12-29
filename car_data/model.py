from typing import Dict, Optional

import clip
import torch
import torch.nn as nn

from .constants import MEDIAN_PRICE_SCALE, NUM_MAKE_MODELS, NUM_PRICE_BINS, NUM_YEARS


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
        self.output = OutputLayer(512, device=device)

    def output_layer(self) -> "OutputLayer":
        return self.output

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.clip.encode_image(x)
        return self.output(h)


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
        self.model.classifier[1] = OutputLayer(1280, device=device)

    def output_layer(self) -> "OutputLayer":
        return self.model.classifier[1]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)


class OutputLayer(nn.Module):
    def __init__(self, n_features: int, device: torch.device):
        super().__init__()
        self.price_bin = nn.Linear(n_features, NUM_PRICE_BINS, device=device)
        self.price_median = nn.Linear(n_features, 1, device=device)
        self.make_model = nn.Linear(n_features, NUM_MAKE_MODELS, device=device)
        self.year = nn.Linear(n_features, NUM_YEARS, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return dict(
            price_bin=self.price_bin(x),
            price_median=self.price_median(x)[..., 0] * MEDIAN_PRICE_SCALE,
            make_model=self.make_model(x),
            year=self.year(x),
        )

    def scale_outputs(self, scales: Dict[str, float]):
        with torch.no_grad():
            for key, scale in scales.items():
                layer = getattr(self, key)
                layer.weight.mul_(scale)
                layer.bias.mul_(scale)
