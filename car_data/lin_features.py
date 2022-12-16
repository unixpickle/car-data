from multiprocessing.pool import ThreadPool
from typing import Callable, Iterator, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def compute_pooled_features(
    device: torch.device,
    model: nn.Module,
    preprocess: Callable[[Image.Image], torch.Tensor],
    paths: List[str],
    batch_size: int = 64,
) -> np.ndarray:
    with ThreadPool(8) as p:
        all_outs = []
        for chunk in chunk_filenames(paths, batch_size):
            crops = [x for y in p.map(image_crops, chunk) for x in y]
            tensors = torch.stack(p.map(preprocess, crops), dim=0).to(device)
            with torch.no_grad():
                features_out = (
                    model.encode_image(tensors).reshape([len(chunk), 3, -1]).mean(1)
                )
                features_out /= torch.linalg.norm(features_out, dim=-1, keepdim=True)
            all_outs.append(features_out.cpu().numpy())
        return np.concatenate(all_outs, axis=0)


def chunk_filenames(paths: List[str], batch_size: int) -> Iterator[List[str]]:
    for i in range(0, len(paths), batch_size):
        yield paths[i : i + batch_size]


def image_crops(path: str):
    img = Image.open(path)
    width, height = img.size
    min_dim = min(width, height)
    cx = width // 2 - min_dim
    cy = height // 2 - min_dim
    if width > height:
        crops = [
            (0, 0, height, height),
            (cx, 0, cx + height, height),
            (width - height, 0, width, height),
        ]
    else:
        crops = [
            (0, 0, width, width),
            (0, cy, width, cy + width),
            (0, height - width, width, height),
        ]
    outs = []
    for box in crops:
        outs.append(img.crop(box=box))
    return outs
