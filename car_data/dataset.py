import os
import traceback
from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler


def looping_loader(
    index_path: str,
    image_dir: str,
    batch_size: int,
    train: bool = True,
    center_crop: bool = True,
    last_seen_phash: Optional[str] = None,
) -> Iterator[List["CarImage"]]:
    dataset = CarImageDataset(
        index_path, image_dir, train=train, center_crop=center_crop
    )
    sampler = CarImageDatasetSampler(dataset, last_seen_phash=last_seen_phash)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        collate_fn=lambda x: x,
    )
    while True:
        yield from loader


@dataclass
class CarImage:
    image: torch.Tensor
    phash: str
    price: float
    make: Optional[str]
    model: Optional[str]
    year: Optional[int]


class CarImageDataset(Dataset):
    def __init__(
        self,
        index_path: str,
        image_dir: str,
        train: bool = True,
        center_crop: bool = True,
    ):
        super().__init__()
        self.index_path = index_path
        self.image_dir = image_dir
        with open(index_path, "rb") as f:
            obj = np.load(f)
            phashes = obj["phashes"]
            ordering = np.argsort(phashes)  # sorting hashes => random order

            test_count = len(ordering) // 10
            if train:
                ordering = ordering[test_count:]
            else:
                ordering = ordering[:test_count]

            self.phashes = phashes[ordering]
            self.prices = obj["prices"][ordering]
            self.makes = obj["makes"][ordering].tolist()
            self.models = obj["models"][ordering].tolist()
            self.years = obj["years"][ordering]
        self.transform = transforms.Compose(
            [
                (
                    transforms.CenterCrop(224)
                    if center_crop
                    else transforms.RandomResizedCrop(
                        224, scale=(0.8, 1.0), ratio=(1.0, 1.0)
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.phashes)

    def __getitem__(self, idx: int) -> CarImage:
        img_path = os.path.join(self.image_dir, self.phashes[idx])
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            # Don't kill the job due to a single missing or corrupted image.
            print(f"error loading: {img_path}")
            traceback.print_exc()
            img = Image.new("RGB", (256, 256))
        return CarImage(
            image=self.transform(img),
            phash=self.phashes[idx].tolist(),
            price=self.prices[idx],
            make=self.makes[idx] or None,
            model=self.models[idx] or None,
            year=self.years[idx] or None,
        )


class CarImageDatasetSampler(Sampler):
    def __init__(
        self, data_source: CarImageDataset, last_seen_phash: Optional[str] = None
    ):
        self.data_source = data_source
        self._start_idx = 0
        if last_seen_phash is not None:
            self._start_idx = np.searchsorted(data_source.phashes, last_seen_phash)

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        size = len(self.data_source)
        for i in range(size):
            yield (i + self._start_idx) % size
