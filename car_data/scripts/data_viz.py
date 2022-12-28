"""
Dump some images from the data loader to make sure it is working.
"""

import argparse
import os

import torch
from car_data.dataset import looping_loader
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--use_data_aug", action="store_true", default=False)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    loader = looping_loader(
        args.index_path,
        args.image_dir,
        1,
        train=not args.test,
        use_data_aug=args.use_data_aug,
    )
    for i in range(args.count):
        obj = next(loader)[0]
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)
        img = (
            (((obj.image * std) + mean) * 255.99)
            .permute(1, 2, 0)
            .clamp(0, 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        Image.fromarray(img).save(os.path.join(args.output_dir, f"{i}_{obj.price}.png"))


if __name__ == "__main__":
    main()
