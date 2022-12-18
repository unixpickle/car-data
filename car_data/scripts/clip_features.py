"""
Compute three-crop CLIP features for all the images in a dataset to apply to
filtering.
"""

import argparse
import itertools
import os
from collections import defaultdict
from typing import Iterator, List, Tuple

import clip
import numpy as np
import torch
from car_data.lin_features import compute_pooled_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_digits", type=int, default=4)
    parser.add_argument("--download_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("image_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(
        "ViT-B/16", device=device, download_root=args.download_root
    )

    print("reading paths...")
    prefixes = ["".join(x) for x in itertools.product(*(["0123456789abcdef"] * 2))]
    listing = sorted(
        x
        for prefix in prefixes
        for x in os.listdir(os.path.join(args.image_dir, prefix))
    )
    print("iterating...")
    for shard_id, filenames in group_by_prefix(listing, args.shard_digits):
        out_path = os.path.join(args.output_dir, f"{shard_id}.npz")
        if os.path.exists(out_path):
            continue
        print(f"working on {out_path}...")
        features = compute_pooled_features(
            device,
            model,
            preprocess,
            [os.path.join(args.image_dir, x[:2], x) for x in filenames],
            batch_size=args.batch_size,
        )
        np.savez(out_path + ".tmp.npz", features=features, filenames=filenames)
        os.rename(out_path + ".tmp.npz", out_path)


def group_by_prefix(
    listing: List[str], prefix_len: int
) -> Iterator[Tuple[str, List[str]]]:
    groups = defaultdict(list)
    for item in listing:
        if len(item) < prefix_len or item.startswith("."):
            continue
        groups[item[:prefix_len]].append(item)
    for k in sorted(groups.keys()):
        yield k, groups[k]


if __name__ == "__main__":
    main()
