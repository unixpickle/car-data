"""
Compute three-crop CLIP features for all the images in a dataset to apply to
filtering.
"""

import argparse
import itertools
import os
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

import clip
import numpy as np
import torch
from car_data.lin_features import compute_pooled_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_digits", type=int, default=4)
    parser.add_argument("--download_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--old_feature_dir", type=str, default=None)
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

        old_features = None
        if args.old_feature_dir:
            old_path = os.path.join(args.old_feature_dir, f"{shard_id}.npz")
            if os.path.exists(old_path):
                old_features = dict(np.load(old_path))

        filenames = filter_out_existing_filenames(old_features, filenames)
        if not len(filenames):
            assert old_features is not None
            out_dict = old_features
        else:
            print(f"working on {out_path}...")
            features = compute_pooled_features(
                device,
                model,
                preprocess,
                [os.path.join(args.image_dir, x[:2], x) for x in filenames],
                batch_size=args.batch_size,
            )
            out_dict = combine_existing_features(old_features, filenames, features)
        np.savez(out_path + ".tmp.npz", **out_dict)
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


def filter_out_existing_filenames(
    old_features: Optional[Dict[str, np.ndarray]], filenames: List[str]
) -> List[str]:
    if old_features is None:
        return filenames
    old_set = set(old_features["filenames"].tolist())
    return [x for x in filenames if x not in old_set]


def combine_existing_features(
    old_features: Optional[Dict[str, np.ndarray]],
    new_filenames: List[str],
    new_features: np.ndarray,
) -> Dict[str, np.ndarray]:
    if old_features is None:
        return dict(features=new_features, filenames=new_filenames)
    all_filenames = np.array(old_features["filenames"].tolist() + new_filenames)
    all_features = np.concatenate([old_features["features"], new_features])
    sorted_indices = np.argsort(all_filenames)
    return dict(
        features=all_features[sorted_indices], filenames=all_filenames[sorted_indices]
    )


if __name__ == "__main__":
    main()
