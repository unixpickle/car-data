"""
Train a simple classifier on pooled features.
"""

import argparse
import os
from collections import defaultdict
from typing import Any, Iterator, List, Tuple

import clip
import numpy as np
import sk2torch
import torch
from car_data.lin_features import compute_pooled_features
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--negative_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--download_root", type=str, default=None)
    parser.add_argument("--model_out", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(
        "ViT-B/16", device=device, download_root=args.download_root
    )

    print("computing positive features...")
    positive_features = compute_pooled_features(
        device=device,
        model=model,
        preprocess=preprocess,
        paths=list(list_dirs(args.positive_dirs)),
    )

    print("computing negative features...")
    negative_features = compute_pooled_features(
        device=device,
        model=model,
        preprocess=preprocess,
        paths=list(list_dirs(args.negative_dirs)),
    )
    inputs = np.concatenate([positive_features, negative_features], axis=0)
    labels = np.array(
        [True] * len(positive_features) + [False] * len(negative_features)
    )

    clf = SVC(random_state=0)

    print("validating...")
    preds = cross_val_preds(clf, inputs, labels, folds=10)
    for threshold in np.linspace(-2.0, 2.0, num=31):
        acc = np.mean((preds > threshold) == labels)
        false_neg = np.mean(((preds > threshold) != labels)[labels])
        filter_frac = np.mean(((preds > threshold) == labels)[~labels])
        print(
            f"threshold {threshold:.02}: acc={acc:.03f} false_neg={false_neg:.03f} filter_frac={filter_frac:.03f}"
        )

    print("training...")
    clf.fit(inputs, labels)

    print("saving...")
    save_model = torch.jit.script(sk2torch.wrap(clf).float())
    torch.jit.save(save_model, args.model_out)


def cross_val_preds(
    model: Any, xs: np.ndarray, ys: np.ndarray, folds: int
) -> np.ndarray:
    """
    Compute out-of-fold decision function outputs for all of the samples.
    """
    perm = np.random.permutation(len(xs))
    chunk_size = len(xs) // folds
    chunk_sizes = [chunk_size + int(i < len(xs) % folds) for i in range(folds)]
    index_chunks = np.split(perm, np.cumsum(chunk_sizes)[:-1])

    all_outs = np.zeros(len(xs), dtype=np.float32)
    for val_indices in index_chunks:
        mask = np.ones(len(xs), dtype=bool)
        mask[val_indices] = False
        model.fit(xs[mask], ys[mask])
        all_outs[~mask] = model.decision_function(xs[~mask])
    return all_outs


def list_dirs(dirs: List[str]) -> Iterator[str]:
    for sub_dir in dirs:
        for x in os.listdir(sub_dir):
            if not x.startswith("."):
                yield os.path.join(sub_dir, x)


if __name__ == "__main__":
    main()
