"""
Filter a dataset index using a filter and pre-computed feature vectors.
"""

import argparse
import os

import numpy as np
import torch
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--classifier_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(args.classifier_path).to(device)

    print("listing feature filenames...")
    feature_filenames = [
        x
        for x in os.listdir(args.feature_dir)
        if not x.startswith(".") and x.endswith(".npz")
    ]

    print("computing kept IDs...")
    positive_ids = set()
    total_ids = 0
    for feature_filename in tqdm(feature_filenames):
        obj = np.load(os.path.join(args.feature_dir, feature_filename))
        features = torch.from_numpy(obj["features"]).float().to(device)
        ids = obj["filenames"].tolist()
        with torch.no_grad():
            preds = model(features).cpu().numpy().tolist()
        total_ids += len(ids)
        for pred, id in zip(preds, ids):
            if pred:
                positive_ids.add(id)

    print(f"filtering index; kept {len(positive_ids)}/{total_ids}...")

    obj = np.load(args.index_path)
    use_indices = np.array([x.tolist() in positive_ids for x in obj["phashes"]])
    np.savez(args.output_path, **{k: obj[k][use_indices] for k in obj.keys()})


if __name__ == "__main__":
    main()
