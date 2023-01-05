"""
Benchmark the data loader.
"""

import argparse

from car_data.dataset import looping_loader
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    args = parser.parse_args()

    dataset = looping_loader(
        index_path=args.index_path, image_dir=args.image_dir, batch_size=64
    )
    next(dataset)
    for _ in tqdm(dataset):
        pass


if __name__ == "__main__":
    main()
