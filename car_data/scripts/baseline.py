"""
Compute the baseline accuracy of a dummy classifier.
"""

import argparse
from collections import Counter

import numpy as np
from car_data.losses import bin_prices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index_path", type=str)
    args = parser.parse_args()

    prices = np.load(args.index_path)["prices"]
    max_count = max(Counter(bin_prices(prices)).values())
    print(max_count / len(prices))


if __name__ == "__main__":
    main()
