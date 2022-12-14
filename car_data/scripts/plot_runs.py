"""
Plot runs via their log files.
"""

import argparse
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ema_rate", type=float, default=0.99)
    parser.add_argument("--max_step", type=int, default=None)
    parser.add_argument("--output_path", type=str, default="plot.png")
    parser.add_argument("names_and_paths", nargs="+", type=str)
    args = parser.parse_args()

    plt.figure()
    for name, path in zip(args.names_and_paths[::2], args.names_and_paths[1::2]):
        lines = read_log_lines(path, args.max_step)
        steps = np.array([x["step"] for x in lines])
        losses = np.array([x["loss"] for x in lines])
        plt.plot(
            steps,
            smooth(losses, args.ema_rate),
            label=name,
        )
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.savefig(args.output_path)


def read_log_lines(path: str, max_step: Optional[int]) -> List[Dict[str, float]]:
    # map step to log dict, to allow restarts to overwrite old steps
    lines = {}

    with open(path, "r") as f:
        for line in f:
            if "step=" not in line:
                continue
            parts = line.split()
            obj = {}
            for item in parts:
                if "=" not in item:
                    continue
                k, v = item.split("=")
                obj[k] = float(v)
            if "step" in obj and (max_step is None or obj["step"] < max_step):
                lines[obj["step"]] = obj

    return sorted(lines.values(), key=lambda x: x["step"])


def smooth(data: np.ndarray, ema_rate: float) -> np.ndarray:
    num = 0.0
    denom = 0.0
    results = []
    for x in data:
        num = ema_rate * num + (1 - ema_rate) * x
        denom = ema_rate * denom + (1 - ema_rate)
        results.append(num / denom)
    return np.array(results)


if __name__ == "__main__":
    main()
