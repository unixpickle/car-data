"""
Run the classifier on an image.
"""

import argparse
from typing import Iterator, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from car_data.constants import MAKES_MODELS, PRICE_BIN_LABELS, YEARS
from car_data.model import create_model
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="clip")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("image", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.model_name, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    image_tensor = transform(Image.open(args.image)).to(device)

    outputs = model(image_tensor[None])

    print("---- Price ----")
    print(f"median: {outputs['price_median'].item():.02}")
    price_probs = F.softmax(outputs["price_bin"], dim=-1)[0].tolist()
    for label, prob in zip(pad_labels(PRICE_BIN_LABELS), price_probs):
        print(f"{label}: {(prob*100):.04}%")

    print("---- Make/model ----")
    make_model_probs = F.softmax(outputs["make_model"], dim=-1)[0].tolist()
    print_top_n(
        [f"{make} {model}" for make, model in MAKES_MODELS] + ["Unknown"],
        make_model_probs,
    )

    print("---- year ----")
    year_probs = F.softmax(outputs["year"], dim=-1)[0].tolist()
    print_top_n([str(year) for year in YEARS] + ["Unknown"], year_probs)


def print_top_n(labels: List[str], probs: List[float], n: int = 5):
    indices = np.argsort(-np.array(probs))
    labels = [labels[i] for i in indices[:n]]
    probs = [probs[i] for i in indices[:n]]
    for label, prob in zip(pad_labels(labels), probs):
        print(f"{label}: {(prob*100):.04}%")


def pad_labels(labels: List[str]) -> Iterator[str]:
    max_len = max(len(x) for x in labels)
    for label in labels:
        while len(label) < max_len:
            label = " " + label
        yield label


if __name__ == "__main__":
    main()
