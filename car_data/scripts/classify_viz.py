"""
Create an SVG with a visual depiction of a model's predictions for a batch of
input images.
"""

import argparse

import torch
from PIL import Image

from car_data.dataset import image_transform
from car_data.graphics import (
    HStack,
    open_context,
    prediction_element,
    prediction_element_size,
)
from car_data.model import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="viz.svg")
    parser.add_argument("--model_name", type=str, default="clip")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("images", type=str, nargs="+")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.model_name, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    transform = image_transform(False)

    with open_context(args.output, *prediction_element_size()) as ctx:
        panels = []
        for i, img_path in enumerate(args.images):
            img = Image.open(img_path).convert("RGB")
            outputs = model(transform(img)[None].to(device))
            panels.append(prediction_element(ctx, i, img, outputs))
        HStack(*panels).draw_at(ctx, 0, 0)


if __name__ == "__main__":
    main()
