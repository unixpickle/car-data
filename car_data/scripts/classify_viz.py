"""
Create an SVG with a visual depiction of a model's predictions for a batch of
input images.
"""

import argparse
import io
from typing import List

import cairo
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from car_data.constants import MAKES_MODELS, PRICE_BIN_LABELS, YEARS
from car_data.dataset import image_transform
from car_data.model import create_model

PANEL_WIDTH = 512
IMAGE_PADDING = 16
IMAGE_SIZE = 224
SEPARATOR_HEIGHT = 16
TEXT_HEIGHT = 48
PROB_BIN_HEIGHT = 64
NUM_PRICE_BINS = 4
NUM_MAKE_MODEL_BINS = 5
NUM_YEAR_BINS = 4
BOTTOM_PADDING = 16

PANEL_HEIGHT = (
    PANEL_WIDTH
    + SEPARATOR_HEIGHT * 4
    + TEXT_HEIGHT
    + PROB_BIN_HEIGHT * (max(NUM_PRICE_BINS, NUM_YEAR_BINS) + NUM_MAKE_MODEL_BINS)
    + BOTTOM_PADDING
)


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

    x_offset = 0
    y_offset = 0

    with cairo.SVGSurface(
        args.output, len(args.images) * PANEL_WIDTH, PANEL_HEIGHT
    ) as surface:
        ctx = cairo.Context(surface)

        def add_image(img: Image.Image):
            nonlocal x_offset
            nonlocal y_offset

            width, height = img.size  # Get dimensions
            size = min(width, height)
            left = (width - size) // 2
            top = (height - size) // 2
            img = img.crop((left, top, left + size, top + size))
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            data = io.BytesIO()
            img.save(data, format="PNG")
            data.seek(0)
            source = cairo.ImageSurface.create_from_png(data)
            ctx.set_source_surface(
                source,
                x_offset + (PANEL_WIDTH - IMAGE_SIZE) // 2,
                y_offset + IMAGE_PADDING,
            )
            ctx.paint()

            y_offset += IMAGE_SIZE + IMAGE_PADDING * 2

        def add_separator():
            nonlocal x_offset
            nonlocal y_offset
            ctx.set_source_rgb(0.8, 0.8, 0.8)
            ctx.set_line_width(1.0)
            ctx.move_to(x_offset + 5, y_offset + SEPARATOR_HEIGHT / 2)
            ctx.line_to(x_offset + PANEL_WIDTH - 5, y_offset + SEPARATOR_HEIGHT / 2)
            ctx.stroke()
            y_offset += SEPARATOR_HEIGHT

        def add_text_label(text: str):
            nonlocal x_offset
            nonlocal y_offset
            ctx.set_source_rgb(0, 0, 0)
            ctx.set_font_size(18)
            ctx.select_font_face(
                "Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL
            )
            ctx.move_to(x_offset + 20, y_offset + 31)
            ctx.show_text(text)
            ctx.stroke()
            y_offset += TEXT_HEIGHT

        def add_probability_bin(name: str, prob: float, width: float = PANEL_WIDTH / 2):
            nonlocal x_offset
            nonlocal y_offset

            ctx.set_source_rgb(0, 0, 0)
            ctx.set_font_size(18)
            ctx.select_font_face(
                "Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL
            )
            ctx.move_to(x_offset + 20, y_offset + 28)
            ctx.show_text(name)
            ctx.stroke()

            ctx.move_to(x_offset + width - 70, y_offset + 52)
            ctx.show_text(f"{(prob*100):2.1f}%")
            ctx.stroke()

            ctx.set_source_rgb(0.9, 0.9, 0.9)
            ctx.rectangle(x_offset + 20, y_offset + 36, (width - 100), 20)
            ctx.fill()

            ctx.set_source_rgb(0x65 / 0xFF, 0xBC / 0xFF, 0xD4 / 0xFF)
            ctx.rectangle(x_offset + 20, y_offset + 36, (width - 100) * prob, 20)
            ctx.fill()

            y_offset += PROB_BIN_HEIGHT

        def add_top_n(labels: List[str], probs: List[float], n: int):
            for i in np.argsort(probs)[::-1][:n]:
                add_probability_bin(labels[i], probs[i])

        for img_path in args.images:
            img = Image.open(img_path).convert("RGB")
            add_image(img)
            add_separator()

            outputs = model(transform(img)[None].to(device))

            add_text_label(f"Median price: ${round(outputs['price_median'].item())}")
            add_separator()

            price_probs = F.softmax(outputs["price_bin"], dim=-1)[0].tolist()
            year_probs = F.softmax(outputs["year"], dim=-1)[0].tolist()
            old_y_offset = y_offset
            add_top_n(PRICE_BIN_LABELS, price_probs, NUM_PRICE_BINS)
            y_offset = old_y_offset
            x_offset += PANEL_WIDTH / 2
            add_top_n(
                [str(year) for year in YEARS] + ["Unknown"],
                year_probs,
                NUM_YEAR_BINS,
            )
            y_offset = max(y_offset, old_y_offset)
            x_offset -= PANEL_WIDTH / 2
            add_separator()

            make_model_probs = F.softmax(outputs["make_model"], dim=-1)[0].tolist()
            x_offset += PANEL_WIDTH / 4
            add_top_n(
                [f"{make} {model}" for make, model in MAKES_MODELS] + ["Unknown"],
                make_model_probs,
                NUM_MAKE_MODEL_BINS,
            )
            x_offset -= PANEL_WIDTH / 4

            year_probs = F.softmax(outputs["year"], dim=-1)[0].tolist()

            x_offset += PANEL_WIDTH
            y_offset = 0


if __name__ == "__main__":
    main()
