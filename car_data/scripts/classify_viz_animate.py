"""
Create an animation of classification results as a crop is moved from one side
of a rectangular image to the other.
"""

import argparse
import io
from typing import Iterator

import cairo
import torch
from PIL import Image

from car_data.dataset import image_transform
from car_data.graphics import HStack, prediction_element, prediction_element_size
from car_data.model import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="viz.gif")
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--frame_rate", type=float, default=10.0)
    parser.add_argument("--model_name", type=str, default="clip")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("images", type=str, nargs="+")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.model_name, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    transform = image_transform(False)

    imgs = [Image.open(path).convert("RGB") for path in args.images]
    crop_iter = zip(*(crops_of_image(img, args.num_frames) for img in imgs))

    images = []
    width, height = prediction_element_size()
    for crops in crop_iter:
        with cairo.ImageSurface(
            cairo.Format.RGB24, width * len(imgs), height
        ) as surface:
            ctx = cairo.Context(surface)
            panels = []
            for i, crop in enumerate(crops):
                outputs = model(transform(crop)[None].to(device))
                panels.append(prediction_element(ctx, i, crop, outputs))
            HStack(*panels).draw_at(ctx, 0, 0)

            # Convert canvas to Pillow in the hackiest possible way.
            f = io.BytesIO()
            surface.write_to_png(f)
            f.seek(0)
            images.append(Image.open(f))

    images[0].save(
        args.output,
        save_all=True,
        append_images=images[1:],
        duration=round(1000 / args.frame_rate),
        loop=0,
    )


def crops_of_image(img: Image.Image, n: int) -> Iterator[Image.Image]:
    width, height = img.size
    min_size = min(width, height)

    dx = (width - min_size) / (n - 1)
    dy = (height - min_size) / (n - 1)

    for i in range(n):
        x = round(i * dx)
        y = round(i * dy)
        yield img.crop((x, y, x + min_size, y + min_size))


if __name__ == "__main__":
    main()
