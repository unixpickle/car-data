"""
Create an SVG with a visual depiction of a model's predictions for a batch of
input images.
"""

import argparse
import io
from abc import ABC, abstractmethod
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

    with cairo.SVGSurface(
        io.BytesIO(), len(args.images) * PANEL_WIDTH, 10000
    ) as surface:
        ctx = cairo.Context(surface)

        panels = []
        for i, img_path in enumerate(args.images):
            img = Image.open(img_path).convert("RGB")
            outputs = model(transform(img)[None].to(device))
            content = VStack(
                pad_to_width(ImageElement(crop_image(img)), PANEL_WIDTH),
                Padded(Separator(PANEL_WIDTH - 40.0), horiz=20, vert=16),
                pad_to_width(
                    HStack(
                        Text(ctx, "Price prediction:", font_size=24.0),
                        Empty(width=10, height=1),
                        Text(
                            ctx,
                            f"${int(round(outputs['price_median'].item()))}",
                            font_size=24.0,
                            bold=True,
                        ),
                    ),
                    PANEL_WIDTH,
                ),
                Padded(Separator(PANEL_WIDTH - 40.0), horiz=20, vert=16),
                Empty(width=PANEL_WIDTH, height=16),
                HStack(
                    TopN(
                        ctx,
                        PANEL_WIDTH / 2,
                        "Price",
                        PRICE_BIN_LABELS,
                        F.softmax(outputs["price_bin"], dim=-1)[0].tolist(),
                        4,
                    ),
                    TopN(
                        ctx,
                        PANEL_WIDTH / 2,
                        "Year",
                        [str(year) for year in YEARS] + ["Unknown"],
                        F.softmax(outputs["year"], dim=-1)[0].tolist(),
                        4,
                    ),
                ),
                Empty(width=PANEL_WIDTH, height=16),
                pad_to_width(
                    TopN(
                        ctx,
                        PANEL_WIDTH * 0.8,
                        "Make/Model",
                        [f"{make} {model}" for make, model in MAKES_MODELS]
                        + ["Unknown"],
                        F.softmax(outputs["make_model"], dim=-1)[0].tolist(),
                        5,
                    ),
                    PANEL_WIDTH,
                ),
                Empty(width=PANEL_WIDTH, height=16),
            )
            panels.append(Overlay(Background(i, PANEL_WIDTH, content.height), content))
        all_content = HStack(*panels)

        with cairo.SVGSurface(
            args.output, all_content.width, all_content.height
        ) as surface:
            ctx = cairo.Context(surface)
            all_content.draw_at(ctx, 0, 0)


def crop_image(img: Image.Image) -> Image.Image:
    width, height = img.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    img = img.crop((left, top, left + size, top + size))
    return img.resize((IMAGE_SIZE, IMAGE_SIZE))


class Element(ABC):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    @abstractmethod
    def draw_at(self, ctx: cairo.Context, x: float, y: float):
        """Draw the UI element at the coordinates."""
        _, _ = x, y


class Combination(Element):
    def __init__(self, *children: Element, horiz: bool = False, vert: bool = False):
        super().__init__(
            width=(sum if horiz else max)(x.width for x in children),
            height=(sum if vert else max)(x.height for x in children),
        )
        self.horiz = horiz
        self.vert = vert
        self.children = children

    def draw_at(self, ctx: cairo.Context, x: float, y: float):
        for child in self.children:
            child.draw_at(ctx, x, y)
            if self.horiz:
                x += child.width
            if self.vert:
                y += child.height


class VStack(Combination):
    def __init__(self, *children: Element):
        super().__init__(
            *children,
            vert=True,
        )


class HStack(Combination):
    def __init__(self, *children: Element):
        super().__init__(
            *children,
            horiz=True,
        )


class Overlay(Combination):
    def __init__(self, *children: Element):
        super().__init__(
            *children,
        )


class Padded(Element):
    def __init__(self, contained: Element, horiz: float = 0.0, vert: float = 0.0):
        super().__init__(
            width=contained.width + horiz * 2,
            height=contained.height + vert * 2,
        )
        self.contained = contained
        self.horiz = horiz
        self.vert = vert

    def draw_at(self, ctx: cairo.Context, x: float, y: float):
        self.contained.draw_at(ctx, x + self.horiz, y + self.vert)


class Empty(Element):
    def __init__(self, width: float, height: float):
        super().__init__(width, height)

    def draw_at(self, ctx: cairo.Context, x: float, y: float):
        _, _, _ = ctx, x, y


class Separator(Element):
    def __init__(self, width: float):
        super().__init__(width, 1)

    def draw_at(self, ctx: cairo.Context, x: float, y: float):
        ctx.set_source_rgb(0.8, 0.8, 0.8)
        ctx.set_line_width(1.0)
        ctx.move_to(x, y + 0.5)
        ctx.line_to(x + self.width, y + 0.5)
        ctx.stroke()


class Background(Element):
    def __init__(self, idx: int, width: float, height: float):
        super().__init__(width, height)
        self.idx = idx

    def draw_at(self, ctx: cairo.Context, x: float, y: float):
        if self.idx % 2:
            brightness = 0.97
            ctx.set_source_rgb(brightness, brightness, brightness)
            ctx.rectangle(x, y, self.width, self.height)
            ctx.fill()
        else:
            ctx.set_source_rgb(1, 1, 1)
            ctx.rectangle(x, y, self.width, self.height)
            ctx.fill()


class ImageElement(Element):
    def __init__(self, img: Image.Image):
        width, height = img.size
        super().__init__(width, height)

        data = io.BytesIO()
        img.save(data, format="PNG")
        data.seek(0)
        self.source = cairo.ImageSurface.create_from_png(data)

    def draw_at(self, ctx: cairo.Context, x: float, y: float):
        ctx.set_source_surface(
            self.source,
            x,
            y,
        )
        ctx.paint()


class Text(Element):
    def __init__(
        self, ctx: cairo.Context, text: str, font_size: float, bold: bool = False
    ):
        ctx.set_font_size(font_size)
        ctx.select_font_face(
            "Arial",
            cairo.FONT_SLANT_NORMAL,
            cairo.FONT_WEIGHT_NORMAL if not bold else cairo.FONT_WEIGHT_BOLD,
        )
        extents = ctx.text_extents(text)
        # Height should not depend on text to make consecutive labels
        # line up perfectly.
        height = ctx.text_extents(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        ).height
        self.text = text
        self.font_size = font_size
        self.bold = bold
        super().__init__(extents.width, height)

    def draw_at(self, ctx: cairo.Context, x: float, y: float):
        ctx.set_font_size(self.font_size)
        ctx.select_font_face(
            "Arial",
            cairo.FONT_SLANT_NORMAL,
            cairo.FONT_WEIGHT_NORMAL if not self.bold else cairo.FONT_WEIGHT_BOLD,
        )
        ctx.set_source_rgb(0, 0, 0)
        hacky_height_centerer = 0.9  # found empirically to center vertically
        ctx.move_to(x, y + self.height * hacky_height_centerer)
        ctx.show_text(self.text)
        ctx.stroke()


class ProbabilityBar(Element):
    def __init__(
        self, ctx: cairo.Context, width: float, title: str, probability: float
    ):
        title_element = Text(ctx, title, font_size=18.0)
        prob_text = Text(ctx, f"{(probability*100):2.1f}%", font_size=18.0)
        super().__init__(width, title_element.height + 30)
        self.probability = probability
        self.title_element = title_element
        self.prob_text = prob_text

    def draw_at(self, ctx: cairo.Context, x: float, y: float):
        self.title_element.draw_at(ctx, x, y)

        bar_y = y + self.title_element.height + 5
        bar_width = self.width - self.prob_text.width - 8

        self.prob_text.draw_at(ctx, x + self.width - self.prob_text.width, bar_y)

        ctx.set_source_rgb(0.9, 0.9, 0.9)
        ctx.rectangle(x, bar_y, bar_width, 20)
        ctx.fill()

        ctx.set_source_rgb(0x65 / 0xFF, 0xBC / 0xFF, 0xD4 / 0xFF)
        ctx.rectangle(x, bar_y, bar_width * self.probability, 20)
        ctx.fill()


class TopN(VStack):
    def __init__(
        self,
        ctx: cairo.Context,
        width: float,
        title: str,
        labels: List[str],
        probs: List[float],
        n: int,
    ):
        elements = []
        elements.append(pad_to_width(Text(ctx, title, 20, bold=True), width))
        elements.append(Padded(Separator(width - 32), horiz=16, vert=8))
        bars = []
        for i in np.argsort(probs)[::-1][:n]:
            bars.append(ProbabilityBar(ctx, width - 40, labels[i], probs[i]))
        max_prob_text_width = max(bar.prob_text.width for bar in bars)
        for bar in bars:
            bar.prob_text.width = max_prob_text_width
            elements.append(Padded(bar, horiz=20))
        super().__init__(*elements)


def pad_to_width(e: Element, width: float) -> Padded:
    return Padded(e, horiz=max(0, (width - e.width) / 2))


if __name__ == "__main__":
    main()