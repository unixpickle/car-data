"""
Re-scale output heads of the model to minimize validation loss.
This can help recalibrate overconfident model predictions after a model has
overfit to the training data.
"""

import argparse
from typing import Iterator, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm.auto import tqdm

from car_data.dataset import looping_loader
from car_data.losses import LossTargets, LossWeights
from car_data.model import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--model_name", type=str, default="clip")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("checkpoint_out", type=str)
    args = parser.parse_args()

    print("creating model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.model_name, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    print("loading data...")
    dataset = looping_loader(
        index_path=args.index_path,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        train=False,
    )
    outputs = []
    targets = []
    for _ in tqdm(range(args.num_images // args.batch_size)):
        batch = next(dataset)
        with torch.no_grad():
            outputs.append(
                model(torch.stack([x.image for x in batch], dim=0).to(device))
            )
            targets.append(LossTargets.from_batch(batch, device))

    outputs = {k: torch.cat([x[k] for x in outputs]) for k in outputs[0].keys()}
    targets = LossTargets.cat(targets)

    print("recalibrating...")

    def loss_fn(scales: torch.Tensor) -> torch.Tensor:
        scaled_outputs = {k: v * scales[i] for i, (k, v) in enumerate(outputs.items())}
        return targets.metrics(LossWeights(), scaled_outputs)["loss"]

    scales = nn.Parameter(torch.ones(len(outputs.keys()), device=device))
    loss_fn = torch.jit.trace(loss_fn, scales)

    init_loss = loss_fn(scales).item()

    opt = torch.optim.Adam([scales], lr=1e-2)
    for i in tqdm(range(args.iterations)):
        loss = loss_fn(scales)
        opt.zero_grad()
        loss.backward()
        opt.step()

    scale_dict = dict(zip(outputs.keys(), scales.detach().cpu().tolist()))

    print("scales:")
    print()
    for name, scale in scale_dict.items():
        print(f"  {name}: {scale}")
    print()
    print(f"loss went from {init_loss} => {loss.item()}")

    model.output_layer().scale_outputs(scale_dict)
    torch.save(model.state_dict(), args.checkpoint_out)


if __name__ == "__main__":
    main()
