"""
Entrypoint for training. Pass hyperparameters and dataset as flags.
"""

import argparse
import shlex
import sys

import torch
from car_data.model import create_model
from car_data.train_loop import TrainLoop, add_training_args, training_args_dict


def main():
    parser = argparse.ArgumentParser()
    add_training_args(parser)
    parser.add_argument("--model_name", type=str, default="clip")
    args = parser.parse_args()
    train_args = training_args_dict(args)

    print(f"COMMAND: {shlex.join(sys.argv)}")

    print("creating model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.model_name, device)

    print("creating trainer...")
    trainer = TrainLoop(
        **train_args,
        model=model,
        device=device,
    )
    while True:
        trainer.run_step()


if __name__ == "__main__":
    main()
