"""
Entrypoint for distillation of one model into another.
Similar to train.py, but pass --teacher_model_name and --teacher_model_path.
"""

import argparse
import shlex
import sys

import torch
from car_data.model import create_model
from car_data.train_loop import (
    DistillationTrainLoop,
    add_training_args,
    training_args_dict,
)


def main():
    parser = argparse.ArgumentParser()
    add_training_args(parser)
    parser.add_argument("--model_name", type=str, default="clip")
    parser.add_argument("--teacher_model_name", type=str, required=True)
    parser.add_argument("--teacher_model_path", type=str, required=True)
    args = parser.parse_args()
    train_args = training_args_dict(args)

    print(f"COMMAND: {shlex.join(sys.argv)}")

    print("creating model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.model_name, device)

    print("creating teacher model...")
    teacher = create_model(args.teacher_model_name, device)
    teacher.load_state_dict(torch.load(args.teacher_model_path, map_location=device))

    print("creating trainer...")
    trainer = DistillationTrainLoop(
        **train_args,
        teacher=teacher,
        model=model,
        device=device,
    )
    while True:
        trainer.run_step()


if __name__ == "__main__":
    main()
