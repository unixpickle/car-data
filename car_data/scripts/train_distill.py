"""
Entrypoint for distillation of one model into another.
Similar to train.py, but pass --teacher_model_name and --teacher_model_path.
"""

import argparse

import torch
from car_data.model import create_model
from car_data.train_loop import DistillationTrainLoop, TrainLoop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="clip")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=0)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--teacher_model_name", type=str, required=True)
    parser.add_argument("--teacher_model_path", type=str, required=True)
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("creating model...")
    model = create_model(args.model_name, device)

    print("creating teacher model...")
    teacher = create_model(args.teacher_model_name, device)
    teacher.load_state_dict(torch.load(args.teacher_model_path, map_location=device))

    print("creating trainer...")
    trainer = DistillationTrainLoop(
        teacher=teacher,
        index_path=args.index_path,
        image_dir=args.image_dir,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        lr=args.lr,
        weight_decay=args.weight_decay,
        model=model,
        device=device,
    )
    while True:
        trainer.run_step()


if __name__ == "__main__":
    main()
