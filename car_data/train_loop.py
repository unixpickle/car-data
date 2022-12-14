import argparse
import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, Iterator, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .dataset import CarImage, looping_loader
from .losses import LossTargets, LossWeights


class TrainLoopBase(ABC):
    def __init__(
        self,
        *,
        index_path: str,
        image_dir: str,
        use_data_aug: bool,
        save_dir: str,
        batch_size: int,
        microbatch: int,
        eval_interval: int,
        save_interval: int,
        lr: float,
        weight_decay: float,
        model: nn.Module,
        device: torch.device,
        loss_weights: LossWeights,
    ):
        self.index_path = index_path
        self.image_dir = image_dir
        self.use_data_aug = use_data_aug
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.microbatch = microbatch
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.model = model
        self.device = device
        self.loss_weights = loss_weights

        os.makedirs(save_dir, exist_ok=True)

        self.dataset_state_path = os.path.join(save_dir, "dataset_state.json")
        if os.path.exists(self.dataset_state_path):
            print("loading dataset state:", self.dataset_state_path)
            with open(self.dataset_state_path, "rb") as f:
                self.dataset_state = json.load(f)
        else:
            self.dataset_state = dict(test=None, train=None)

        self.train_dataset = looping_loader(
            index_path,
            image_dir,
            batch_size=self.batch_size,
            train=True,
            use_data_aug=use_data_aug,
            last_seen_phash=self.dataset_state["train"],
        )
        self.test_dataset = looping_loader(
            index_path,
            image_dir,
            batch_size=self.batch_size,
            train=False,
            last_seen_phash=self.dataset_state["test"],
        )

        self.opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.opt_state_path = os.path.join(save_dir, "opt.pt")
        if os.path.exists(self.opt_state_path):
            print("loading optimizer:", self.opt_state_path)
            self.opt.load_state_dict(
                torch.load(self.opt_state_path, map_location=device)
            )

        self.model_state_path = os.path.join(save_dir, "model.pt")
        if os.path.exists(self.model_state_path):
            print("loading model:", self.model_state_path)
            model.load_state_dict(
                torch.load(self.model_state_path, map_location=device)
            )

        self.step = 0
        self.step_state_path = os.path.join(save_dir, "step.json")
        if os.path.exists(self.step_state_path):
            print("loading step:", self.step_state_path)
            with open(self.step_state_path, "rb") as f:
                self.step = json.load(f)

    def run_step(self):
        results = LossAverage()

        if self.step % self.eval_interval == 0:
            with torch.no_grad():
                batch = next(self.test_dataset)
                for microbatch in self._microbatches(batch):
                    results.add(
                        {
                            f"eval_{k}": v
                            for k, v in self.compute_losses(microbatch).items()
                        },
                        len(microbatch),
                    )
                self.dataset_state["test"] = batch[-1].phash

        batch = next(self.train_dataset)
        self.opt.zero_grad()
        for microbatch in self._microbatches(batch):
            losses = self.compute_losses(microbatch)
            batch_frac = len(microbatch) / len(batch)
            (batch_frac * losses["loss"]).backward()
            results.add(losses, len(microbatch))
        self.opt.step()
        self.dataset_state["train"] = batch[-1].phash

        print(results.format(self.step))

        self.step += 1

        if not self.step % self.save_interval:
            print(f"saving at step {self.step}...")
            self.save()

    def _microbatches(self, batch: List[CarImage]) -> Iterator[List[CarImage]]:
        if not self.microbatch:
            yield batch
        else:
            for i in range(0, len(batch), self.microbatch):
                yield batch[i : i + self.microbatch]

    @abstractmethod
    def compute_losses(self, batch: List[CarImage]) -> Dict[str, torch.Tensor]:
        """
        Compute a dict of loss scalars for the batch of images.
        """

    def save(self):
        torch.save(self.model.state_dict(), _tmp_path(self.model_state_path))
        torch.save(self.opt.state_dict(), _tmp_path(self.opt_state_path))
        with open(_tmp_path(self.step_state_path), "w") as f:
            json.dump(self.step, f)
        with open(_tmp_path(self.dataset_state_path), "w") as f:
            json.dump(self.dataset_state, f)
        _rename_from_tmp(self.model_state_path)
        _rename_from_tmp(self.opt_state_path)
        _rename_from_tmp(self.step_state_path)
        _rename_from_tmp(self.dataset_state_path)


class TrainLoop(TrainLoopBase):
    def compute_losses(self, batch: List[CarImage]) -> Dict[str, torch.Tensor]:
        images = torch.stack([x.image for x in batch], dim=0).to(self.device)
        targets = LossTargets.from_batch(batch, self.device)
        outputs = self.model(images)
        return targets.metrics(self.loss_weights, outputs)


class DistillationTrainLoop(TrainLoopBase):
    def __init__(self, *, teacher: nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher

    def compute_losses(self, batch: List[CarImage]) -> Dict[str, torch.Tensor]:
        images = torch.stack([x.image for x in batch], dim=0).to(self.device)
        targets = LossTargets.from_batch(batch, self.device)
        with torch.no_grad():
            teacher_out = self.teacher(images)
            teacher_targets = LossTargets.from_model_out(teacher_out)
        outputs = self.model(images)
        with torch.no_grad():
            metrics = targets.metrics(self.loss_weights, outputs)
        metrics.update(
            {
                f"teacher_{k}": v
                for k, v in teacher_targets.metrics(self.loss_weights, outputs).item()
            }
        )
        metrics["loss"] = metrics.pop("teacher_loss")
        return metrics


class LossAverage:
    def __init__(self):
        self.results = defaultdict(lambda: 0.0)
        self.counts = defaultdict(lambda: 0)

    def add(self, losses: Dict[str, torch.Tensor], count: int):
        for k, v in losses.items():
            self.results[k] += v.item() * count
            self.counts[k] += count

    def average(self) -> Dict[str, float]:
        return {k: v / self.counts[k] for k, v in self.results.items()}

    def format(self, step: int) -> str:
        key_strs = [f"step={step}"]
        avg = self.average()
        for k in sorted(avg.keys()):
            key_strs.append(f"{k}={avg[k]:.04f}")
        return " ".join(key_strs)


def _tmp_path(orig_path: str) -> str:
    return orig_path + ".tmp"


def _rename_from_tmp(path: str) -> str:
    os.rename(_tmp_path(path), path)


def add_training_args(parser: argparse.ArgumentParser):
    parser.add_argument("--loss_weights", type=str, default="default")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=0)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--use_data_aug", action="store_true")
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)


def training_args_dict(args: argparse.Namespace) -> Dict[str, Any]:
    res = {}
    for k in [
        "lr",
        "weight_decay",
        "batch_size",
        "microbatch",
        "eval_interval",
        "save_interval",
        "use_data_aug",
        "index_path",
        "image_dir",
        "save_dir",
    ]:
        res[k] = getattr(args, k)
    res["loss_weights"] = LossWeights.parse(args.loss_weights)
    return res
