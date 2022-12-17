import json
import os
from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .dataset import CarImage, looping_loader
from .losses import bin_price


class TrainLoop:
    def __init__(
        self,
        index_path: str,
        image_dir: str,
        save_dir: str,
        batch_size: int,
        microbatch: int,
        eval_interval: int,
        save_interval: int,
        lr: float,
        weight_decay: float,
        model: nn.Module,
        device: torch.device,
        model_init_fn: Callable[[nn.Module], Any] = (lambda _: None),
    ):
        self.index_path = index_path
        self.image_dir = image_dir
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.microbatch = microbatch
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.model = model
        self.device = device

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
        else:
            model_init_fn(model)

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
                            for k, v in self._compute_losses(microbatch).items()
                        },
                        len(microbatch),
                    )
                self.dataset_state["test"] = batch[-1].phash

        batch = next(self.train_dataset)
        self.opt.zero_grad()
        for microbatch in self._microbatches(batch):
            losses = self._compute_losses(microbatch)
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

    def _compute_losses(self, batch: List[CarImage]) -> Dict[str, torch.Tensor]:
        images = torch.stack([x.image for x in batch], dim=0).to(self.device)
        price_targets = torch.tensor(
            [bin_price(x.price) for x in batch], device=self.device
        )
        outputs = self.model(images)
        price_ce = F.cross_entropy(outputs["price_bin"], price_targets)
        acc = torch.mean((outputs["price_bin"].argmax(-1) == price_targets).float())
        return dict(
            price_ce=price_ce,
            price_acc=acc,
            loss=price_ce,
        )

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
            key_strs.append(f"{k}={avg[k]:.04}")
        return " ".join(key_strs)


def _tmp_path(orig_path: str) -> str:
    return orig_path + ".tmp"


def _rename_from_tmp(path: str) -> str:
    os.rename(_tmp_path(path), path)
