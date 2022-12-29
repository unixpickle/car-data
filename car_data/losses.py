from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .constants import MAKE_MODEL_TO_INDEX, MEDIAN_PRICE_SCALE, PRICE_CUTOFFS, YEARS
from .dataset import CarImage


@dataclass
class LossWeights:
    price_ce: float = 1.0
    price_mae: float = 1.0
    year_ce: float = 1.0
    make_model_ce: float = 1.0

    @classmethod
    def parse(cls, x: str) -> "LossWeights":
        presets = {
            "default": LossWeights(),
            "price_ce_only": LossWeights(
                price_ce=1.0, price_mae=0.0, year_ce=0.0, make_model_ce=0.0
            ),
        }
        if x in presets:
            return presets[x]

        res = {}
        for part in x.split(","):
            pair = part.split("=")
            if len(pair) != 2:
                raise ValueError(f"expected pairs of k=v, but got token `{pair}`")
            res[pair[0]] = float(pair[1])
        return cls(**res)


@dataclass
class LossTargets:
    prices: torch.Tensor
    price_bins: torch.Tensor
    years: torch.Tensor
    make_models: torch.Tensor

    @classmethod
    def cat(cls, items: Sequence["LossTargets"]) -> "LossTargets":
        return LossTargets(
            prices=torch.cat([x.prices for x in items]),
            price_bins=torch.cat([x.price_bins for x in items]),
            years=torch.cat([x.years for x in items]),
            make_models=torch.cat([x.make_models for x in items]),
        )

    @classmethod
    def from_batch(cls, batch: List[CarImage], device: torch.device) -> "LossTargets":
        return cls(
            prices=torch.tensor(
                [x.price for x in batch], dtype=torch.float32, device=device
            ),
            price_bins=torch.tensor([bin_price(x.price) for x in batch], device=device),
            years=torch.tensor([bin_year(x.year) for x in batch], device=device),
            make_models=torch.tensor(
                [bin_make_model(x.make, x.model) for x in batch], device=device
            ),
        )

    @classmethod
    def from_model_out(cls, outputs: Dict[str, torch.Tensor]) -> "LossTargets":
        return cls(
            prices=outputs["price_median"],
            price_bins=F.softmax(outputs["price_bin"], dim=-1),
            years=F.softmax(outputs["year"], dim=-1),
            make_models=F.softmax(outputs["make_model"], dim=-1),
        )

    def metrics(
        self, weights: LossWeights, outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        metrics = dict(
            price_ce=F.cross_entropy(outputs["price_bin"], self.price_bins),
            price_acc=(
                (outputs["price_bin"].argmax(-1) == self.price_bins).float().mean()
            ),
            price_mae=(outputs["price_median"] - self.prices).abs().float().mean(),
            year_ce=F.cross_entropy(outputs["year"], self.years),
            make_model_ce=F.cross_entropy(outputs["make_model"], self.make_models),
        )
        metrics["loss"] = (
            (weights.price_ce * metrics["price_ce"])
            + (weights.price_mae * metrics["price_mae"] / MEDIAN_PRICE_SCALE)
            + (weights.year_ce * metrics["year_ce"])
            + (weights.make_model_ce * metrics["make_model_ce"])
        )
        return metrics


def bin_price(price: float) -> int:
    for i, cutoff in enumerate(PRICE_CUTOFFS):
        if price <= cutoff:
            return i
    return len(PRICE_CUTOFFS)


def bin_prices(prices: np.ndarray) -> np.ndarray:
    return np.searchsorted(PRICE_CUTOFFS, prices)


def bin_make_model(make: str, model: str) -> int:
    return MAKE_MODEL_TO_INDEX.get((make, model), len(MAKE_MODEL_TO_INDEX))


def bin_year(year: int) -> int:
    if year not in YEARS:
        return len(YEARS)
    return YEARS.index(year)
