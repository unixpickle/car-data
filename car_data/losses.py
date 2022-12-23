from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .constants import MAKE_MODEL_TO_INDEX, MEDIAN_PRICE_SCALE, PRICE_CUTOFFS, YEARS
from .dataset import CarImage


@dataclass
class PriceTargets:
    prices: torch.Tensor
    price_bins: torch.Tensor
    years: torch.Tensor
    make_models: torch.Tensor

    @classmethod
    def from_batch(cls, batch: List[CarImage], device: torch.device) -> "PriceTargets":
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
    def from_model_out(cls, outputs: Dict[str, torch.Tensor]) -> "PriceTargets":
        return cls(
            prices=outputs["price_median"],
            price_bins=F.softmax(outputs["price_bin"], dim=-1),
            years=F.softmax(outputs["year"], dim=-1),
            make_models=F.softmax(outputs["make_model"], dim=-1),
        )

    def metrics(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
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
            metrics["price_ce"]
            + (metrics["price_mae"] / MEDIAN_PRICE_SCALE)
            + metrics["year_ce"]
            + metrics["make_model_ce"]
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
