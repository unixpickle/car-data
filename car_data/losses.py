import numpy as np

PRICE_CUTOFFS = [
    10_000.0,
    15_000.0,
    20_000.0,
    25_000.0,
    30_000.0,
    35_000.0,
    40_000.0,
    50_000.0,
    60_000.0,
]
NUM_PRICE_BINS = len(PRICE_CUTOFFS) + 1
PRICE_BIN_LABELS = [
    "$0-$10,000",
    "$10,000-$15,000",
    "$15,000-$20,000",
    "$20,000-$25,000",
    "$25,000-$30,000",
    "$30,000-$35,000",
    "$35,000-$40,000",
    "$40,000-$50,000",
    "$50,000-$60,000",
    "$60,000+",
]


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
