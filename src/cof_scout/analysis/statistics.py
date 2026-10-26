from __future__ import annotations

import numpy as np


def tie_adjusted_superiority(left: list[float], right: list[float]) -> float:
    if not left or not right:
        raise ValueError("Both samples must be non-empty")
    a = np.asarray(left, dtype=float)[:, None]
    b = np.asarray(right, dtype=float)[None, :]
    return float(np.mean((a > b) + 0.5 * (a == b)))


def bootstrap_mean_ci(
    values: list[float], *, samples: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    if not values:
        raise ValueError("values must be non-empty")
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    means = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)
