from __future__ import annotations

import numpy as np

from isinglib.core.problem import Problem

__all__ = ("constant_spins", "random_spins")


def random_spins(
    problem: Problem,
    *,
    p: float = 0.5,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Draw a random spin configuration sized and typed for `problem`.

    Args:
        problem: Problem to size and type the state for.
        p: Probability of a spin being +1.
        rng: Random generator or seed for reproducibility.
    """
    rng = np.random.default_rng(rng)
    return np.where(rng.random(problem.n) < p, 1.0, -1.0).astype(problem.dtype)


def constant_spins(problem: Problem, value: float = 1.0) -> np.ndarray:
    """Return a uniform spin configuration, sized and typed for `problem`.

    Args:
        problem: Problem to size and type the state for.
        value: Spin value to fill every entry with; must be -1.0 or 1.0.
    """
    if value not in (-1.0, 1.0):
        raise ValueError("value must be -1.0 or 1.0.")
    return np.full(problem.n, value, dtype=problem.dtype)
