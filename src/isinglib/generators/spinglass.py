from __future__ import annotations

import numpy as np

from isinglib.generators.distributions import gaussian, pm_one
from isinglib.problem import Problem
from isinglib.topology.regular import complete, grid

__all__ = ("edwards_anderson", "sherrington_kirkpatrick")


def sherrington_kirkpatrick(
    n: int,
    *,
    rng: np.random.Generator | int | None = None,
) -> Problem:
    """Sherrington-Kirkpatrick model: complete graph, couplings `N(0, 1/n)`.

    Args:
        n: Number of spins.
        rng: Random generator or seed.
    """
    return gaussian(complete(n), 1.0 / np.sqrt(n), rng=rng)


def edwards_anderson(
    L: int,
    d: int = 3,
    *,
    gaussian_couplings: bool = False,
    rng: np.random.Generator | int | None = None,
) -> Problem:
    """Edwards-Anderson model: the periodic hypercubic lattice of `L^d` spins.

    Args:
        L: Spins along each axis (at least 3).
        d: Number of dimensions.
        gaussian_couplings: If True, draw couplings from `N(0, 1)` instead of ±1.
        rng: Random generator or seed.
    """
    lattice = grid(*(L,) * d, periodic=True)
    return gaussian(lattice, rng=rng) if gaussian_couplings else pm_one(lattice, rng=rng)
