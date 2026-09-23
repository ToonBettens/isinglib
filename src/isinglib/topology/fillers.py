from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from isinglib.dtypes import (
    FloatArray,
    ScalarLike,
    ensure_float_array,
    ensure_float_dtype,
    is_scalar_like,
)

__all__ = (
    "FillerLike",
    "choice",
    "constant",
    "gaussian",
    "pm_one",
    "resolve",
    "uniform",
)


# A recipe for weights rather than the weights themselves: it only learns how many are
# needed at the point it is used. A scalar gives every element that same weight. A
# callable is handed the count and returns that many weights.
type FillerLike = ScalarLike | Callable[[int], npt.ArrayLike]


def resolve(filler: FillerLike, n: int, *, dtype: npt.DTypeLike | None = None) -> FloatArray:
    """Normalise a weight specification into an `(n,)` float array.

    Args:
        filler: A weight specification.
        n: Number of weights to produce.
        dtype: Target float dtype (default float64).

    Raises:
        ValueError: If a callable returns the wrong shape.
        TypeError: If `filler` is neither scalar-like nor callable.
    """
    resolved = ensure_float_dtype(dtype)
    if is_scalar_like(filler):
        return np.full(n, float(filler), dtype=resolved)
    if callable(filler):
        values = ensure_float_array(filler(n), resolved)
        if values.shape != (n,):
            raise ValueError(f"filler returned shape {values.shape}; expected ({n},).")
        return values
    raise TypeError("filler must be a scalar or a callable(n) -> array.")


# ------------------------------------------------------------------
# Factories
# ------------------------------------------------------------------


def constant(value: ScalarLike) -> Callable[[int], FloatArray]:
    """Every weight equal to `value`."""
    return lambda n: np.full(n, float(value))


def uniform(
    low: float = 0.0,
    high: float = 1.0,
    *,
    rng: np.random.Generator | int | None = None,
) -> Callable[[int], FloatArray]:
    """Weights drawn uniformly from `[low, high)`.

    Args:
        low: Lower bound (inclusive).
        high: Upper bound (exclusive).
        rng: Random generator or seed.
    """
    if high < low:
        raise ValueError("high must be greater than or equal to low.")
    generator = np.random.default_rng(rng)
    return lambda n: generator.uniform(low, high, n)


def gaussian(
    sigma: float = 1.0,
    *,
    mean: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> Callable[[int], FloatArray]:
    """Weights drawn from a normal distribution.

    Args:
        sigma: Standard deviation; must be non-negative.
        mean: Distribution mean.
        rng: Random generator or seed.
    """
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative.")
    generator = np.random.default_rng(rng)
    return lambda n: generator.normal(mean, sigma, n)


def pm_one(
    p: float = 0.5,
    *,
    rng: np.random.Generator | int | None = None,
) -> Callable[[int], FloatArray]:
    """Weights of either -1 or +1.

    Args:
        p: Probability of drawing +1.
        rng: Random generator or seed.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1].")
    generator = np.random.default_rng(rng)
    return lambda n: np.where(generator.random(n) < p, 1.0, -1.0)


def choice(
    values: npt.ArrayLike,
    *,
    p: npt.ArrayLike | None = None,
    rng: np.random.Generator | int | None = None,
) -> Callable[[int], FloatArray]:
    """Weights drawn from a discrete set, e.g. `choice([-1, 0, 1])`.

    Args:
        values: The values to draw from.
        p: Probability of each value; uniform if not given.
        rng: Random generator or seed.
    """
    pool = ensure_float_array(values)
    if pool.ndim != 1 or pool.size == 0:
        raise ValueError("values must be a non-empty 1-D sequence.")
    probabilities = None if p is None else ensure_float_array(p)
    generator = np.random.default_rng(rng)
    return lambda n: generator.choice(pool, size=n, p=probabilities)
