from __future__ import annotations

import numpy as np
import numpy.typing as npt

from isinglib.dtypes import FloatArray, ensure_float_dtype

__all__ = ("constant_spins", "random_spins")


def random_spins(
    n: int,
    *,
    p: float = 0.5,
    rng: np.random.Generator | int | None = None,
    dtype: npt.DTypeLike | None = None,
) -> FloatArray:
    """Draw a random spin configuration in {-1, +1}.

    Args:
        n: Number of spins.
        p: Probability of a spin being +1.
        rng: Random generator or seed for reproducibility.
        dtype: Target float dtype (default float64).
    """
    _rng = np.random.default_rng(rng)
    return np.where(_rng.random(n) < p, 1.0, -1.0).astype(ensure_float_dtype(dtype))


def constant_spins(
    n: int,
    value: float = 1.0,
    *,
    dtype: npt.DTypeLike | None = None,
) -> FloatArray:
    """Return a uniform spin configuration.

    Args:
        n: Number of spins.
        value: Spin value to fill every entry with; must be -1.0 or 1.0.
        dtype: Target float dtype (default float64).
    """
    if value not in (-1.0, 1.0):
        raise ValueError("value must be -1.0 or 1.0.")
    return np.full(n, value, dtype=ensure_float_dtype(dtype))
