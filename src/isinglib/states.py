from __future__ import annotations

import numpy as np
import numpy.typing as npt

from isinglib.dtypes import FloatArray, ensure_float_array, ensure_float_dtype

__all__ = ("constant_spins", "ensure_spins", "random_spins")


def ensure_spins(
    spins: npt.ArrayLike,
    n: int,
    *,
    dtype: npt.DTypeLike | None = None,
    copy: bool = False,
) -> FloatArray:
    """Coerce and validate a spin configuration: shape `(n,)`, valued in {-1, +1}.

    Args:
        spins: Candidate spin configuration.
        n: Required length.
        dtype: Target float dtype (default float64).
        copy: If True, guarantee the output is memory-independent from `spins`.

    Raises:
        ValueError: If the shape is wrong, or a value isn't -1.0 or 1.0.
    """
    arr = ensure_float_array(spins, dtype=dtype, copy=copy)
    if arr.shape != (n,):
        raise ValueError(f"spins must have shape ({n},), got {arr.shape}.")
    if not np.all(np.isin(arr, (-1.0, 1.0))):
        raise ValueError("spins must contain only -1.0 and 1.0.")
    return arr


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
