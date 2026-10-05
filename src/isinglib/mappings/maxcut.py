from __future__ import annotations

import numpy as np
import numpy.typing as npt

from isinglib.dtypes import ensure_float_array, ensure_float_dtype
from isinglib.problem import Problem

__all__ = ("decode", "encode", "objective")


def encode(w: npt.ArrayLike, *, dtype: npt.DTypeLike | None = None) -> Problem:
    """Encode max-cut on the weighted graph `w` as an Ising problem with `E(s) = -cut(s)`.

    `cut(s) = Σ_{i<j} w_ij (1 - s_i s_j) / 2` sums the weights of the edges
    whose ends land on different sides. With `j = -w/2` and `c = -Σ_{i<j} w_ij / 2`
    the energy is exactly minus the cut, so the ground state is a maximum cut
    and a published cut value compares to `-problem.energy(spins)` as is.
    Weights may be negative.

    Args:
        w: (n, n) symmetric, zero-diagonal weighted adjacency matrix.
        dtype: Target float dtype (default float64).
    """
    resolved = ensure_float_dtype(dtype)
    weights = ensure_float_array(w, resolved)
    return Problem(-weights / 2, c=-float(weights.sum()) / 4, dtype=resolved)


def decode(spins: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Decode a spin vector to the cut's sides: True where `s_i = +1`."""
    return np.asarray(spins) > 0


def objective(w: npt.ArrayLike, side: npt.ArrayLike) -> float:
    """Total weight of the edges of `w` crossing the cut that `side` defines.

    Equals `-problem.energy(spins)` for `problem = encode(w)` and `side = decode(spins)`.

    Args:
        w: (n, n) symmetric weighted adjacency matrix.
        side: (n,) bool mask of one side of the cut.
    """
    weights = ensure_float_array(w)
    mask = np.asarray(side, dtype=bool)
    if mask.shape != (weights.shape[0],):
        raise ValueError(f"side must have shape ({weights.shape[0]},), got {mask.shape}.")
    return float(weights[np.ix_(mask, ~mask)].sum())
