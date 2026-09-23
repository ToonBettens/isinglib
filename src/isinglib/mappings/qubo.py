from __future__ import annotations

import numpy as np
import numpy.typing as npt

from isinglib.dtypes import FloatArray, ensure_float_array, ensure_float_dtype
from isinglib.problem import Problem

__all__ = ("decode", "encode", "from_ising")


def encode(
    q: npt.ArrayLike,
    offset: float = 0.0,
    *,
    dtype: npt.DTypeLike | None = None,
) -> Problem:
    """Encode QUBO form `E(x) = xᵀQx + offset`, with `x ∈ {0, 1}ⁿ`, as an Ising problem.

    Any representation of `q` is accepted: upper triangular, lower triangular,
    or symmetric, since `xᵀQx` reads `q` only through `q + qᵀ`.

    Args:
        q: (n, n) QUBO matrix.
        offset: Additive constant in the QUBO objective.
        dtype: Target float dtype (default float64).
    """
    resolved = ensure_float_dtype(dtype)
    given = np.asarray(q)
    q_arr = ensure_float_array(np.triu(given) + np.tril(given, k=-1).T, resolved)
    n = q_arr.shape[0]

    rows, cols = np.triu_indices(n, k=1)

    j = np.zeros((n, n), dtype=resolved)
    j[rows, cols] = -q_arr[rows, cols] / 4
    j[cols, rows] = j[rows, cols]

    h = j.sum(axis=1) - np.diag(q_arr) / 2
    c = float(offset) - float(h.sum()) + float(j[rows, cols].sum())

    return Problem(j, h, c, resolved)


def from_ising(problem: Problem) -> tuple[FloatArray, float]:
    """Convert an Ising problem to QUBO form `E(x) = xᵀQx + offset`, with `x ∈ {0, 1}ⁿ`.

    Returns:
        q: Upper-triangular (n, n) QUBO matrix.
        offset: Additive constant such that `E_ising(s) = E_qubo(x) + offset`
            when `s_i = 2x_i - 1`.
    """
    n = problem.n
    h = problem.h
    rows, cols = np.triu_indices(n, k=1)

    q = np.zeros((n, n), dtype=problem.dtype)
    q[rows, cols] = -4 * problem.j[rows, cols]
    q[np.arange(n), np.arange(n)] = 2 * problem.j.sum(axis=1) - 2 * h

    offset = float(problem.c + h.sum() - problem.j[rows, cols].sum())
    return q, offset


def decode(spins: npt.ArrayLike) -> FloatArray:
    """Decode a spin vector back to QUBO variables `x ∈ {0, 1}ⁿ`."""
    return (ensure_float_array(spins) + 1.0) / 2.0
