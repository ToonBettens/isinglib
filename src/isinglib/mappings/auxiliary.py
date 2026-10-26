from __future__ import annotations

import numpy as np
import numpy.typing as npt

from isinglib.dtypes import FloatArray, ensure_float_array
from isinglib.problem import Problem

__all__ = ("decode", "encode")


def encode(problem: Problem) -> Problem:
    """Fold the fields `h` into couplings to one auxiliary spin, appended as spin `n`.

    The result has `n + 1` spins, `j[i, n] = h[i]`, no fields and the same `c`, so
    `E'(s, s_n) = -0.5 sᵀ j s - s_n hᵀ s + c = E(s_n s)`: every state has exactly the
    energy of the original state it decodes to, and the ground energies are equal.
    The spin is appended even when `h` is zero, so the size is always `n + 1`.

    Args:
        problem: The problem whose fields to fold in.
    """
    n = problem.n
    j = np.zeros((n + 1, n + 1), dtype=problem.dtype)
    j[:n, :n] = problem.j
    j[:n, n] = j[n, :n] = problem.h
    return Problem(j, 0.0, problem.c, problem.dtype)


def decode(spins: npt.ArrayLike) -> FloatArray:
    """Decode `n + 1` spins to the original `n`: `s_n * s[:n]`."""
    s = ensure_float_array(spins)
    if s.ndim != 1 or s.size < 1:
        raise ValueError(f"spins must be a non-empty 1-D array, got shape {s.shape}.")
    return s[-1] * s[:-1]
