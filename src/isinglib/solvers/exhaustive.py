from __future__ import annotations

import heapq
import itertools
import time
from collections.abc import Iterator

import numpy as np

from isinglib._evaluate import (
    effective_field,
    energy,
    spin_flip_effective_field_update,
    spin_flip_energy_update,
)
from isinglib.problem import Problem
from isinglib.solvers.base import Solution
from isinglib.states import constant_spins

__all__ = ("exhaustive",)


def _gray_flip_indices(n: int) -> Iterator[int]:
    """Yield the bit index that flips at each step of an n-bit Gray-code walk."""
    if n < 0:
        raise ValueError("n must be non-negative.")
    prev = 0
    for step in range(1, 1 << n):
        gray = step ^ (step >> 1)
        flip = prev ^ gray
        prev = gray
        yield (flip & -flip).bit_length() - 1


def exhaustive(problem: Problem, *, n_best: int = 1) -> Solution:
    """Exact ground state of all 2^n spin assignments.

    Starts seriously slowing down for problem sizes larger than (n > 24).

    Args:
        problem: The problem to solve.
        n_best: How many of the lowest-energy distinct states to keep.

    Returns:
        A `Solution` with:
        - `problem_id`: fingerprint of `problem`.
        - `solver`: `"exhaustive"`.
        - `spins`: the exact ground state.
        - `energy`: its energy.
        - `elapsed_s`: wall-clock time the solve took, in seconds.
        - `degeneracy`: number of assignments that share the ground-state energy.
        - `top_spins`: best spins, sorted in ascending energy, only when `n_best > 1`.
        - `top_energies`: best energies, only when `n_best > 1`.
    """
    if n_best < 1:
        raise ValueError("n_best must be at least 1.")

    t0 = time.perf_counter()
    j, h, c, n = problem.j, problem.h, problem.c, problem.n

    s = constant_spins(problem.n, -1.0, dtype=problem.dtype)
    h_eff = effective_field(j, h, s)
    curr = energy(j, h, c, s, h_eff=h_eff)

    best_energy = curr
    best_spins = s.copy()
    degeneracy = 1

    keep_top = n_best > 1
    counter = itertools.count()
    heap: list[tuple[float, int, np.ndarray]] = []
    if keep_top:
        heap.append((-float(curr), next(counter), s.copy()))

    for k in _gray_flip_indices(n):
        curr += spin_flip_energy_update(s, h_eff, i=k)
        h_eff += spin_flip_effective_field_update(j, s, k)
        s[k] = -s[k]

        if curr < best_energy:
            best_energy = curr
            best_spins = s.copy()
            degeneracy = 1
        elif curr == best_energy:
            degeneracy += 1

        if keep_top and (len(heap) < n_best or curr < -heap[0][0]):
            candidate = (-float(curr), next(counter), s.copy())
            if len(heap) < n_best:
                heapq.heappush(heap, candidate)
            else:
                heapq.heappushpop(heap, candidate)

    kwargs = {}
    if keep_top:
        ordered = sorted(heap, key=lambda item: -item[0])
        kwargs["top_spins"] = np.array([item[2] for item in ordered])
        kwargs["top_energies"] = np.array([-item[0] for item in ordered], dtype=problem.dtype)

    return Solution(
        problem_id=problem.fingerprint,
        solver="exhaustive",
        spins=best_spins,
        energy=float(best_energy),
        elapsed_s=time.perf_counter() - t0,
        degeneracy=degeneracy,
        **kwargs,
    )
