from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np

from isinglib.core import evaluate
from isinglib.core.problem import Problem
from isinglib.core.solution import Solution
from isinglib.core.solver import Solver

__all__ = ("ExhaustiveSolver",)


def _gray_flip_indices(n: int) -> Iterator[int]:
    """Yield the bit index that flips at each step of an n-bit Gray-code walk.

    Walking in Gray-code order visits all 2^n spin assignments while changing
    exactly one spin per step, so each energy update is O(n) instead of O(n²).
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    prev = 0
    for step in range(1, 1 << n):
        gray = step ^ (step >> 1)
        flip = prev ^ gray
        prev = gray
        yield (flip & -flip).bit_length() - 1


class ExhaustiveSolver(Solver):
    """Exact ground state by Gray-code enumeration of all 2^n spin assignments.

    Starts seriously slowing down for problem sizes larger than (n > 24).
    """

    @property
    def name(self) -> str:
        return "exhaustive"

    def solve(self, problem: Problem) -> Solution:
        t0 = time.perf_counter()
        j, h, c, n = problem.j, problem.h, problem.c, problem.n
        assert h is not None  # always set by Problem.__post_init__

        s = np.full(n, -1.0, dtype=problem.dtype)
        h_eff = evaluate.effective_field(j, h, s)
        curr = evaluate.energy(j, h, c, s, h_eff=h_eff)

        best_energy = curr
        best_spins = s.copy()

        for k in _gray_flip_indices(n):
            curr += evaluate.spin_flip_energy_update(s, h_eff, i=k)
            h_eff = h_eff + evaluate.spin_flip_effective_field_update(j, s, k)
            s[k] = -s[k]
            if curr < best_energy:
                best_energy = curr
                best_spins = s.copy()

        return Solution(
            spins=best_spins,
            energy=float(best_energy),
            time_s=time.perf_counter() - t0,
            solver_name=self.name,
            problem_id=problem.id,
            meta={"method": "gray-code", "evaluated": 1 << n},
        )
