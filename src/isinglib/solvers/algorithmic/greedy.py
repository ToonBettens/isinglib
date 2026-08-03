from __future__ import annotations

import time

import numpy as np

from isinglib.core import evaluate
from isinglib.core.problem import Problem
from isinglib.core.solution import Solution
from isinglib.core.solver import Solver
from isinglib.solvers.utils.states import random_spins

__all__ = ("GreedySolver",)


class GreedySolver(Solver):
    """Greedy local search: repeatedly flip the spin that most reduces energy.

    Converges to a local minimum in O(n) per flip. Supports multiple random
    restarts to escape bad initial states; the best solution across all restarts
    is returned.

    Args:
        n_restarts: Number of random starting states to try.
        rng: Random generator or seed for reproducibility.
    """

    def __init__(
        self,
        n_restarts: int = 1,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        self.n_restarts = n_restarts
        self.rng = rng

    @property
    def name(self) -> str:
        return "greedy"

    def solve(self, problem: Problem) -> Solution:
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.rng)
        j, h, c, n = problem.j, problem.h, problem.c, problem.n
        assert h is not None  # always set by Problem.__post_init__

        best_energy = np.inf
        best_spins = np.empty(n, dtype=problem.dtype)
        total_flips = 0

        for _ in range(self.n_restarts):
            s = random_spins(problem, rng=rng)
            h_eff = evaluate.effective_field(j, h, s)

            while True:
                delta = evaluate.spin_flip_energy_update(s, h_eff)
                k = int(np.argmin(delta))
                if delta[k] >= 0.0:
                    break
                h_eff += evaluate.spin_flip_effective_field_update(j, s, k)
                s[k] = -s[k]
                total_flips += 1

            energy = evaluate.energy(j, h, c, s, h_eff=h_eff)
            if energy < best_energy:
                best_energy = energy
                best_spins = s.copy()

        return Solution(
            spins=best_spins,
            energy=best_energy,
            time_s=time.perf_counter() - t0,
            solver_name=self.name,
            problem_id=problem.id,
            meta={"n_restarts": self.n_restarts, "total_flips": total_flips},
        )
