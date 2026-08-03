from __future__ import annotations

import time

import numpy as np

from isinglib.core import evaluate
from isinglib.core.problem import Problem
from isinglib.core.solution import Solution
from isinglib.core.solver import Solver

__all__ = ("TabuSearchSolver",)


class TabuSearchSolver(Solver):
    """Tabu search: best-improvement local search with a recency-based tabu list.

    At each step the best non-tabu flip is chosen. A flipped spin is forbidden
    for ``tabu_tenure`` subsequent steps. The aspiration criterion overrides the
    tabu status when a flip would yield a new global best.

    Args:
        n_steps: Number of search iterations.
        tabu_tenure: Number of steps a spin remains forbidden after being flipped.
        rng: Random generator or seed for the initial state.
        record_trajectory: If True, record the spin configuration after every
            step into ``Solution.trajectory``. Off by default — see
            ``SimulatedAnnealingSolver`` for the same tradeoff.
    """

    def __init__(
        self,
        n_steps: int = 1_000,
        tabu_tenure: int = 10,
        rng: np.random.Generator | int | None = None,
        record_trajectory: bool = False,
    ) -> None:
        if tabu_tenure < 1:
            raise ValueError("tabu_tenure must be at least 1.")
        self.n_steps = n_steps
        self.tabu_tenure = tabu_tenure
        self.rng = rng
        self.record_trajectory = record_trajectory

    @property
    def name(self) -> str:
        return "tabu_search"

    def solve(self, problem: Problem) -> Solution:
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.rng)
        j, h, c, n = problem.j, problem.h, problem.c, problem.n
        assert h is not None  # always set by Problem.__post_init__

        s = rng.choice(np.array([-1.0, 1.0], dtype=problem.dtype), size=n)
        h_eff = evaluate.effective_field(j, h, s)
        curr_energy = evaluate.energy(j, h, c, s, h_eff=h_eff)

        best_energy = curr_energy
        best_spins = s.copy()

        # tabu_until[i] is the first step at which spin i is no longer tabu.
        tabu_until = np.zeros(n, dtype=np.int64)

        trajectory = np.empty((self.n_steps, n), dtype=problem.dtype) if self.record_trajectory else None
        steps_taken = 0

        for step in range(self.n_steps):
            delta = evaluate.spin_flip_energy_update(s, h_eff)

            best_k = -1
            best_delta = np.inf
            for k in range(n):
                is_tabu = tabu_until[k] > step
                aspiration = (curr_energy + delta[k]) < best_energy
                if (not is_tabu or aspiration) and delta[k] < best_delta:
                    best_delta = delta[k]
                    best_k = k

            if best_k == -1:
                break

            h_eff += evaluate.spin_flip_effective_field_update(j, s, best_k)
            s[best_k] = -s[best_k]
            curr_energy += best_delta
            tabu_until[best_k] = step + self.tabu_tenure

            if curr_energy < best_energy:
                best_energy = curr_energy
                best_spins = s.copy()

            if trajectory is not None:
                trajectory[step] = s
            steps_taken = step + 1

        if trajectory is not None:
            trajectory = trajectory[:steps_taken]

        meta = {
            "n_steps": self.n_steps,
            "tabu_tenure": self.tabu_tenure,
        }
        if trajectory is not None:
            meta["trajectory_kind"] = "spins"

        return Solution(
            spins=best_spins,
            energy=best_energy,
            time_s=time.perf_counter() - t0,
            solver_name=self.name,
            problem_id=problem.id,
            trajectory=trajectory,
            meta=meta,
        )
