from __future__ import annotations

import time

import numpy as np

from isinglib.core import evaluate
from isinglib.core.problem import Problem
from isinglib.core.solution import Solution
from isinglib.core.solver import Solver
from isinglib.solvers.utils.states import random_spins

__all__ = ("SimulatedAnnealingSolver",)


class SimulatedAnnealingSolver(Solver):
    """Simulated annealing with a monotonically decreasing temperature schedule.

    At each step a random spin is proposed; the flip is accepted if it lowers
    energy, or with Boltzmann probability `exp(-ΔE / T)` otherwise. The best
    state seen across all steps is returned, not the final state.

    Args:
        n_steps: Total number of spin-flip proposals.
        T_start: Initial temperature.
        T_end: Final temperature (must be > 0).
        schedule: `"geometric"` (exponential decay) or `"linear"`.
        rng: Random generator or seed. Seeded once here, so repeated `solve`
            calls consume fresh randomness instead of repeating one run.
        record_trajectory: If True, record the spin configuration after every
            step into `Solution.trajectory` (shape `(n_steps, n)`). Off by
            default — costs `O(n_steps * n)` memory, not worth paying for
            routine batch runs, only for one-off convergence diagnostics.
    """

    def __init__(
        self,
        n_steps: int = 10_000,
        T_start: float = 2.0,
        T_end: float = 0.01,
        schedule: str = "geometric",
        rng: np.random.Generator | int | None = None,
        record_trajectory: bool = False,
    ) -> None:
        if T_end <= 0:
            raise ValueError("T_end must be positive.")
        if T_start <= T_end:
            raise ValueError("T_start must be greater than T_end.")
        if schedule not in ("geometric", "linear"):
            raise ValueError("schedule must be 'geometric' or 'linear'.")
        self.n_steps = n_steps
        self.T_start = T_start
        self.T_end = T_end
        self.schedule = schedule
        self._rng = np.random.default_rng(rng)
        self.record_trajectory = record_trajectory

    @property
    def name(self) -> str:
        return "simulated_annealing"

    def solve(self, problem: Problem) -> Solution:
        t0 = time.perf_counter()
        rng = self._rng
        j, h, c, n = problem.j, problem.h, problem.c, problem.n
        assert h is not None  # always set by Problem.__post_init__

        s = random_spins(problem, rng=rng)
        h_eff = evaluate.effective_field(j, h, s)
        curr_energy = evaluate.energy(j, h, c, s, h_eff=h_eff)

        best_energy = curr_energy
        best_spins = s.copy()
        n_accepted = 0

        if self.schedule == "geometric":
            temps = np.geomspace(self.T_start, self.T_end, self.n_steps)
        else:
            temps = np.linspace(self.T_start, self.T_end, self.n_steps)

        spin_proposals = rng.integers(0, n, size=self.n_steps)
        log_uniform = np.log(rng.random(self.n_steps))

        trajectory = np.empty((self.n_steps, n), dtype=problem.dtype) if self.record_trajectory else None

        for step in range(self.n_steps):
            k = int(spin_proposals[step])
            delta = evaluate.spin_flip_energy_update(s, h_eff, i=k)

            if delta < 0.0 or log_uniform[step] < -delta / temps[step]:
                h_eff += evaluate.spin_flip_effective_field_update(j, s, k)
                s[k] = -s[k]
                curr_energy += delta
                n_accepted += 1

                if curr_energy < best_energy:
                    best_energy = curr_energy
                    best_spins = s.copy()

            if trajectory is not None:
                trajectory[step] = s

        meta = {
            "n_steps": self.n_steps,
            "T_start": self.T_start,
            "T_end": self.T_end,
            "schedule": self.schedule,
            "n_accepted": n_accepted,
            "acceptance_rate": n_accepted / self.n_steps,
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
