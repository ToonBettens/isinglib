from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ("Solution",)


@dataclass(frozen=True)
class Solution:
    """The output of a single solver run.

    Immutable value type passed between the solver and analysis layers.
    Bundles two things: the answer itself, and a record of the attempt that produced it.

    Attributes:
        spins: (n,) final binarized state in {-1, +1}.
        energy: Energy of the final state.
        time_s: Wall-clock solve time in seconds.
        solver_name: Stable name of the solver that produced this solution.
        problem_id: `Problem.fingerprint` of the originating problem, or None.
        trajectory: (T, n) optional per-step/per-sample record of intermediate states.
            What the `T` axis represents and how samples were chosen is solver-specific.
        meta: Solver-specific extras (n_steps, schedule, seed, ...).
    """

    spins: np.ndarray
    energy: float
    time_s: float
    solver_name: str
    problem_id: str | None = None
    trajectory: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        traj = "None" if self.trajectory is None else f"{self.trajectory.shape}"
        return (
            f"Solution(n={self.spins.shape[0]}, energy={self.energy:.6g}, "
            f"solver={self.solver_name!r}, time_s={self.time_s:.4g}, trajectory={traj})"
        )
