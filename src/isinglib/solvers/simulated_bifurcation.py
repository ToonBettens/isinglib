from __future__ import annotations

import time

import numpy as np

from isinglib._evaluate import energy
from isinglib.problem import Problem
from isinglib.solvers.base import Solution

__all__ = ("simulated_bifurcation",)


def simulated_bifurcation(
    problem: Problem,
    *,
    n_steps: int = 1_000,
    dt: float = 1.25,
    variant: str = "ballistic",
    c0: float | None = None,
    rng: np.random.Generator | int | None = None,
) -> Solution:
    """Simulated bifurcation.

    Each spin is a particle with position `x_i` and momentum `y_i`. The
    positions are driven by the coupling field while a pump `a(t)` ramps
    linearly from 0 to `a0 = 1`, bifurcating every `x_i` towards +1 or -1.
    Positions that leave `[-1, 1]` are clamped to the wall with zero momentum.
    The result is the sign of the final positions.

    `"ballistic"` couples through the positions themselves.
    `"discrete"` couples through their signs.

    Args:
        problem: The problem to solve.
        n_steps: Number of time steps.
        dt: Time step.
        variant: `"ballistic"` or `"discrete"`.
        c0: Coupling strength. Defaults to `0.5 * sqrt(n - 1) / ||j||_F`.
        rng: Random generator or seed.

    Returns:
        A `Solution` with:
        - `problem_id`: fingerprint of `problem`.
        - `solver`: `"simulated_bifurcation"`.
        - `spins`: sign of the final positions.
        - `energy`: its energy.
        - `elapsed_s`: wall-clock time the solve took, in seconds.
    """
    if variant not in ("ballistic", "discrete"):
        raise ValueError("variant must be 'ballistic' or 'discrete'.")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    t0 = time.perf_counter()
    _rng = np.random.default_rng(rng)
    j, h, c, n = problem.j, problem.h, problem.c, problem.n

    if c0 is None:
        norm = float(np.sqrt(np.sum(j * j)))
        c0 = 0.5 * np.sqrt(max(n - 1, 1)) / norm if norm > 0 else 0.5

    a0 = 1.0
    x = _rng.uniform(-0.1, 0.1, size=n)
    y = _rng.uniform(-0.1, 0.1, size=n)
    discrete = variant == "discrete"

    for a in np.linspace(0.0, a0, n_steps):
        field = j @ (np.sign(x) if discrete else x) + h
        y += (-(a0 - a) * x + c0 * field) * dt
        x += a0 * y * dt
        walls = np.abs(x) > 1.0
        x[walls] = np.sign(x[walls])
        y[walls] = 0.0

    spins = np.where(x >= 0.0, 1.0, -1.0).astype(problem.dtype)

    return Solution(
        problem_id=problem.fingerprint,
        solver="simulated_bifurcation",
        spins=spins,
        energy=energy(j, h, c, spins),
        elapsed_s=time.perf_counter() - t0,
    )
