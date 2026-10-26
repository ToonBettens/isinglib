from __future__ import annotations

import time

import numpy as np

from isinglib._evaluate import energy
from isinglib.problem import Problem
from isinglib.solvers.base import Solution

__all__ = ("max_stable_dt", "simulated_bifurcation")

_A0 = 1.0  # final pump amplitude


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

    The force on each position is `c0 * (j @ x + h)`. In the ballistic variant
    `j @ x` grows with the positions' amplitude while `h` does not, so early in
    the ramp `h` weighs more against the couplings than it does in the energy.
    To avoid it, fold `h` into couplings to an auxiliary spin before solving.
    The discrete variant has no such mismatch.

    Too large a `dt` makes the integration unstable: each step overshoots the
    fastest oscillation, the overshoot grows, and the positions slam into the
    walls together. How large is too large depends on the problem.
    `max_stable_dt` computes the limit for ballistic SB.

    Args:
        problem: The problem to solve.
        n_steps: Number of time steps.
        dt: Time step. The default, 1.25, is the reference value.
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

    c0 = _default_c0(j) if c0 is None else c0
    a0 = _A0
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


def max_stable_dt(problem: Problem, *, c0: float | None = None) -> float:
    """The largest stable time step of ballistic simulated bifurcation on `problem`.

    Each step is symplectic Euler, stable while `dt * omega < 2` for the
    fastest oscillation. That is at the start of the pump ramp, in the pattern
    the couplings oppose most: the eigenvector of the most negative eigenvalue
    `lambda_min` of `j`, with `omega**2 = a0 * (a0 + c0 * |lambda_min|)`.

    The reference `c0` assumes couplings of mixed signs, which put
    `c0 * |lambda_min|` near 1 and the limit near 1.41, above the default
    `dt = 1.25`. Where the couplings share a sign, far stiffer states occur
    and the limit drops below 1.25, making the reference `dt` unstable.

    At the limit itself that oscillation no longer decays. It is recommended
    to stay below it, e.g. at 0.9 times the limit.

    The discrete variant couples through signs, a bounded force, and fails
    another way: the whole state bounces from wall to wall. That has no simple
    bound. As an empiric rule, use 0.8 times the ballistic maximum stable `dt`.

    Args:
        problem: The problem to solve.
        c0: Coupling strength, as in `simulated_bifurcation` if not given.
    """
    c0 = _default_c0(problem.j) if c0 is None else c0
    lambda_min = float(np.linalg.eigvalsh(problem.j)[0]) if problem.n else 0.0
    return float(2.0 / np.sqrt(_A0 * (_A0 + c0 * max(-lambda_min, 0.0))))


def _default_c0(j: np.ndarray) -> float:
    """`0.5 * sqrt(n - 1) / ||j||_F`: puts the couplings' spectrum near `[-1, 1]` when their signs are mixed."""
    norm = float(np.sqrt(np.sum(j * j)))
    return 0.5 * np.sqrt(max(j.shape[0] - 1, 1)) / norm if norm > 0 else 0.5
