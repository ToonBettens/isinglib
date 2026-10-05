from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt

from isinglib._evaluate import (
    effective_field,
    energy,
    spin_flip_effective_field_update,
    spin_flip_energy_update,
)
from isinglib.dtypes import DEFAULT_INDEX_DTYPE
from isinglib.problem import Problem
from isinglib.solvers.base import Solution
from isinglib.states import ensure_spins, random_spins

__all__ = ("simulated_annealing",)


def simulated_annealing(
    problem: Problem,
    *,
    initial: npt.ArrayLike | None = None,
    n_sweeps: int = 1_000,
    T_start: float = 2.0,
    T_end: float = 0.01,
    schedule: str = "geometric",
    rng: np.random.Generator | int | None = None,
    keep_flips: bool = False,
    keep_energy_trace: bool = False,
) -> Solution:
    """Simulated annealing with a monotonically decreasing temperature schedule.

    Each sweep proposes every spin once, in a fresh random order, at that
    sweep's temperature. A flip is accepted if it lowers energy, or with
    Boltzmann probability `exp(-ΔE / T)` otherwise. Between sweeps, temperature
    is lowerd according to the chosen `schedule`. The best state seen across
    all proposals is returned.

    Args:
        problem: The problem to solve.
        initial: Starting spin configuration. A random state is drawn if not given.
        n_sweeps: Number of sweeps; one sweep is `n` flip proposals.
        T_start: Initial temperature.
        T_end: Final temperature (must be > 0).
        schedule: `"geometric"` or `"linear"`.
        rng: Random generator or seed.
        keep_flips: If True, record the flipped spin indices. `-1` on rejection.
        keep_energy_trace: If True, record the energy after every proposal.

    Returns:
        A `Solution` with:
        - `problem_id`: fingerprint of `problem`.
        - `solver`: `"simulated_annealing"`.
        - `spins`: the best state found.
        - `energy`: its energy.
        - `elapsed_s`: wall-clock time the solve took, in seconds.
        - `n_accepted`: accepted flips out of `n_sweeps * n` proposals.
        - `flips`: spin index of every proposal, or `-1` if rejected, only when `keep_flips=True`.
        - `energy_trace`: energy after every proposal, only when `keep_energy_trace=True`.
    """
    if T_end <= 0:
        raise ValueError("T_end must be positive.")
    if T_start <= T_end:
        raise ValueError("T_start must be greater than T_end.")
    if schedule not in ("geometric", "linear"):
        raise ValueError("schedule must be 'geometric' or 'linear'.")

    t0 = time.perf_counter()
    _rng = np.random.default_rng(rng)
    j, h, c, n = problem.j, problem.h, problem.c, problem.n

    s = (
        random_spins(n, rng=_rng, dtype=problem.dtype)
        if initial is None
        else ensure_spins(initial, n, dtype=problem.dtype, copy=True)
    )
    h_eff = effective_field(j, h, s)
    curr_energy = energy(j, h, c, s, h_eff=h_eff)

    best_energy = curr_energy
    best_spins = s.copy()
    n_accepted = 0

    spacing = np.geomspace if schedule == "geometric" else np.linspace
    temps = spacing(T_start, T_end, n_sweeps)

    flips = np.empty(n_sweeps * n, dtype=DEFAULT_INDEX_DTYPE) if keep_flips else None
    energy_trace = np.empty(n_sweeps * n, dtype=problem.dtype) if keep_energy_trace else None

    step = 0
    for T in temps:
        order = _rng.permutation(n)
        log_uniform = np.log(_rng.random(n))

        for t in range(n):
            k = int(order[t])
            delta = spin_flip_energy_update(s, h_eff, i=k)
            accepted = delta < 0.0 or log_uniform[t] < -delta / T

            if accepted:
                h_eff += spin_flip_effective_field_update(j, s, k)
                s[k] = -s[k]
                curr_energy += delta
                n_accepted += 1

                if curr_energy < best_energy:
                    best_energy = curr_energy
                    best_spins = s.copy()

            if flips is not None:
                flips[step] = k if accepted else -1
            if energy_trace is not None:
                energy_trace[step] = curr_energy
            step += 1

    kwargs = {}
    if flips is not None:
        kwargs["flips"] = flips
    if energy_trace is not None:
        kwargs["energy_trace"] = energy_trace

    return Solution(
        problem_id=problem.fingerprint,
        solver="simulated_annealing",
        spins=best_spins,
        energy=best_energy,
        elapsed_s=time.perf_counter() - t0,
        n_accepted=n_accepted,
        **kwargs,
    )
