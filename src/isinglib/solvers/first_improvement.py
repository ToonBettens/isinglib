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

__all__ = ("first_improvement",)


def first_improvement(
    problem: Problem,
    *,
    initial: npt.ArrayLike | None = None,
    rng: np.random.Generator | int | None = None,
) -> Solution:
    """First-improvement local search: flip the first improving spin found.

    Each sweep visits spins in a fresh random order and flips the first one
    whose delta is negative, immediately. A sweep that finds no improving
    spin ends the search. The order of visited spins is randomized.

    Args:
        problem: The problem to solve.
        initial: Starting spin configuration. A random state is drawn if not given.
        rng: Random generator or seed.

    Returns:
        A `Solution` with:
        - `problem_id`: fingerprint of `problem`.
        - `solver`: `"first_improvement"`.
        - `spins`: the state found.
        - `energy`: its energy.
        - `elapsed_s`: wall-clock time the solve took, in seconds.
        - `flips`: spin index flipped at each step, in order.
        - `energy_trace`: energy after each flip, same length as `flips`.
    """
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
    flips: list[int] = []
    energy_trace: list[float] = []

    while True:
        improved = False
        for k in _rng.permutation(n):
            k = int(k)
            delta = spin_flip_energy_update(s, h_eff, i=k)
            if delta < 0.0:
                h_eff += spin_flip_effective_field_update(j, s, k)
                s[k] = -s[k]
                curr_energy += delta
                flips.append(k)
                energy_trace.append(curr_energy)
                improved = True
                break
        if not improved:
            break

    return Solution(
        problem_id=problem.fingerprint,
        solver="first_improvement",
        spins=s,
        energy=curr_energy,
        elapsed_s=time.perf_counter() - t0,
        flips=np.array(flips, dtype=DEFAULT_INDEX_DTYPE),
        energy_trace=np.array(energy_trace, dtype=problem.dtype),
    )
