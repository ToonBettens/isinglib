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

__all__ = ("steepest_descent",)


def steepest_descent(
    problem: Problem,
    *,
    initial: npt.ArrayLike | None = None,
    rng: np.random.Generator | int | None = None,
) -> Solution:
    """Steepest-descent local search: flip the spin that most reduces energy.

    At each step, every spin's flip delta is checked and the single best one is
    taken. When several spins tie for the best delta, one is chosen at random.

    Args:
        problem: The problem to solve.
        initial: Starting spin configuration. A random state is drawn if not given.
        rng: Random generator or seed.

    Returns:
        A `Solution` with:
        - `problem_id`: fingerprint of `problem`.
        - `solver`: `"steepest_descent"`.
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
        delta = spin_flip_energy_update(s, h_eff)
        best = delta.min()
        if best >= 0.0:
            break
        ties = np.flatnonzero(delta == best)
        k = int(ties[0]) if ties.size == 1 else int(_rng.choice(ties))
        h_eff += spin_flip_effective_field_update(j, s, k)
        s[k] = -s[k]
        curr_energy += best
        flips.append(k)
        energy_trace.append(curr_energy)

    return Solution(
        problem_id=problem.fingerprint,
        solver="steepest_descent",
        spins=s,
        energy=curr_energy,
        elapsed_s=time.perf_counter() - t0,
        flips=np.array(flips, dtype=DEFAULT_INDEX_DTYPE),
        energy_trace=np.array(energy_trace, dtype=problem.dtype),
    )
