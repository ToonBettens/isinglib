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

__all__ = ("local_search",)

_RULES = ("steepest", "random", "first")


def local_search(
    problem: Problem,
    *,
    rule: str = "steepest",
    initial: npt.ArrayLike | None = None,
    rng: np.random.Generator | int | None = None,
) -> Solution:
    """Local search: flip one improving spin at a time until none is left.

    Every flip lowers the energy, so the search ends in a single-flip local
    minimum. The rules differ only in which improving spin is flipped next:

    - `"steepest"`: the one that lowers the energy most; ties broken at random.
    - `"random"`: any of them, each equally likely.
    - `"first"`: the next one in a sweep over the spins.

    Args:
        problem: The problem to solve.
        rule: `"steepest"`, `"random"` or `"first"`.
        initial: Starting spin configuration. A random state is drawn if not given.
        rng: Random generator or seed.

    Returns:
        A `Solution` with:
        - `problem_id`: fingerprint of `problem`.
        - `solver`: `"local_search"`.
        - `spins`: the state found.
        - `energy`: its energy.
        - `elapsed_s`: wall-clock time the solve took, in seconds.
        - `rule`: the rule used.
        - `flips`: spin index flipped at each step, in order.
        - `energy_trace`: energy after each flip, same length as `flips`.
    """
    if rule not in _RULES:
        raise ValueError(f"rule must be one of {_RULES}, got {rule!r}.")

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

    def flip(k: int, delta: float) -> None:
        nonlocal h_eff, curr_energy
        h_eff += spin_flip_effective_field_update(j, s, k)
        s[k] = -s[k]
        curr_energy += delta
        flips.append(k)
        energy_trace.append(curr_energy)

    if rule == "first":
        improved = True
        while improved:
            improved = False
            for k in _rng.permutation(n):
                delta = spin_flip_energy_update(s, h_eff, i=int(k))
                if delta < 0.0:
                    flip(int(k), delta)
                    improved = True
    else:
        while True:
            delta = spin_flip_energy_update(s, h_eff)
            improving = delta < 0.0
            if not improving.any():
                break
            candidates = np.flatnonzero(delta == delta.min() if rule == "steepest" else improving)
            k = int(candidates[0]) if candidates.size == 1 else int(_rng.choice(candidates))
            flip(k, float(delta[k]))

    return Solution(
        problem_id=problem.fingerprint,
        solver="local_search",
        spins=s,
        energy=curr_energy,
        elapsed_s=time.perf_counter() - t0,
        rule=rule,
        flips=np.array(flips, dtype=DEFAULT_INDEX_DTYPE),
        energy_trace=np.array(energy_trace, dtype=problem.dtype),
    )
