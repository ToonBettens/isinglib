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

__all__ = ("tabu_search",)


def tabu_search(
    problem: Problem,
    *,
    initial: npt.ArrayLike | None = None,
    n_steps: int = 1_000,
    tabu_tenure: int = 10,
    rng: np.random.Generator | int | None = None,
    keep_flips: bool = False,
    keep_energy_trace: bool = False,
) -> Solution:
    """Tabu search: best-improvement local search with a recency-based tabu list.

    At each step the best non-tabu flip is chosen. A flipped spin is forbidden
    for `tabu_tenure` subsequent steps. The aspiration criterion overrides the
    tabu status when a flip would yield a new global best.

    Args:
        problem: The problem to solve.
        initial: Starting spin configuration. A random state is drawn if not given.
        n_steps: Number of search iterations.
        tabu_tenure: Number of steps a spin remains forbidden after being flipped.
        rng: Random generator or seed.
        keep_flips: If True, record the flipped spin indices.
        keep_energy_trace: If True, record the energy after every step.

    Returns:
        A `Solution` with:
        - `problem_id`: fingerprint of `problem`.
        - `solver`: `"tabu_search"`.
        - `spins`: the best state found.
        - `energy`: its energy.
        - `elapsed_s`: wall-clock time the solve took, in seconds.
        - `steps_taken`: steps actually run (different from `n_steps` when run ended in a tabu lock).
        - `flips`: spin index flipped at each step, only when `keep_flips=True`.
        - `energy_trace`: energy after each step, only when `keep_energy_trace=True`.
    """
    if tabu_tenure < 1:
        raise ValueError("tabu_tenure must be at least 1.")

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

    tabu_until = np.zeros(n, dtype=np.int64)

    flips = np.empty(n_steps, dtype=DEFAULT_INDEX_DTYPE) if keep_flips else None
    energy_trace = np.empty(n_steps, dtype=problem.dtype) if keep_energy_trace else None
    steps_taken = 0

    for step in range(n_steps):
        delta = spin_flip_energy_update(s, h_eff)

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

        h_eff += spin_flip_effective_field_update(j, s, best_k)
        s[best_k] = -s[best_k]
        curr_energy += best_delta
        tabu_until[best_k] = step + tabu_tenure

        if curr_energy < best_energy:
            best_energy = curr_energy
            best_spins = s.copy()

        if flips is not None:
            flips[step] = best_k
        if energy_trace is not None:
            energy_trace[step] = curr_energy
        steps_taken = step + 1

    if flips is not None:
        flips = flips[:steps_taken]
    if energy_trace is not None:
        energy_trace = energy_trace[:steps_taken]

    kwargs = {}
    if flips is not None:
        kwargs["flips"] = flips
    if energy_trace is not None:
        kwargs["energy_trace"] = energy_trace

    return Solution(
        spins=best_spins,
        energy=best_energy,
        elapsed_s=time.perf_counter() - t0,
        solver="tabu_search",
        problem_id=problem.fingerprint,
        steps_taken=steps_taken,
        **kwargs,
    )
