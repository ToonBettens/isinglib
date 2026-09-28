from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt

from isinglib.problem import Problem
from isinglib.solvers.base import Solution
from isinglib.states import ensure_spins, random_spins

__all__ = ("parallel_tempering",)


def parallel_tempering(
    problem: Problem,
    *,
    initial: npt.ArrayLike | None = None,
    n_sweeps: int = 1_000,
    n_replicas: int = 16,
    T_min: float = 0.05,
    T_max: float = 2.0,
    rng: np.random.Generator | int | None = None,
) -> Solution:
    """Parallel tempering (replica-exchange Monte Carlo).

    `n_replicas` copies of the system each run Metropolis single-spin flips at
    different fixed temperatures. Periodically, neighbouring replicas attempt to
    swap configurations. Low-temperature replicas sample local energy states but
    get stuck in local minima. High-temperature replicas flatten energy barriers
    and freely explore large volumes of the state space. The best seen by any
    replica is returned.

    Args:
        problem: The problem to solve.
        initial: Starting spin configuration for every replica. Randomly drawn if not given.
        n_sweeps: Number of sweeps, or equivalently, number of swap attempts.
        n_replicas: Number of replicas (at least 2).
        T_min: Coldest temperature.
        T_max: Hottest temperature.
        rng: Random generator or seed.

    Returns:
        A `Solution` with:
        - `problem_id`: fingerprint of `problem`.
        - `solver`: `"parallel_tempering"`.
        - `spins`: the best state found.
        - `energy`: its energy.
        - `elapsed_s`: wall-clock time the solve took, in seconds.
        - `n_accepted`: accepted flips, summed over replicas.
        - `n_swaps`: accepted replica swaps.
    """
    if n_replicas < 2:
        raise ValueError("n_replicas must be at least 2.")
    if T_min <= 0:
        raise ValueError("T_min must be positive.")
    if T_max <= T_min:
        raise ValueError("T_max must be greater than T_min.")

    t0 = time.perf_counter()
    _rng = np.random.default_rng(rng)
    j, h, c, n = problem.j, problem.h, problem.c, problem.n

    betas = 1.0 / np.geomspace(T_min, T_max, n_replicas)
    rows = np.arange(n_replicas)

    if initial is None:
        s = np.stack([random_spins(n, rng=_rng, dtype=problem.dtype) for _ in rows])
    else:
        s = np.tile(ensure_spins(initial, n, dtype=problem.dtype), (n_replicas, 1))
    h_eff = s @ j + h
    energies = -0.5 * np.sum(s * (h_eff + h), axis=1) + c

    best = int(np.argmin(energies))
    best_energy = float(energies[best])
    best_spins = s[best].copy()
    n_accepted = 0
    n_swaps = 0

    for sweep in range(n_sweeps):
        proposals = _rng.permuted(np.tile(np.arange(n), (n_replicas, 1)), axis=1).T
        log_uniform = np.log(_rng.random((n, n_replicas)))

        for t in range(n):
            k = proposals[t]
            sk = s[rows, k]
            delta = 2.0 * sk * h_eff[rows, k]
            accepted = (delta < 0.0) | (log_uniform[t] < -delta * betas)
            if not accepted.any():
                continue

            a, ka, ska = rows[accepted], k[accepted], sk[accepted]
            h_eff[a] -= 2.0 * ska[:, None] * j[ka]
            s[a, ka] = -ska
            energies[a] += delta[accepted]
            n_accepted += int(accepted.sum())

            m = int(np.argmin(energies))
            if energies[m] < best_energy:
                best_energy = float(energies[m])
                best_spins = s[m].copy()

        # Alternate even/odd pairs so every neighbouring pair gets a chance.
        for r in range(sweep % 2, n_replicas - 1, 2):
            log_ratio = (betas[r] - betas[r + 1]) * (energies[r] - energies[r + 1])
            if log_ratio >= 0.0 or np.log(_rng.random()) < log_ratio:
                pair = [r, r + 1]
                swapped = [r + 1, r]
                s[pair] = s[swapped]
                h_eff[pair] = h_eff[swapped]
                energies[pair] = energies[swapped]
                n_swaps += 1

    return Solution(
        problem_id=problem.fingerprint,
        solver="parallel_tempering",
        spins=best_spins,
        energy=best_energy,
        elapsed_s=time.perf_counter() - t0,
        n_accepted=n_accepted,
        n_swaps=n_swaps,
    )
