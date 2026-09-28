from __future__ import annotations

import sys
import time

import numpy as np

from isinglib.problem import Problem
from isinglib.solvers.base import Solution

__all__ = ("branch_and_bound",)


def branch_and_bound(problem: Problem) -> Solution:
    """Exact ground state, pruning subtrees a cheap lower bound already rules out.

    Spins are fixed one at a time, most-coupled first (`sum_j |j_ij| + |h_i|`,
    descending, computed once up front). At each partial assignment the
    remaining energy is lower-bounded by: the exact energy of what's fixed so
    far, plus the exact best case for each free spin's own field), plus the
    loosest possible case for the interactions among the free spins themselves.
    Once that lower bound is no better than the best complete solution already
    found, the whole subtree under it is skipped, since nothing under it could
    ever beat the incumbent. Whichever sign a free spin's own field already
    prefers is tried first, so a strong incumbent tends to show up early and
    prunes harder for the rest of the search.

    Args:
        problem: The problem to solve.

    Returns:
        A `Solution` with:
        - `problem_id`: fingerprint of `problem`.
        - `solver`: `"branch_and_bound"`.
        - `spins`: the exact ground state.
        - `energy`: its energy.
        - `elapsed_s`: wall-clock time the solve took, in seconds.
        - `visited`: partial and complete assignments actually explored.
    """
    t0 = time.perf_counter()
    j, h, c, n = problem.j, problem.h, problem.c, problem.n

    if n == 0:
        return Solution(
            problem_id=problem.fingerprint,
            solver="branch_and_bound",
            spins=np.empty(0, dtype=problem.dtype),
            energy=float(c),
            elapsed_s=time.perf_counter() - t0,
            visited=1,
        )

    abs_j = np.abs(j)
    influence = abs_j.sum(axis=1) + np.abs(h)
    order = np.argsort(-influence)

    remaining_pairs_abs = np.empty(n + 1)
    remaining_pairs_abs[0] = 0.5 * float(abs_j.sum())
    for d in range(n):
        k = order[d]
        still_free = order[d + 1 :]
        remaining_pairs_abs[d + 1] = remaining_pairs_abs[d] - float(abs_j[k, still_free].sum())

    h_eff = h.astype(np.float64, copy=True)
    s = np.empty(n, dtype=problem.dtype)
    best_spins = np.empty(n, dtype=problem.dtype)
    best_energy = np.inf
    visited = 0

    sys.setrecursionlimit(max(sys.getrecursionlimit(), n + 100))

    def recurse(depth: int, efixed: float) -> None:
        nonlocal best_energy, visited
        visited += 1
        if depth == n:
            if efixed < best_energy:
                best_energy = efixed
                best_spins[:] = s
            return

        idx = int(order[depth])
        hi = float(h_eff[idx])
        preferred = 1.0 if hi >= 0.0 else -1.0
        remaining_free = order[depth + 1 :]
        bound_rest = remaining_pairs_abs[depth + 1]

        for v in (preferred, -preferred):
            new_efixed = efixed - v * hi
            h_eff[remaining_free] += v * j[idx, remaining_free]
            s[idx] = v

            lb = new_efixed - float(np.abs(h_eff[remaining_free]).sum()) - bound_rest
            if lb < best_energy:
                recurse(depth + 1, new_efixed)

            h_eff[remaining_free] -= v * j[idx, remaining_free]

    recurse(0, float(c))

    return Solution(
        problem_id=problem.fingerprint,
        solver="branch_and_bound",
        spins=best_spins,
        energy=float(best_energy),
        elapsed_s=time.perf_counter() - t0,
        visited=visited,
    )
