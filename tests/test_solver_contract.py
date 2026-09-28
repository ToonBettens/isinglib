"""The solver contract: the one thing every solver must get right.

Register new solvers in ``SOLVERS`` — a solver not in this list does not exist
as far as CI is concerned.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from isinglib import (
    Problem,
    branch_and_bound,
    exhaustive,
    first_improvement,
    parallel_tempering,
    simulated_annealing,
    simulated_bifurcation,
    steepest_descent,
    tabu_search,
)

# (name, callable, kwargs, is_exact): exact solvers are additionally checked
# against the brute-force ground state.
SOLVERS: list[tuple[str, Callable[..., Any], dict[str, Any], bool]] = [
    ("exhaustive", exhaustive, {}, True),
    ("branch_and_bound", branch_and_bound, {}, True),
    ("steepest_descent", steepest_descent, {"rng": 0}, False),
    ("first_improvement", first_improvement, {"rng": 0}, False),
    ("simulated_annealing", simulated_annealing, {"rng": 0}, False),
    ("tabu_search", tabu_search, {"rng": 0}, False),
    ("parallel_tempering", parallel_tempering, {"rng": 0}, False),
    ("simulated_bifurcation", simulated_bifurcation, {"rng": 0}, False),
    ("simulated_bifurcation_discrete", simulated_bifurcation, {"rng": 0, "variant": "discrete"}, False),
]


@pytest.fixture(params=SOLVERS, ids=lambda s: s[0])
def solver(request: pytest.FixtureRequest) -> tuple[str, Callable[..., Any], dict[str, Any], bool]:
    return request.param


def test_energy_matches_problem_energy(
    solver: tuple[str, Callable[..., Any], dict[str, Any], bool], small_problem: Problem
) -> None:
    """The one thing that makes solvers comparable: whatever a solver computed
    internally, its reported energy must agree with `Problem.energy` on its own
    reported spins. Nothing else checks this."""
    _, fn, kwargs, _ = solver
    sol = fn(small_problem, **kwargs)
    assert sol.energy == pytest.approx(small_problem.energy(sol.spins))


def test_seeded_solve_is_deterministic(
    solver: tuple[str, Callable[..., Any], dict[str, Any], bool], small_problem: Problem
) -> None:
    _, fn, kwargs, _ = solver
    if "rng" not in kwargs:
        pytest.skip("solver takes no rng")
    first, second = fn(small_problem, **kwargs), fn(small_problem, **kwargs)
    assert second.energy == first.energy
    assert np.array_equal(second.spins, first.spins)


def test_exact_solvers_find_ground_state(
    solver: tuple[str, Callable[..., Any], dict[str, Any], bool],
    make_problem: Callable[..., Problem],
) -> None:
    _, fn, kwargs, is_exact = solver
    if not is_exact:
        pytest.skip("solver is not exact")

    p = make_problem(7, seed=42)
    sol = fn(p, **kwargs)

    # Brute-force ground-state energy over all 2^n assignments.
    n = p.n
    best = np.inf
    for bits in range(1 << n):
        s = np.array([1.0 if (bits >> k) & 1 else -1.0 for k in range(n)])
        best = min(best, p.energy(s))
    assert sol.energy == pytest.approx(best)
