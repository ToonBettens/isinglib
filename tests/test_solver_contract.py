"""The solver contract.

Every solver must pass these checks on a small problem. Register new solvers in
``SOLVERS`` — a solver not in this list does not exist as far as CI is concerned.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from isinglib import (
    ExhaustiveSolver,
    GreedySolver,
    Problem,
    SimulatedAnnealingSolver,
    Solution,
    Solver,
    TabuSearchSolver,
)

# (solver_instance, is_exact): exact solvers are additionally checked against the
# brute-force ground state.
SOLVERS: list[tuple[Solver, bool]] = [
    (ExhaustiveSolver(), True),
    (GreedySolver(n_restarts=5, rng=0), False),
    (SimulatedAnnealingSolver(rng=0), False),
    (TabuSearchSolver(rng=0), False),
]


@pytest.fixture(params=SOLVERS, ids=lambda sb: sb[0].name)
def solver_and_exactness(request: pytest.FixtureRequest) -> tuple[Solver, bool]:
    return request.param


def test_solver_contract(solver_and_exactness: tuple[Solver, bool], small_problem: Problem) -> None:
    solver, _ = solver_and_exactness
    sol = solver.solve(small_problem)

    assert isinstance(sol, Solution)
    assert sol.spins.shape == (small_problem.n,)
    assert set(np.unique(sol.spins)).issubset({-1.0, 1.0})
    assert np.isfinite(sol.energy)
    assert sol.energy == pytest.approx(small_problem.energy(sol.spins))
    assert sol.time_s >= 0.0
    assert sol.solver_name == solver.name
    assert sol.problem_id == small_problem.id


def test_exact_solvers_find_ground_state(
    solver_and_exactness: tuple[Solver, bool],
    make_problem: Callable[..., Problem],
) -> None:
    solver, is_exact = solver_and_exactness
    if not is_exact:
        pytest.skip("solver is not exact")

    p = make_problem(7, seed=42)
    sol = solver.solve(p)

    # Brute-force ground-state energy over all 2^n assignments.
    n = p.n
    best = np.inf
    for bits in range(1 << n):
        s = np.array([1.0 if (bits >> k) & 1 else -1.0 for k in range(n)])
        best = min(best, p.energy(s))
    assert sol.energy == pytest.approx(best)
