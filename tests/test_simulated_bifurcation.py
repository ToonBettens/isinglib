from __future__ import annotations

import numpy as np
import pytest

from isinglib import Problem
from isinglib.solvers import max_stable_dt, simulated_bifurcation


def antiferromagnet(n: int) -> Problem:
    """Every pair coupled at -1, as in unweighted max-cut on a complete graph: all spins
    moving together is the stiffest pattern."""
    return Problem(-(np.ones((n, n)) - np.eye(n)))


def test_max_stable_dt_from_the_smallest_eigenvalue() -> None:
    # lambda_min = -(n - 1) = -29, so the limit is 2 / sqrt(1 + c0 * 29)
    assert max_stable_dt(antiferromagnet(30), c0=0.1) == pytest.approx(2.0 / np.sqrt(1.0 + 2.9))


def test_max_stable_dt_ignores_the_scale_of_the_couplings() -> None:
    problem = antiferromagnet(30)
    assert max_stable_dt(problem.replace(j=7.0 * problem.j)) == pytest.approx(max_stable_dt(problem))


def test_max_stable_dt_is_near_1_41_for_mixed_signs() -> None:
    rng = np.random.default_rng(0)
    j = np.triu(rng.choice([-1.0, 1.0], size=(400, 400)), 1)
    assert max_stable_dt(Problem(j + j.T)) == pytest.approx(np.sqrt(2.0), rel=0.05)


def test_ballistic_sb_is_stable_below_the_limit_and_not_above() -> None:
    problem = antiferromagnet(30)
    limit = max_stable_dt(problem)
    assert limit < 1.25  # the default step is unstable here

    def all_equal(dt: float) -> bool:
        spins = simulated_bifurcation(problem, dt=dt, n_steps=1000, rng=0).spins
        return bool(abs(spins.sum()) == problem.n)

    assert not all_equal(0.9 * limit)
    assert all_equal(1.1 * limit)
