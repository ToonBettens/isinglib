from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from isinglib import Problem
from isinglib.solvers import local_search

RULES = ["steepest", "random", "first"]


def decoupled(n: int) -> Problem:
    """No couplings, fields 1 ... n: from all -1, every spin improves, each by a different gain."""
    return Problem(np.zeros((n, n)), h=np.arange(1.0, n + 1))


@pytest.mark.parametrize("rule", RULES)
def test_every_flip_lowers_the_energy_and_the_search_ends_in_a_local_minimum(
    make_problem: Callable[..., Problem], rule: str
) -> None:
    problem = make_problem(30, seed=3)
    initial = np.where(np.random.default_rng(3).random(30) < 0.5, 1.0, -1.0)
    sol = local_search(problem, rule=rule, initial=initial, rng=0)
    assert sol.rule == rule
    assert len(sol.flips) > 0
    s = initial.copy()
    energies = [problem.energy(s)]
    for k in sol.flips:
        s[k] = -s[k]
        energies.append(problem.energy(s))
    assert np.allclose(energies[1:], sol.energy_trace)
    assert np.all(np.diff(energies) < 0.0)
    assert np.all(2.0 * sol.spins * (problem.j @ sol.spins + problem.h) >= 0.0)


def test_steepest_takes_the_largest_gain(make_problem: Callable[..., Problem]) -> None:
    problem = make_problem(30, seed=4)
    initial = np.where(np.random.default_rng(4).random(30) < 0.5, 1.0, -1.0)
    sol = local_search(problem, initial=initial, rng=0)
    s = initial.copy()
    for k in sol.flips:
        delta = 2.0 * s * (problem.j @ s + problem.h)
        assert delta[k] == pytest.approx(delta.min())
        s[k] = -s[k]


def test_random_picks_every_improving_spin_equally_often() -> None:
    problem, initial = decoupled(4), -np.ones(4)
    rng = np.random.default_rng(0)
    first = [local_search(problem, rule="random", initial=initial, rng=rng).flips[0] for _ in range(4000)]
    counts = np.bincount(first, minlength=4)
    assert np.all(np.abs(counts - 1000) < 150), counts  # about 5 standard deviations


def test_first_flips_as_it_sweeps_in_a_random_order() -> None:
    problem, initial = decoupled(10), -np.ones(10)
    sol = local_search(problem, rule="first", initial=initial, rng=7)
    assert np.array_equal(sol.flips, np.random.default_rng(7).permutation(10))


def test_unknown_rule_raises(small_problem: Problem) -> None:
    with pytest.raises(ValueError, match="rule"):
        local_search(small_problem, rule="best")
