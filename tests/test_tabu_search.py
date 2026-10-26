from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from isinglib import Problem
from isinglib.solvers import tabu_search


@pytest.mark.parametrize("tenure", [1, 2, 3, 10])
def test_a_flipped_spin_stays_tabu_for_tabu_tenure_steps(make_problem: Callable[..., Problem], tenure: int) -> None:
    """A spin flipped again within `tenure` steps must have reached a new best: the aspiration."""
    problem = make_problem(30, seed=1)
    sol = tabu_search(problem, n_steps=300, tabu_tenure=tenure, rng=0, keep_flips=True, keep_energy_trace=True)
    last_flip: dict[int, int] = {}
    best = np.inf  # the start's energy is not reported, so this only lets more through
    for step, (k, e) in enumerate(zip(sol.flips.tolist(), sol.energy_trace, strict=True)):
        if k in last_flip and step - last_flip[k] <= tenure:
            assert e < best, f"spin {k} flipped at step {last_flip[k]} and again at {step}"
        last_flip[k] = step
        best = min(best, e)


def test_tenure_one_prevents_flipping_straight_back(make_problem: Callable[..., Problem]) -> None:
    problem = make_problem(30, seed=2)
    flips = tabu_search(problem, n_steps=300, tabu_tenure=1, rng=0, keep_flips=True).flips
    assert len(flips) == 300
    assert np.all(flips[1:] != flips[:-1])
