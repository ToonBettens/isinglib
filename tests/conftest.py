from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from isinglib import Problem
from isinglib.topology import complete, fillers, star


def random_problem(n: int, seed: int = 0) -> Problem:
    """A small dense problem with a symmetric, zero-diagonal coupling matrix."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    j = (a + a.T) / 2.0
    np.fill_diagonal(j, 0.0)
    h = rng.standard_normal(n)
    return Problem(j, h)


@pytest.fixture
def make_problem() -> Callable[..., Problem]:
    """Factory fixture so tests need no cross-module import of the helper."""
    return random_problem


@pytest.fixture
def small_problem() -> Problem:
    return random_problem(6)


@pytest.fixture
def sparse_problem() -> Problem:
    """A star: 29 edges out of C(30, 2) = 435, well under the sparsity threshold."""
    t = star(30)
    return Problem(t.fill(fillers.uniform(0.1, 1.0, rng=0)), 0.5 * np.ones(t.n))


@pytest.fixture
def dense_problem() -> Problem:
    t = complete(10)
    return Problem(t.fill(fillers.uniform(0.1, 1.0, rng=1)), 0.5 * np.ones(t.n))
