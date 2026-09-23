from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from isinglib import Problem


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
