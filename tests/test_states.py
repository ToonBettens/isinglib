from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from isinglib import Problem
from isinglib.solvers.utils.states import constant_spins, random_spins


def test_random_spins_shape_dtype_and_values(make_problem: Callable[..., Problem]) -> None:
    p = make_problem(10, seed=0)
    s = random_spins(p, rng=0)
    assert s.shape == (p.n,)
    assert s.dtype == p.dtype
    assert set(np.unique(s)).issubset({-1.0, 1.0})


def test_random_spins_is_reproducible_with_same_rng(make_problem: Callable[..., Problem]) -> None:
    p = make_problem(10, seed=0)
    s1 = random_spins(p, rng=42)
    s2 = random_spins(p, rng=42)
    assert np.array_equal(s1, s2)


def test_random_spins_p_zero_and_one_are_deterministic(make_problem: Callable[..., Problem]) -> None:
    p = make_problem(10, seed=0)
    assert np.array_equal(random_spins(p, p=0.0, rng=0), np.full(p.n, -1.0))
    assert np.array_equal(random_spins(p, p=1.0, rng=0), np.full(p.n, 1.0))


def test_constant_spins_shape_dtype_and_value(make_problem: Callable[..., Problem]) -> None:
    p = make_problem(5, seed=0)
    s = constant_spins(p, -1.0)
    assert s.shape == (p.n,)
    assert s.dtype == p.dtype
    assert np.array_equal(s, np.full(p.n, -1.0))


def test_constant_spins_defaults_to_plus_one(make_problem: Callable[..., Problem]) -> None:
    p = make_problem(5, seed=0)
    assert np.array_equal(constant_spins(p), np.full(p.n, 1.0))


def test_constant_spins_rejects_invalid_value(make_problem: Callable[..., Problem]) -> None:
    p = make_problem(5, seed=0)
    with pytest.raises(ValueError):
        constant_spins(p, 0.5)
