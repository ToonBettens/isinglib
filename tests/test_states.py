from __future__ import annotations

import numpy as np
import pytest

from isinglib.states import constant_spins, random_spins

# ------------------------------------------------------------------
# random_spins
# ------------------------------------------------------------------


def test_random_spins_shape_dtype_and_values() -> None:
    s = random_spins(10, rng=0)
    assert s.shape == (10,)
    assert s.dtype == np.float64
    assert set(np.unique(s)).issubset({-1.0, 1.0})


def test_random_spins_is_reproducible_with_same_rng() -> None:
    assert np.array_equal(random_spins(10, rng=42), random_spins(10, rng=42))


def test_random_spins_p_zero_and_one_are_deterministic() -> None:
    assert np.array_equal(random_spins(10, p=0.0, rng=0), np.full(10, -1.0))
    assert np.array_equal(random_spins(10, p=1.0, rng=0), np.full(10, 1.0))


def test_random_spins_respects_dtype() -> None:
    assert random_spins(5, rng=0, dtype=np.float32).dtype == np.float32


# ------------------------------------------------------------------
# constant_spins
# ------------------------------------------------------------------


def test_constant_spins_shape_dtype_and_value() -> None:
    s = constant_spins(5, -1.0)
    assert s.shape == (5,)
    assert s.dtype == np.float64
    assert np.array_equal(s, np.full(5, -1.0))


def test_constant_spins_defaults_to_plus_one() -> None:
    assert np.array_equal(constant_spins(5), np.full(5, 1.0))


def test_constant_spins_respects_dtype() -> None:
    assert constant_spins(5, dtype=np.float32).dtype == np.float32


def test_constant_spins_rejects_invalid_value() -> None:
    with pytest.raises(ValueError):
        constant_spins(5, 0.5)
