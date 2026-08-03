from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from isinglib import Problem


def test_energy_matches_explicit_formula() -> None:
    j = np.array([[0.0, 1.0], [1.0, 0.0]])
    h = np.array([0.5, -0.5])
    p = Problem(j=j, h=h, c=2.0)
    s = np.array([1.0, -1.0])
    # E = -0.5 sᵀ j s - hᵀ s + c = -0.5*(-2) - (0.5*1 + -0.5*-1) + 2 = 1 - 1 + 2 = 2
    assert p.energy(s) == pytest.approx(2.0)


def test_defaults_zero_bias_and_offset() -> None:
    p = Problem(j=np.zeros((3, 3)))
    assert p.n == 3
    assert p.h is not None and np.array_equal(p.h, np.zeros(3))
    assert p.c == 0.0
    assert p.dtype == np.dtype(np.float64)


@pytest.mark.parametrize(
    "j",
    [
        np.array([[0.0, 1.0, 0.0]]),            # not square
        np.array([[0.0, 1.0], [2.0, 0.0]]),     # not symmetric
        np.array([[1.0, 0.0], [0.0, 0.0]]),     # nonzero diagonal
    ],
)
def test_validate_rejects_bad_coupling(j: np.ndarray) -> None:
    with pytest.raises(ValueError):
        Problem(j=j)


def test_bias_length_must_match() -> None:
    with pytest.raises(ValueError):
        Problem(j=np.zeros((2, 2)), h=np.zeros(3))


def test_id_ignores_meta_and_dtype() -> None:
    j = np.array([[0.0, 1.0], [1.0, 0.0]])
    p1 = Problem(j=j)
    p2 = Problem(j=j, meta={"name": "g1"})
    p3 = Problem(j=j, dtype=np.dtype(np.float32))
    p4 = Problem(j=2 * j)
    # Same content, different meta/dtype → same id (and equal).
    assert p1.id == p2.id == p3.id
    assert p1 == p2 == p3
    # Different content → different id (and not equal).
    assert p1.id != p4.id
    assert p1 != p4


def test_frozen() -> None:
    p = Problem(j=np.zeros((2, 2)))
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.c = 5.0  # type: ignore
