from __future__ import annotations

import itertools

import numpy as np
import pytest

from isinglib import Problem
from isinglib.mappings import qubo
from isinglib.topology import complete, fillers


def _problem(n: int = 5, seed: int = 0) -> Problem:
    t = complete(n)
    return Problem(
        t.fill(fillers.gaussian(rng=seed)),
        fillers.resolve(fillers.gaussian(rng=seed + 1), n),
    )


# ------------------------------------------------------------------
# encode
# ------------------------------------------------------------------


def test_encode_matches_the_qubo_objective() -> None:
    rng = np.random.default_rng(3)
    q = np.triu(rng.standard_normal((4, 4)))
    p = qubo.encode(q, offset=1.5)
    for bits in itertools.product((0, 1), repeat=4):
        x = np.array(bits, dtype=float)
        assert p.energy(2 * x - 1) == pytest.approx(float(x @ q @ x) + 1.5)


@pytest.mark.parametrize(
    "q",
    [
        np.array([[1.0, 4.0], [0.0, 2.0]]),   # upper triangular
        np.array([[1.0, 0.0], [4.0, 2.0]]),   # lower triangular
        np.array([[1.0, 2.0], [2.0, 2.0]]),   # symmetric
        np.array([[1.0, 3.0], [1.0, 2.0]]),   # split across both triangles
    ],
)
def test_encode_is_independent_of_how_q_is_split(q: np.ndarray) -> None:
    """`xᵀQx` depends only on `q + qᵀ`, so all four spellings must agree."""
    p = qubo.encode(q)
    for bits in itertools.product((0, 1), repeat=2):
        x = np.array(bits, dtype=float)
        assert p.energy(2 * x - 1) == pytest.approx(float(x @ q @ x))


# ------------------------------------------------------------------
# from_ising
# ------------------------------------------------------------------


def test_from_ising_preserves_the_objective() -> None:
    """For every x, xᵀQx + offset must equal the Ising energy at s = 2x - 1."""
    p = _problem(5)
    q, offset = qubo.from_ising(p)
    for bits in itertools.product((0, 1), repeat=p.n):
        x = np.array(bits, dtype=float)
        assert float(x @ q @ x) + offset == pytest.approx(p.energy(2 * x - 1))


def test_from_ising_returns_an_upper_triangular_matrix() -> None:
    q, _ = qubo.from_ising(_problem(4))
    assert np.array_equal(q, np.triu(q))


# ------------------------------------------------------------------
# decode
# ------------------------------------------------------------------


def test_decode_maps_spins_to_binary() -> None:
    assert np.array_equal(qubo.decode([-1.0, 1.0, -1.0]), [0.0, 1.0, 0.0])


# ------------------------------------------------------------------
# Round trip
# ------------------------------------------------------------------


def test_encode_inverts_from_ising() -> None:
    p = _problem(4, seed=7)
    q, offset = qubo.from_ising(p)
    assert qubo.encode(q, offset) == p
