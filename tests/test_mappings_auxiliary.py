from __future__ import annotations

import itertools

import numpy as np
import pytest

from isinglib import Problem, generators
from isinglib.mappings import auxiliary
from isinglib.topology import complete


def _problem(n: int = 5, seed: int = 0) -> Problem:
    h = np.random.default_rng(seed + 1).normal(0.0, 1.0, n)
    return generators.gaussian(complete(n), rng=seed).replace(h=h, c=2.5)


def _states(n: int) -> list[np.ndarray]:
    return [np.array(bits) for bits in itertools.product((-1.0, 1.0), repeat=n)]


def test_encode_moves_the_fields_into_couplings_to_the_last_spin() -> None:
    p = _problem()
    big = auxiliary.encode(p)
    assert big.n == p.n + 1
    assert np.array_equal(big.j[: p.n, : p.n], p.j)
    assert np.array_equal(big.j[: p.n, p.n], p.h)
    assert np.all(big.h == 0.0)
    assert big.c == p.c


def test_every_state_keeps_the_energy_of_the_state_it_decodes_to() -> None:
    p = _problem()
    big = auxiliary.encode(p)
    for s in _states(big.n):
        assert big.energy(s) == pytest.approx(p.energy(auxiliary.decode(s)))


def test_ground_energies_are_equal() -> None:
    p = _problem()
    big = auxiliary.encode(p)
    assert min(map(big.energy, _states(big.n))) == pytest.approx(min(map(p.energy, _states(p.n))))


def test_decode_undoes_a_global_flip() -> None:
    s = np.array([1.0, -1.0, -1.0, 1.0])
    assert np.array_equal(auxiliary.decode(np.append(s, 1.0)), s)
    assert np.array_equal(auxiliary.decode(np.append(-s, -1.0)), s)


def test_without_fields_the_auxiliary_spin_is_free() -> None:
    p = _problem().replace(h=0.0)
    big = auxiliary.encode(p)
    assert big.n == p.n + 1
    assert np.all(big.j[p.n] == 0.0)


def test_encode_keeps_the_dtype() -> None:
    p = _problem().astype(np.float32)
    assert auxiliary.encode(p).dtype == np.float32


def test_decode_rejects_a_non_vector() -> None:
    with pytest.raises(ValueError, match="1-D"):
        auxiliary.decode(np.ones((2, 2)))
