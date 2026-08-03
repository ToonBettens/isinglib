from __future__ import annotations

import numpy as np
import pytest

from isinglib import ExhaustiveSolver
from isinglib.problems import planted_solution
from isinglib.topology import complete, erdos_renyi


def test_planted_is_the_returned_ground_truth() -> None:
    t = complete(8)
    rng = np.random.default_rng(0)
    p = planted_solution(t, coupling=lambda n: rng.uniform(0.1, 1.0, n), bias=0.5, rng=1)
    planted = np.array(p.meta["planted_spins"])
    assert p.energy(planted) == pytest.approx(p.meta["ground_truth_energy"])


def test_planted_matches_exhaustive_ground_state() -> None:
    t = erdos_renyi(9, p=0.4, rng=np.random.default_rng(2))
    rng = np.random.default_rng(3)
    p = planted_solution(t, coupling=lambda n: rng.uniform(0.1, 1.0, n), bias=0.5, rng=4)
    planted = np.array(p.meta["planted_spins"])

    sol = ExhaustiveSolver().solve(p)
    assert sol.energy == pytest.approx(p.energy(planted))
    assert np.array_equal(sol.spins, planted)


def test_explicit_planted_spins_are_used() -> None:
    t = complete(4)
    planted = np.array([1.0, -1.0, 1.0, -1.0])
    p = planted_solution(t, coupling=1.0, bias=0.5, planted=planted)
    assert np.array_equal(np.array(p.meta["planted_spins"]), planted)
    assert p.energy(planted) == pytest.approx(p.meta["ground_truth_energy"])


def test_rejects_wrong_shape_planted() -> None:
    t = complete(4)
    with pytest.raises(ValueError):
        planted_solution(t, coupling=1.0, planted=np.array([1.0, -1.0]))


def test_rejects_non_pm1_planted() -> None:
    t = complete(4)
    with pytest.raises(ValueError):
        planted_solution(t, coupling=1.0, planted=np.array([1.0, -1.0, 0.5, 1.0]))


def test_rejects_negative_coupling() -> None:
    t = complete(4)
    with pytest.raises(ValueError):
        planted_solution(t, coupling=-1.0)


def test_rejects_negative_bias() -> None:
    t = complete(4)
    with pytest.raises(ValueError):
        planted_solution(t, coupling=1.0, bias=-1.0)


def test_zero_bias_leaves_flip_degeneracy() -> None:
    t = complete(6)
    p = planted_solution(t, coupling=1.0, bias=0.0, rng=5)
    planted = np.array(p.meta["planted_spins"])
    assert p.energy(planted) == pytest.approx(p.energy(-planted))
