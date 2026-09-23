from __future__ import annotations

import numpy as np
import pytest

from isinglib import ExhaustiveSolver
from isinglib.generators import planted_solution
from isinglib.states import random_spins
from isinglib.topology import complete, erdos_renyi, fillers

# ------------------------------------------------------------------
# The planted state is a ground state
# ------------------------------------------------------------------


def test_planted_is_a_ground_state_of_its_own_problem() -> None:
    spins = random_spins(8, rng=1)
    p = planted_solution(
        complete(8), fillers.uniform(0.1, 1.0, rng=0), bias=0.5, planted=spins
    )
    # Every term is individually satisfied at `spins`, so no state beats it.
    assert p.energy(spins) == pytest.approx(-0.5 * np.abs(p.j).sum() - np.abs(p.h).sum())


def test_planted_matches_exhaustive_ground_state() -> None:
    t = erdos_renyi(9, p=0.4, rng=np.random.default_rng(2))
    spins = random_spins(9, rng=4)
    p = planted_solution(t, fillers.uniform(0.1, 1.0, rng=3), bias=0.5, planted=spins)

    sol = ExhaustiveSolver().solve(p)
    assert sol.energy == pytest.approx(p.energy(spins))
    assert np.array_equal(sol.spins, spins)


def test_ground_truth_energy_is_recoverable_by_the_caller() -> None:
    """No meta: the caller holds the spins, so the energy is one call away."""
    spins = np.array([1.0, -1.0, 1.0, -1.0])
    p = planted_solution(complete(4), coupling=1.0, bias=0.5, planted=spins)
    assert p.energy(spins) == pytest.approx(-0.5 * np.abs(p.j).sum() - np.abs(p.h).sum())


def test_zero_bias_leaves_flip_degeneracy() -> None:
    spins = random_spins(6, rng=5)
    p = planted_solution(complete(6), coupling=1.0, bias=0.0, planted=spins)
    assert p.energy(spins) == pytest.approx(p.energy(-spins))


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


def test_rejects_wrong_shape_planted() -> None:
    with pytest.raises(ValueError):
        planted_solution(complete(4), coupling=1.0, planted=np.array([1.0, -1.0]))


def test_rejects_non_pm1_planted() -> None:
    with pytest.raises(ValueError):
        planted_solution(complete(4), coupling=1.0, planted=np.array([1.0, -1.0, 0.5, 1.0]))


def test_rejects_negative_coupling() -> None:
    with pytest.raises(ValueError):
        planted_solution(complete(4), coupling=-1.0, planted=np.ones(4))


def test_rejects_negative_bias() -> None:
    with pytest.raises(ValueError):
        planted_solution(complete(4), coupling=1.0, bias=-1.0, planted=np.ones(4))
