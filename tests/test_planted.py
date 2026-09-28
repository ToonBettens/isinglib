from __future__ import annotations

import numpy as np
import pytest

from isinglib import generators
from isinglib.solvers import exhaustive
from isinglib.states import random_spins
from isinglib.topology import Topology, complete, erdos_renyi

# ------------------------------------------------------------------
# The state is a ground state
# ------------------------------------------------------------------


def test_every_coupling_is_satisfied_at_the_state() -> None:
    spins = random_spins(8, rng=1)
    p = generators.mattis(complete(8), spins, 0.7)
    assert p.energy(spins) == pytest.approx(-0.5 * np.abs(p.j).sum())


def test_state_and_its_flip_are_the_ground_states() -> None:
    t = erdos_renyi(9, p=0.5, rng=np.random.default_rng(2))
    spins = random_spins(9, rng=4)
    p = generators.mattis(t, spins)
    ground = exhaustive(p, n_best=2).top_energies
    assert ground[0] == pytest.approx(p.energy(spins))
    assert p.energy(-spins) == pytest.approx(p.energy(spins))


def test_an_aligned_bias_makes_the_state_unique() -> None:
    spins = random_spins(8, rng=5)
    p = generators.mattis(complete(8), spins).replace(h=0.1 * spins)
    sol = exhaustive(p)
    assert np.array_equal(sol.spins, spins)
    assert sol.degeneracy == 1


def test_all_ones_is_the_ferromagnet() -> None:
    t = complete(5)
    assert generators.mattis(t, np.ones(5)) == generators.constant(t, 1.0)


def test_couplings_live_only_on_the_topology() -> None:
    t = Topology(4, [0, 1], [1, 2])
    p = generators.mattis(t, np.array([1.0, -1.0, 1.0, -1.0]))
    assert p.j[0, 1] == -1.0 and p.j[1, 2] == -1.0
    assert np.count_nonzero(p.j) == 4


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


@pytest.mark.parametrize("state", [np.array([1.0, -1.0]), np.array([1.0, -1.0, 0.5, 1.0])])
def test_rejects_a_state_that_is_not_a_spin_configuration(state: np.ndarray) -> None:
    with pytest.raises(ValueError):
        generators.mattis(complete(4), state)


def test_rejects_negative_strength() -> None:
    with pytest.raises(ValueError):
        generators.mattis(complete(4), np.ones(4), -1.0)
