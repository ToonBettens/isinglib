from __future__ import annotations

import itertools

import numpy as np
import pytest

from isinglib.mappings import maxcut
from isinglib.solvers import exhaustive
from isinglib.topology import complete, cycle, erdos_renyi


def _cut_by_definition(w: np.ndarray, side: np.ndarray) -> float:
    """Σ_{i<j} w_ij over pairs on different sides, written out as a loop."""
    n = len(side)
    return sum(w[i, k] for i in range(n) for k in range(i + 1, n) if side[i] != side[k])


# ------------------------------------------------------------------
# encode: the energy is minus the cut
# ------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_energy_is_minus_the_cut(seed: int) -> None:
    """The convention scoring against the literature rests on: exact, for every state."""
    rng = np.random.default_rng(seed)
    w = erdos_renyi(7, p=0.5, rng=rng).fill(lambda m: rng.normal(0.0, 1.0, m))
    p = maxcut.encode(w)
    for bits in itertools.product((-1.0, 1.0), repeat=7):
        s = np.array(bits)
        assert p.energy(s) == pytest.approx(-maxcut.objective(w, maxcut.decode(s)))


def test_odd_cycle_cuts_all_but_one_edge() -> None:
    """C_5 is not bipartite: the best cut leaves exactly one edge uncut."""
    p = maxcut.encode(cycle(5).fill(1.0))
    assert -exhaustive(p).energy == pytest.approx(4.0)


def test_complete_graph_max_cut() -> None:
    """K_n's maximum cut is floor(n/2) * ceil(n/2)."""
    p = maxcut.encode(complete(7).fill(1.0))
    assert -exhaustive(p).energy == pytest.approx(3 * 4)


def test_no_bias() -> None:
    assert not np.any(maxcut.encode(cycle(4).fill(1.0)).h)


def test_dtype() -> None:
    assert maxcut.encode(cycle(4).fill(1.0), dtype=np.float32).dtype == np.float32


def test_asymmetric_weights_rejected() -> None:
    with pytest.raises(ValueError):
        maxcut.encode(np.array([[0.0, 1.0], [0.0, 0.0]]))


# ------------------------------------------------------------------
# decode
# ------------------------------------------------------------------


def test_decode_gives_the_plus_side() -> None:
    assert np.array_equal(maxcut.decode([1.0, -1.0, -1.0, 1.0]), [True, False, False, True])


def test_decoded_sides_reproduce_the_cut() -> None:
    w = cycle(6).fill(1.0)
    p = maxcut.encode(w)
    sol = exhaustive(p)
    assert maxcut.objective(w, maxcut.decode(sol.spins)) == pytest.approx(-sol.energy)
    assert -sol.energy == pytest.approx(6.0)


# ------------------------------------------------------------------
# objective
# ------------------------------------------------------------------


def test_objective_matches_the_definition() -> None:
    rng = np.random.default_rng(4)
    w = complete(6).fill(lambda m: rng.normal(0.0, 1.0, m))
    for bits in itertools.product((False, True), repeat=6):
        side = np.array(bits)
        assert maxcut.objective(w, side) == pytest.approx(_cut_by_definition(w, side))


def test_objective_is_symmetric_in_the_sides() -> None:
    w = cycle(5).fill(1.0)
    side = np.array([True, False, False, True, False])
    assert maxcut.objective(w, side) == maxcut.objective(w, ~side)


def test_objective_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="side"):
        maxcut.objective(cycle(4).fill(1.0), [True, False])
