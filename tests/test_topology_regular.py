from __future__ import annotations

import numpy as np
import pytest

from isinglib import Problem
from isinglib.solvers import exhaustive
from isinglib.topology import (
    chimera,
    complete,
    cycle,
    grid,
    king,
    mobius_ladder,
    path,
    ring,
    star,
)

# ----------------------------------------------------------------
# complete
# ----------------------------------------------------------------


def test_complete_edge_count() -> None:
    t = complete(5)
    assert t.num_edges == 5 * 4 // 2


# ----------------------------------------------------------------
# star
# ----------------------------------------------------------------


def test_star_hub_degree() -> None:
    t = star(6)
    degrees = t.degrees()
    assert degrees[0] == 5
    assert np.all(degrees[1:] == 1)


# ----------------------------------------------------------------
# path
# ----------------------------------------------------------------


def test_path_is_tridiagonal() -> None:
    t = path(5)
    assert t.num_edges == 4


# ----------------------------------------------------------------
# cycle / ring
# ----------------------------------------------------------------


def test_cycle_equals_ring_k2() -> None:
    t1 = cycle(6)
    t2 = ring(6, k=2)
    assert np.array_equal(t1.src, t2.src)
    assert np.array_equal(t1.dst, t2.dst)


# ----------------------------------------------------------------
# grid
# ----------------------------------------------------------------


def test_grid_4x4_edge_count() -> None:
    t = grid(4, 4)
    # 4x4 open grid: 4*3 horizontal + 3*4 vertical = 24 edges
    assert t.num_edges == 24


def test_grid_periodic_extra_edges() -> None:
    t_open = grid(3, 3)
    t_peri = grid(3, 3, periodic=True)
    assert t_peri.num_edges > t_open.num_edges


# ----------------------------------------------------------------
# king
# ----------------------------------------------------------------


def test_king_has_more_edges_than_grid() -> None:
    t_grid = grid(4, 4)
    t_king = king(4, 4)
    assert t_king.num_edges > t_grid.num_edges


# ----------------------------------------------------------------
# mobius_ladder
# ----------------------------------------------------------------


def test_mobius_ladder_is_cubic() -> None:
    t = mobius_ladder(10)
    assert t.num_edges == 15
    assert np.all(t.degrees() == 3)


def test_mobius_ladder_4_is_complete() -> None:
    t, k4 = mobius_ladder(4), complete(4)
    assert np.array_equal(t.src, k4.src)
    assert np.array_equal(t.dst, k4.dst)


@pytest.mark.parametrize("n", [2, 3, 7])
def test_mobius_ladder_rejects_odd_or_small_n(n: int) -> None:
    with pytest.raises(ValueError):
        mobius_ladder(n)


@pytest.mark.parametrize(("n", "ground_energy"), [(6, -9.0), (8, -8.0)])
def test_mobius_ladder_antiferromagnet_ground_energy(n: int, ground_energy: float) -> None:
    """Bipartite when n/2 is odd, so every bond is satisfied (-3n/2); otherwise
    the twist frustrates the ladder and the max cut is 3n/2 - 2."""
    p = Problem(mobius_ladder(n).fill(-1.0))
    assert exhaustive(p).energy == pytest.approx(ground_energy)


# ----------------------------------------------------------------
# chimera
# ----------------------------------------------------------------


def test_chimera_matches_the_dwave_2000q_graph() -> None:
    t = chimera(16)
    assert t.n == 2048
    assert t.num_edges == 6016


def test_chimera_single_cell_is_complete_bipartite() -> None:
    t = chimera(1, t=3)
    assert t.num_edges == 9
    assert np.all(t.degrees() == 3)


def test_chimera_interior_degree_is_t_plus_2() -> None:
    t = chimera(3, t=4)
    # Cell (1, 1): its vertical and horizontal shores both have neighbours on both sides.
    interior = np.arange(((1 * 3 + 1) * 2) * 4, ((1 * 3 + 1) * 2 + 2) * 4)
    assert np.all(t.degrees()[interior] == 6)


def test_chimera_couples_vertical_down_and_horizontal_right() -> None:
    t = chimera(2, 2, t=2)
    edges = set(zip(t.src.tolist(), t.dst.tolist(), strict=True))
    # (0,0,0,1) = 1 couples to (1,0,0,1) = 9; (0,0,1,0) = 2 couples to (0,1,1,0) = 6.
    assert (1, 9) in edges
    assert (2, 6) in edges


def test_chimera_antiferromagnet_is_unfrustrated() -> None:
    """Chimera is bipartite, so every antiferromagnetic bond can be satisfied."""
    topo = chimera(1, 2, t=2)
    p = Problem(topo.fill(-1.0))
    assert exhaustive(p).energy == pytest.approx(-topo.num_edges)
