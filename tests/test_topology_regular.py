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


@pytest.mark.parametrize(("dims", "degree"), [((5,), 2), ((3, 4), 8), ((3, 3, 4), 26)])
def test_periodic_king_has_3_to_the_d_minus_1_neighbours(dims: tuple[int, ...], degree: int) -> None:
    assert np.all(king(*dims, periodic=True).degrees() == degree)


def test_king_couples_diagonally_in_3d() -> None:
    t = king(3, 3, 3)
    edges = {(int(s), int(d)) for s, d in zip(*t.edges(), strict=True)}
    # (0,0,0) = 0 neighbours its body diagonal (1,1,1) = 13 and a face diagonal (0,1,1) = 4.
    assert {(0, 13), (0, 4)} <= edges
    # The centre (1,1,1) = 13 couples to all 26 other nodes.
    assert t.degrees()[13] == 26


def test_king_1d_is_the_path() -> None:
    assert np.array_equal(king(6).src, grid(6).src) and np.array_equal(king(6).dst, grid(6).dst)


@pytest.mark.parametrize(("dims", "periodic"), [((), False), ((3, 0), False), ((3, 2), True)])
def test_king_rejects_bad_shapes(dims: tuple[int, ...], periodic: bool) -> None:
    with pytest.raises(ValueError):
        king(*dims, periodic=periodic)


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


# ----------------------------------------------------------------
# grid in any dimension
# ----------------------------------------------------------------


def test_grid_numbers_nodes_row_major() -> None:
    t = grid(2, 3)
    edges = set(zip(t.src.tolist(), t.dst.tolist(), strict=True))
    assert edges == {(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)}


def test_grid_1d_is_a_path_or_cycle() -> None:
    assert np.array_equal(grid(6).src, path(6).src) and np.array_equal(grid(6).dst, path(6).dst)
    ring6 = grid(6, periodic=True)
    assert np.array_equal(ring6.src, cycle(6).src) and np.array_equal(ring6.dst, cycle(6).dst)


def test_grid_periodic_is_2d_regular() -> None:
    t = grid(3, 3, 3, 3, periodic=True)
    assert t.n == 81
    assert t.num_edges == 4 * 81
    assert np.all(t.degrees() == 8)


def test_grid_open_edge_count_for_unequal_sides() -> None:
    # Per axis: (size - 1) edges along each of the n / size lines.
    assert grid(2, 3, 4).num_edges == 12 + 16 + 18


def test_grid_couples_along_every_axis() -> None:
    edges = set(zip(*grid(3, 3, 3).edges(), strict=True))
    # (0,0,0) = 0 neighbours (1,0,0) = 9, (0,1,0) = 3 and (0,0,1) = 1.
    assert {(0, 9), (0, 3), (0, 1)} <= {(int(s), int(d)) for s, d in edges}


@pytest.mark.parametrize(("dims", "periodic"), [((), False), ((3, 0), False), ((0,), False), ((3, 2, 3), True)])
def test_grid_rejects_bad_shapes(dims: tuple[int, ...], periodic: bool) -> None:
    with pytest.raises(ValueError):
        grid(*dims, periodic=periodic)


# ----------------------------------------------------------------
# periodic lattices need room to wrap
# ----------------------------------------------------------------


@pytest.mark.parametrize("make", [grid, king])
@pytest.mark.parametrize(("rows", "cols"), [(1, 4), (2, 4), (4, 2)])
def test_periodic_grid_rejects_sides_too_short_to_wrap(make, rows: int, cols: int) -> None:
    with pytest.raises(ValueError, match=">= 3"):
        make(rows, cols, periodic=True)


@pytest.mark.parametrize(("make", "degree"), [(grid, 4), (king, 8)])
def test_periodic_grid_is_regular(make, degree: int) -> None:
    assert np.all(make(3, 5, periodic=True).degrees() == degree)
