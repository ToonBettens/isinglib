from __future__ import annotations

import numpy as np

from isinglib.topology import complete, cycle, grid, king, path, ring, star

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
