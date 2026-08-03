from __future__ import annotations

import numpy as np

from isinglib import Problem
from isinglib.topology import (
    barabasi_albert,
    complete,
    cycle,
    erdos_renyi,
    grid,
    king,
    path,
    ring,
    star,
    watts_strogatz,
)


def test_complete_edge_count() -> None:
    t = complete(5)
    assert t.num_edges == 5 * 4 // 2


def test_star_hub_degree() -> None:
    t = star(6)
    degrees = t.degrees()
    assert degrees[0] == 5
    assert np.all(degrees[1:] == 1)


def test_path_is_tridiagonal() -> None:
    t = path(5)
    assert t.num_edges == 4


def test_cycle_equals_ring_k2() -> None:
    t1 = cycle(6)
    t2 = ring(6, k=2)
    assert np.array_equal(t1.src, t2.src)
    assert np.array_equal(t1.dst, t2.dst)


def test_grid_4x4_edge_count() -> None:
    t = grid(4, 4)
    # 4x4 open grid: 4*3 horizontal + 3*4 vertical = 24 edges
    assert t.num_edges == 24


def test_grid_periodic_extra_edges() -> None:
    t_open = grid(3, 3)
    t_peri = grid(3, 3, periodic=True)
    assert t_peri.num_edges > t_open.num_edges


def test_king_has_more_edges_than_grid() -> None:
    t_grid = grid(4, 4)
    t_king = king(4, 4)
    assert t_king.num_edges > t_grid.num_edges


def test_erdos_renyi_empty_at_p0() -> None:
    t = erdos_renyi(10, p=0.0, rng=np.random.default_rng(0))
    assert t.num_edges == 0


def test_erdos_renyi_complete_at_p1() -> None:
    t = erdos_renyi(6, p=1.0, rng=np.random.default_rng(0))
    assert t.num_edges == 6 * 5 // 2


def test_barabasi_albert_node_count() -> None:
    t = barabasi_albert(20, m=3, rng=np.random.default_rng(1))
    assert t.n == 20


def test_watts_strogatz_node_count() -> None:
    t = watts_strogatz(20, k=4, p=0.1, rng=np.random.default_rng(2))
    assert t.n == 20


def test_gaussian_coupling() -> None:
    rng = np.random.default_rng(3)
    t = complete(8)
    p = Problem.from_topology(t, rng.standard_normal)
    p.validate()
    assert not np.all(p.j == p.j[0, 1])  # not all identical


def test_custom_bias() -> None:
    rng = np.random.default_rng(4)
    t = complete(5)
    p = Problem.from_topology(t, coupling=0.0, bias=rng.standard_normal)
    assert not np.all(p.h == 0.0)
