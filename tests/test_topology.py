from __future__ import annotations

import numpy as np
import pytest

from isinglib.topology import Topology

# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------


def test_canonical_edge_order() -> None:
    """Edges are normalised to src < dst and sorted, whatever order they arrive in."""
    t = Topology(4, [3, 1, 0], [0, 2, 2])
    s, d = t.edges()
    assert np.all(s < d)
    assert np.all(np.diff(s) >= 0)


def test_self_loop_rejected() -> None:
    with pytest.raises(ValueError):
        Topology(3, [0], [0])


def test_duplicates_removed() -> None:
    t = Topology(3, [0, 0], [1, 1])
    assert t.num_edges == 1


def test_empty_graph() -> None:
    t = Topology(5, [], [])
    assert t.num_edges == 0
    assert t.fill(1.0).shape == (5, 5)
    assert np.all(t.fill(1.0) == 0.0)


# ------------------------------------------------------------------
# Alternative constructors
# ------------------------------------------------------------------


def test_from_adjacency_roundtrip() -> None:
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    t = Topology.from_adjacency(adj)
    assert t.n == 3
    assert t.num_edges == 2


# ------------------------------------------------------------------
# Public methods
# ------------------------------------------------------------------


def test_degrees() -> None:
    t = Topology(4, [0, 0, 1], [1, 2, 2])
    deg = t.degrees()
    assert list(deg) == [2, 2, 2, 0]


def test_neighbors() -> None:
    t = Topology(3, [0, 1], [1, 2])
    assert set(t.neighbors(1)) == {0, 2}


def test_fill_scalar() -> None:
    t = Topology(3, [0, 1], [1, 2])
    j = t.fill(2.0)
    assert j.shape == (3, 3)
    assert j[0, 1] == j[1, 0] == 2.0
    assert np.all(np.diag(j) == 0.0)


def test_fill_callable() -> None:
    t = Topology(3, [0, 1], [1, 2])
    rng = np.random.default_rng(0)
    j = t.fill(rng.standard_normal)
    assert j.shape == (3, 3)
    assert np.allclose(j, j.T)
    assert np.all(np.diag(j) == 0.0)
