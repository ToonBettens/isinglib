from __future__ import annotations

import numpy as np

from isinglib.topology import barabasi_albert, erdos_renyi, watts_strogatz

# ------------------------------------------------------------------
# erdos_renyi
# ------------------------------------------------------------------


def test_erdos_renyi_empty_at_p0() -> None:
    t = erdos_renyi(10, p=0.0, rng=np.random.default_rng(0))
    assert t.num_edges == 0


def test_erdos_renyi_complete_at_p1() -> None:
    t = erdos_renyi(6, p=1.0, rng=np.random.default_rng(0))
    assert t.num_edges == 6 * 5 // 2


# ------------------------------------------------------------------
# barabasi_albert
# ------------------------------------------------------------------


def test_barabasi_albert_node_count() -> None:
    t = barabasi_albert(20, m=3, rng=np.random.default_rng(1))
    assert t.n == 20


# ------------------------------------------------------------------
# watts_strogatz
# ------------------------------------------------------------------


def test_watts_strogatz_node_count() -> None:
    t = watts_strogatz(20, k=4, p=0.1, rng=np.random.default_rng(2))
    assert t.n == 20
