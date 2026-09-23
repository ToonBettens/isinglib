from __future__ import annotations

import numpy as np
import pytest

from isinglib import Problem
from isinglib.io import npz
from isinglib.topology import complete, fillers, star


def _sparse_problem() -> Problem:
    t = star(30)  # 29 edges out of C(30, 2) = 435 possible -> well under 25%
    return Problem(t.fill(fillers.uniform(0.1, 1.0, rng=0)), 0.5 * np.ones(t.n))


def _dense_problem() -> Problem:
    t = complete(10)
    return Problem(t.fill(fillers.uniform(0.1, 1.0, rng=1)), 0.5 * np.ones(t.n))


@pytest.mark.parametrize("encoding", ["sparse", "dense"])
def test_round_trip_explicit_encoding(tmp_path, encoding) -> None:
    p = _sparse_problem()
    path = tmp_path / "p.npz"
    npz.write(p, path, encoding=encoding)
    loaded = npz.read(path)
    assert loaded == p
    assert loaded.dtype == p.dtype


def test_auto_picks_sparse_for_sparse_graph(tmp_path) -> None:
    p = _sparse_problem()
    path = tmp_path / "p.npz"
    npz.write(p, path, encoding="auto")
    with np.load(path) as data:
        assert str(data["encoding"]) == "sparse"
    assert npz.read(path) == p


def test_auto_picks_dense_for_dense_graph(tmp_path) -> None:
    p = _dense_problem()
    path = tmp_path / "p.npz"
    npz.write(p, path, encoding="auto")
    with np.load(path) as data:
        assert str(data["encoding"]) == "dense"
    assert npz.read(path) == p


def test_rejects_unknown_encoding(tmp_path) -> None:
    p = _sparse_problem()
    with pytest.raises(ValueError):
        npz.write(p, tmp_path / "p.npz", encoding="bogus")  # type: ignore
