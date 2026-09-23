from __future__ import annotations

import numpy as np
import pytest

from isinglib import Problem
from isinglib.io import hdf5
from isinglib.topology import complete, fillers, star

h5py = pytest.importorskip("h5py")


def _sparse_problem() -> Problem:
    t = star(30)
    return Problem(t.fill(fillers.uniform(0.1, 1.0, rng=0)), 0.5 * np.ones(t.n))


def _dense_problem() -> Problem:
    t = complete(10)
    return Problem(t.fill(fillers.uniform(0.1, 1.0, rng=1)), 0.5 * np.ones(t.n))


@pytest.mark.parametrize("encoding", ["sparse", "dense"])
def test_round_trip_explicit_encoding(tmp_path, encoding) -> None:
    p = _sparse_problem()
    path = tmp_path / "p.h5"
    hdf5.write(p, path, encoding=encoding)
    loaded = hdf5.read(path)
    assert loaded == p
    assert loaded.dtype == p.dtype


def test_auto_picks_sparse_for_sparse_graph(tmp_path) -> None:
    p = _sparse_problem()
    path = tmp_path / "p.h5"
    hdf5.write(p, path, encoding="auto")
    with h5py.File(path, "r") as f:
        assert f["problem"].attrs["encoding"] == "sparse"
    assert hdf5.read(path) == p


def test_auto_picks_dense_for_dense_graph(tmp_path) -> None:
    p = _dense_problem()
    path = tmp_path / "p.h5"
    hdf5.write(p, path, encoding="auto")
    with h5py.File(path, "r") as f:
        assert f["problem"].attrs["encoding"] == "dense"
    assert hdf5.read(path) == p


def test_reads_legacy_file_without_encoding_attr_as_dense(tmp_path) -> None:
    p = _dense_problem()
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "a") as f:
        grp = f.create_group("problem")
        grp.create_dataset("j", data=p.j)
        grp.create_dataset("h", data=p.h)
        grp.attrs["c"] = p.c
        grp.attrs["dtype"] = str(p.dtype)
        grp.attrs["meta"] = "{}"

    loaded = hdf5.read(path)
    assert loaded.h is not None and p.h is not None  # always set by Problem.__post_init__
    assert loaded.j.tolist() == p.j.tolist()
    assert loaded.h.tolist() == p.h.tolist()


def test_multiple_keys_coexist(tmp_path) -> None:
    a, b = _sparse_problem(), _dense_problem()
    path = tmp_path / "multi.h5"
    hdf5.write(a, path, key="a")
    hdf5.write(b, path, key="b")
    assert hdf5.read(path, key="a") == a
    assert hdf5.read(path, key="b") == b


def test_rejects_unknown_encoding(tmp_path) -> None:
    p = _sparse_problem()
    with pytest.raises(ValueError):
        hdf5.write(p, tmp_path / "p.h5", encoding="bogus")  # type: ignore
