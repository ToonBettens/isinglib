from __future__ import annotations

import pytest

from isinglib.io import hdf5

h5py = pytest.importorskip("h5py")



@pytest.mark.parametrize("layout", ["sparse", "dense"])
def test_round_trip_explicit_layout(tmp_path, sparse_problem, layout) -> None:
    p = sparse_problem
    path = tmp_path / "p.h5"
    hdf5.write(p, path, layout=layout)
    loaded = hdf5.read(path)
    assert loaded == p
    assert loaded.dtype == p.dtype


def test_auto_picks_sparse_for_sparse_graph(tmp_path, sparse_problem) -> None:
    p = sparse_problem
    path = tmp_path / "p.h5"
    hdf5.write(p, path, layout="auto")
    with h5py.File(path, "r") as f:
        assert f["problem"].attrs["layout"] == "sparse"
    assert hdf5.read(path) == p


def test_auto_picks_dense_for_dense_graph(tmp_path, dense_problem) -> None:
    p = dense_problem
    path = tmp_path / "p.h5"
    hdf5.write(p, path, layout="auto")
    with h5py.File(path, "r") as f:
        assert f["problem"].attrs["layout"] == "dense"
    assert hdf5.read(path) == p


def test_reads_legacy_file_without_layout_attr_as_dense(tmp_path, dense_problem) -> None:
    p = dense_problem
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


def test_multiple_keys_coexist(tmp_path, sparse_problem, dense_problem) -> None:
    a, b = sparse_problem, dense_problem
    path = tmp_path / "multi.h5"
    hdf5.write(a, path, key="a")
    hdf5.write(b, path, key="b")
    assert hdf5.read(path, key="a") == a
    assert hdf5.read(path, key="b") == b


def test_rejects_unknown_layout(tmp_path, sparse_problem) -> None:
    p = sparse_problem
    with pytest.raises(ValueError):
        hdf5.write(p, tmp_path / "p.h5", layout="bogus")  # type: ignore


def test_write_refuses_an_existing_key_by_default(tmp_path, sparse_problem, dense_problem) -> None:
    path = tmp_path / "p.h5"
    hdf5.write(sparse_problem, path, key="run1")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        hdf5.write(dense_problem, path, key="run1")


def test_overwrite_true_replaces_the_key(tmp_path, sparse_problem, dense_problem) -> None:
    path = tmp_path / "p.h5"
    hdf5.write(sparse_problem, path, key="run1")
    hdf5.write(dense_problem, path, key="run1", overwrite=True)
    assert hdf5.read(path, key="run1") == dense_problem


def test_a_new_key_needs_no_overwrite_even_when_the_file_exists(tmp_path, sparse_problem, dense_problem) -> None:
    path = tmp_path / "p.h5"
    hdf5.write(sparse_problem, path, key="run1")
    hdf5.write(dense_problem, path, key="run2")  # no overwrite=True needed
    assert hdf5.read(path, key="run1") == sparse_problem
    assert hdf5.read(path, key="run2") == dense_problem
