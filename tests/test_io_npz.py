from __future__ import annotations

import numpy as np
import pytest

from isinglib import Problem
from isinglib.io import npz


@pytest.mark.parametrize("layout", ["sparse", "dense"])
def test_round_trip_explicit_layout(tmp_path, sparse_problem, layout) -> None:
    p = sparse_problem
    path = tmp_path / "p.npz"
    npz.write(p, path, layout=layout)
    loaded = npz.read(path)
    assert loaded == p
    assert loaded.dtype == p.dtype


def test_auto_picks_sparse_for_sparse_graph(tmp_path, sparse_problem) -> None:
    p = sparse_problem
    path = tmp_path / "p.npz"
    npz.write(p, path, layout="auto")
    with np.load(path) as data:
        assert str(data["layout"]) == "sparse"
    assert npz.read(path) == p


def test_auto_picks_dense_for_dense_graph(tmp_path, dense_problem) -> None:
    p = dense_problem
    path = tmp_path / "p.npz"
    npz.write(p, path, layout="auto")
    with np.load(path) as data:
        assert str(data["layout"]) == "dense"
    assert npz.read(path) == p


def test_rejects_unknown_layout(tmp_path, sparse_problem) -> None:
    p = sparse_problem
    with pytest.raises(ValueError):
        npz.write(p, tmp_path / "p.npz", layout="bogus")  # type: ignore


def test_write_refuses_an_existing_path_by_default(tmp_path, sparse_problem, dense_problem) -> None:
    path = tmp_path / "p.npz"
    npz.write(sparse_problem, path)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        npz.write(dense_problem, path)


def test_overwrite_true_replaces_the_file(tmp_path, sparse_problem, dense_problem) -> None:
    path = tmp_path / "p.npz"
    npz.write(sparse_problem, path)
    npz.write(dense_problem, path, overwrite=True)
    assert npz.read(path) == dense_problem


def test_write_never_appends_the_npz_suffix_on_its_own(tmp_path, sparse_problem) -> None:
    """Unlike bare np.savez_compressed: writes to exactly the given path, matching
    json and hdf5 -- no path given a suffix it wasn't handed."""
    path = tmp_path / "p"  # no suffix
    npz.write(sparse_problem, path)
    assert path.exists()
    assert not path.with_suffix(".npz").exists()
    assert npz.read(path) == sparse_problem


def test_file_records_its_format_version(tmp_path, sparse_problem: Problem) -> None:
    path = tmp_path / "p.npz"
    npz.write(sparse_problem, path)
    with np.load(path) as data:
        assert int(data["format_version"]) == 1


def test_read_rejects_a_newer_format_version(tmp_path, sparse_problem: Problem) -> None:
    path = tmp_path / "p.npz"
    npz.write(sparse_problem, path)
    with np.load(path) as data:
        fields = dict(data)
    fields["format_version"] = np.array(99)
    np.savez_compressed(path, **fields)
    with pytest.raises(ValueError, match="newer"):
        npz.read(path)


def test_reads_a_file_written_before_versioning(tmp_path, sparse_problem: Problem) -> None:
    """A file with no format_version holds the version-1 schema by definition."""
    path = tmp_path / "p.npz"
    npz.write(sparse_problem, path)
    with np.load(path) as data:
        fields = {k: v for k, v in data.items() if k != "format_version"}
    np.savez_compressed(path, **fields)
    assert npz.read(path) == sparse_problem
