"""The `io.read` / `io.write` suffix dispatchers."""

from __future__ import annotations

import pytest

from isinglib import Problem, io

# ------------------------------------------------------------------
# Suffix dispatch
# ------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".json", ".npz", ".h5", ".hdf5"])
def test_round_trip_through_every_suffix(tmp_path, sparse_problem: Problem, suffix: str) -> None:
    if suffix in (".h5", ".hdf5"):
        pytest.importorskip("h5py")
    path = tmp_path / f"p{suffix}"
    io.write(sparse_problem, path)
    assert io.read(path) == sparse_problem


def test_suffix_is_case_insensitive(tmp_path, sparse_problem: Problem) -> None:
    path = tmp_path / "p.JSON"
    io.write(sparse_problem, path)
    assert io.read(path) == sparse_problem


def test_format_overrides_the_suffix(tmp_path, sparse_problem: Problem) -> None:
    path = tmp_path / "p.weird"
    io.write(sparse_problem, path, format="json")
    assert io.read(path, format="json") == sparse_problem


@pytest.mark.parametrize("name", ["p.txt", "p"])
def test_unknown_suffix_is_rejected(tmp_path, sparse_problem: Problem, name: str) -> None:
    with pytest.raises(ValueError, match="format="):
        io.write(sparse_problem, tmp_path / name)


def test_unknown_format_is_rejected(tmp_path, sparse_problem: Problem) -> None:
    with pytest.raises(ValueError, match="Unknown format"):
        io.write(sparse_problem, tmp_path / "p.json", format="bogus")  # ty: ignore[invalid-argument-type]


# ------------------------------------------------------------------
# kwargs passthrough
# ------------------------------------------------------------------


def test_forwards_backend_options(tmp_path, sparse_problem: Problem) -> None:
    pytest.importorskip("h5py")
    path = tmp_path / "runs.h5"
    io.write(sparse_problem, path, key="run1", compression=None)
    assert io.read(path, key="run1") == sparse_problem


def test_forwards_layout_to_every_backend(tmp_path, dense_problem: Problem) -> None:
    for suffix in (".json", ".npz"):
        path = tmp_path / f"p{suffix}"
        io.write(dense_problem, path, layout="sparse")
        assert io.read(path) == dense_problem


def test_a_wrong_option_names_both_backend_and_option(tmp_path, sparse_problem: Problem) -> None:
    """Accepted cost of **kwargs: this is a runtime error, but a legible one."""
    with pytest.raises(TypeError, match="key"):
        io.write(sparse_problem, tmp_path / "p.json", key="run1")
