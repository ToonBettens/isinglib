from __future__ import annotations

import json as json_lib

import numpy as np
import pytest

from isinglib import Problem
from isinglib.io import json

# ------------------------------------------------------------------
# Round trip
# ------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["sparse", "dense"])
def test_round_trip_explicit_layout(tmp_path, sparse_problem: Problem, layout: str) -> None:
    path = tmp_path / "p.json"
    json.write(sparse_problem, path, layout=layout)  # ty: ignore[invalid-argument-type]
    loaded = json.read(path)
    assert loaded == sparse_problem
    assert loaded.dtype == sparse_problem.dtype


def test_write_actually_writes(tmp_path, sparse_problem: Problem) -> None:
    """Regression: `Path.open()` defaults to mode 'r', so json.dump used to raise."""
    path = tmp_path / "p.json"
    json.write(sparse_problem, path)
    assert path.stat().st_size > 0


def test_round_trip_preserves_float32(tmp_path) -> None:
    p = Problem(np.array([[0.0, 0.5], [0.5, 0.0]]), h=np.array([1.0, 2.0]), dtype=np.float32)
    path = tmp_path / "p.json"
    json.write(p, path)
    loaded = json.read(path)
    assert loaded.dtype == np.float32
    assert loaded == p


# ------------------------------------------------------------------
# Overwrite protection
# ------------------------------------------------------------------


def test_write_refuses_an_existing_path_by_default(tmp_path, sparse_problem: Problem) -> None:
    path = tmp_path / "p.json"
    json.write(sparse_problem, path)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        json.write(sparse_problem, path)


def test_overwrite_true_replaces_the_file(tmp_path, sparse_problem: Problem, dense_problem: Problem) -> None:
    path = tmp_path / "p.json"
    json.write(sparse_problem, path)
    json.write(dense_problem, path, overwrite=True)
    assert json.read(path) == dense_problem


# ------------------------------------------------------------------
# Layout selection
# ------------------------------------------------------------------


def test_auto_picks_sparse_for_sparse_graph(tmp_path, sparse_problem: Problem) -> None:
    path = tmp_path / "p.json"
    json.write(sparse_problem, path)
    data = json_lib.loads(path.read_text())
    assert data["layout"] == "sparse"
    assert json.read(path) == sparse_problem


def test_auto_picks_dense_for_dense_graph(tmp_path, dense_problem: Problem) -> None:
    path = tmp_path / "p.json"
    json.write(dense_problem, path)
    data = json_lib.loads(path.read_text())
    assert data["layout"] == "dense"
    assert json.read(path) == dense_problem


def test_rejects_unknown_layout(tmp_path, sparse_problem: Problem) -> None:
    with pytest.raises(ValueError, match="layout"):
        json.write(sparse_problem, tmp_path / "p.json", layout="bogus")  # ty: ignore[invalid-argument-type]


# ------------------------------------------------------------------
# On-disk shape
# ------------------------------------------------------------------


def test_sparse_file_stores_edge_triples(tmp_path) -> None:
    """The point of json: a sparse graph reads as edges, not a grid of zeros."""
    p = Problem(np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    path = tmp_path / "p.json"
    json.write(p, path, layout="sparse")
    data = json_lib.loads(path.read_text())
    assert data["edges"] == [[0, 1, 0.5]]
    assert "j" not in data


@pytest.mark.parametrize("indent", [2, 4])
def test_each_edge_is_one_line(tmp_path, sparse_problem: Problem, indent: int) -> None:
    """The reason to store edges at all: one edge per line, not one number per line."""
    path = tmp_path / "p.json"
    json.write(sparse_problem, path, layout="sparse", indent=indent)
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    edges = [ln for ln in lines if ln.startswith("[") and ln.rstrip(",").endswith("]")]
    assert len(edges) == sparse_problem.num_interactions
    assert json.read(path) == sparse_problem


def test_file_records_its_format_version(tmp_path, sparse_problem: Problem) -> None:
    path = tmp_path / "p.json"
    json.write(sparse_problem, path)
    assert json_lib.loads(path.read_text())["format_version"] == 1


def test_read_rejects_a_newer_format_version(tmp_path, sparse_problem: Problem) -> None:
    path = tmp_path / "p.json"
    json.write(sparse_problem, path)
    data = json_lib.loads(path.read_text())
    data["format_version"] = 99
    path.write_text(json_lib.dumps(data))
    with pytest.raises(ValueError, match="newer"):
        json.read(path)
