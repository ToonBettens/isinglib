from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from isinglib.dtypes import (
    DEFAULT_INDEX_DTYPE,
    FloatArray,
    IndexArray,
    ensure_index_array,
)
from isinglib.problem import Problem

__all__ = (
    "FORMAT_VERSION",
    "CooTriples",
    "Layout",
    "LayoutOptions",
    "Record",
    "check_version",
    "from_record",
    "to_record",
)

FORMAT_VERSION = 1
SPARSITY_THRESHOLD = 0.10

type CooTriples = tuple[IndexArray, IndexArray, FloatArray]  # (row, col, val)
type Layout = Literal["sparse", "dense"]
type LayoutOptions = Literal["auto", "sparse", "dense"]


@dataclass(frozen=True)
class Record:
    """One problem as it sits on disk."""

    dtype: str
    n: int
    c: float
    h: FloatArray
    j: FloatArray | CooTriples
    version: int = FORMAT_VERSION

    @property
    def layout(self) -> Layout:
        return "sparse" if isinstance(self.j, tuple) else "dense"


def check_version(version: int | None) -> None:
    """Raise if a file's `format_version` is one this library cannot read.

    Args:
        version: The stored version.
    """
    if version is None:
        return
    if version > FORMAT_VERSION:
        raise ValueError(
            f"File format_version {version} is newer than this isinglib supports "
            f"({FORMAT_VERSION}); upgrade isinglib to read it."
        )
    if version < 1:
        raise ValueError( f"Invalid format_version {version!r}; expected a positive integer.")


def choose_layout(problem: Problem) -> Layout:
    """Pick a layout from `j`'s upper-triangle nonzero density."""
    max_edges = problem.n * (problem.n - 1) // 2
    density = problem.num_interactions / max_edges if max_edges else 0.0
    return "sparse" if density < SPARSITY_THRESHOLD else "dense"


def to_record(problem: Problem, layout: LayoutOptions = "auto") -> Record:
    """Convert a Problem into the record a backend stores.

    Args:
        problem: The Problem to serialise.
        layout: `"sparse"`, `"dense"` or `"auto"`.
    """
    if layout == "auto":
        layout = choose_layout(problem)
    if layout not in ("sparse", "dense"):
        raise ValueError( f"Unknown layout {layout!r}; expected 'auto', 'sparse', or 'dense'.")

    return Record(
        j=_to_edges(problem) if layout == "sparse" else problem.j,
        dtype=str(problem.dtype),
        n=problem.n,
        c=problem.c,
        h=problem.h,
    )


def from_record(record: Record) -> Problem:
    """Reconstruct a Problem from a stored record, faithfully and without casting."""
    check_version(record.version)
    dtype = np.dtype(record.dtype)
    if isinstance(record.j, tuple):
        src, dst, weight = record.j
        j = _from_edges(record.n, src, dst, weight, dtype)
    else:
        j = record.j
    return Problem(j, record.h, record.c, dtype)


def _to_edges(problem: Problem) -> CooTriples:
    """Return `(src, dst, weight)` for the nonzero upper-triangle entries of `j`."""
    rows, cols = np.nonzero(problem.j)
    upper = rows < cols
    src, dst = (
        rows[upper].astype(DEFAULT_INDEX_DTYPE),
        cols[upper].astype(DEFAULT_INDEX_DTYPE),
    )
    return src, dst, problem.j[src, dst]


def _from_edges(
    n: int,
    src: npt.ArrayLike,
    dst: npt.ArrayLike,
    weight: npt.ArrayLike,
    dtype: npt.DTypeLike,
) -> FloatArray:
    """Rebuild a dense symmetric, zero-diagonal `(n, n)` matrix from edges."""
    src_idx = ensure_index_array(src)
    dst_idx = ensure_index_array(dst)
    j = np.zeros((n, n), dtype=dtype)
    j[src_idx, dst_idx] = weight
    j[dst_idx, src_idx] = weight
    return j
