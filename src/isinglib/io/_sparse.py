from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

from isinglib.problem import Problem

_SPARSITY_THRESHOLD = 0.10

__all__ = ("choose_encoding", "from_edges", "to_edges")


def choose_encoding(problem: Problem) -> Literal["sparse", "dense"]:
    """Return "sparse" or "dense" based on `j`'s upper-triangle nonzero density.

    Sparse storage only pays off once the graph actually is sparse — its
    per-edge index overhead outweighs the zeros it skips on near-complete
    graphs.
    """
    n = problem.n
    max_edges = n * (n - 1) // 2
    density = problem.num_interactions / max_edges if max_edges else 0.0
    return "sparse" if density < _SPARSITY_THRESHOLD else "dense"


def to_edges(
    problem: Problem,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.floating]]:
    """Return `(src, dst, weight)` for the nonzero upper-triangle entries of `j`."""
    rows, cols = np.triu_indices(problem.n, k=1)
    nonzero = problem.j[rows, cols] != 0.0
    src, dst = rows[nonzero].astype(np.int32), cols[nonzero].astype(np.int32)
    weight = problem.j[src, dst]
    return src, dst, weight


def from_edges(
    n: int,
    src: npt.ArrayLike,
    dst: npt.ArrayLike,
    weight: npt.ArrayLike,
    dtype: npt.DTypeLike,
) -> npt.NDArray[np.floating]:
    """Reconstruct a dense symmetric, zero-diagonal `(n, n)` matrix from edges."""
    src_idx = np.asarray(src, dtype=np.int32)
    dst_idx = np.asarray(dst, dtype=np.int32)
    j = np.zeros((n, n), dtype=dtype)
    j[src_idx, dst_idx] = weight
    j[dst_idx, src_idx] = weight
    return j
