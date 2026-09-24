from __future__ import annotations

import numpy as np
import numpy.typing as npt

from isinglib.dtypes import (
    DEFAULT_FLOAT_DTYPE,
    DEFAULT_INDEX_DTYPE,
    FloatArray,
    IndexArray,
    ensure_index_array,
)
from isinglib.topology.fillers import FillerLike, resolve

__all__ = ("Topology",)


class Topology:
    """Undirected simple graph stored as a canonical edge list.

    Attributes:
        n: Number of nodes.
        src: Edge source nodes (src < dst).
        dst: Edge destination nodes (src < dst).
    """

    __slots__ = ("dst", "n", "src")

    def __init__(self, n: int, src: npt.ArrayLike, dst: npt.ArrayLike) -> None:
        if n < 0:
            raise ValueError("n must be non-negative.")
        self.n = int(n)

        s = ensure_index_array(src)
        d = ensure_index_array(dst)

        if s.ndim != 1 or d.ndim != 1 or s.size != d.size:
            raise ValueError("src and dst must be 1-D arrays of equal length.")

        if s.size:
            if np.any(s == d):
                raise ValueError("Self-loops are not allowed.")
            if np.any(s < 0) or np.any(d < 0) or np.any(s >= self.n) or np.any(d >= self.n):
                raise ValueError("Edge indices out of range [0, n).")

            swap = s > d
            if np.any(swap):
                s, d = s.copy(), d.copy()
                s[swap], d[swap] = d[swap], s[swap]

            order = np.lexsort((d, s))
            s, d = s[order], d[order]

            dupe = (np.diff(s, prepend=-1) == 0) & (np.diff(d, prepend=-1) == 0)
            if np.any(dupe):
                keep = ~dupe
                s, d = s[keep], d[keep]

        self.src: IndexArray = s
        self.dst: IndexArray = d

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_edges(cls, n: int, src: npt.ArrayLike, dst: npt.ArrayLike) -> Topology:
        """Construct from an explicit edge list."""
        return cls(n=n, src=src, dst=dst)

    @classmethod
    def from_adjacency(cls, adj: npt.ArrayLike) -> Topology:
        """Construct from a dense adjacency matrix (nonzero upper triangle = edge)."""
        arr = np.asarray(adj)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("Adjacency must be a square 2-D array.")
        n = int(arr.shape[0])
        if n == 0:
            return cls(0, np.empty(0, DEFAULT_INDEX_DTYPE), np.empty(0, DEFAULT_INDEX_DTYPE))
        iu, ju = np.triu_indices(n, k=1)
        mask = arr[iu, ju] != 0
        return cls(n=n, src=iu[mask], dst=ju[mask])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return self.n

    @property
    def num_edges(self) -> int:
        return int(self.src.size)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def edges(self) -> tuple[IndexArray, IndexArray]:
        """Return `(src, dst)` with `src[k] < dst[k]`."""
        return self.src, self.dst

    def degrees(self) -> IndexArray:
        """Return the degree of each node as an int32 array of length `n`."""
        deg = np.zeros(self.n, dtype=DEFAULT_INDEX_DTYPE)
        if self.num_edges:
            np.add.at(deg, self.src, 1)
            np.add.at(deg, self.dst, 1)
        return deg

    def fill(self, filler: FillerLike) -> FloatArray:
        """Build a dense (n, n) coupling matrix from a weight specification.

        Args:
            filler: A weight specification (see fillers.py).

        Returns:
            Symmetric (n, n) array of dtype `DEFAULT_FLOAT_DTYPE` with zero diagonal.
        """
        s, d = self.edges()
        values = resolve(filler, self.num_edges)

        mat = np.zeros((self.n, self.n), dtype=DEFAULT_FLOAT_DTYPE)
        if self.num_edges:
            mat[s, d] = values
            mat[d, s] = values
        return mat

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Topology(n={self.n}, edges={self.num_edges})"
