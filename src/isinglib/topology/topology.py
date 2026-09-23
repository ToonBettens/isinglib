from __future__ import annotations

import numpy as np
import numpy.typing as npt

from isinglib.dtypes import DEFAULT_FLOAT_DTYPE, FloatArray
from isinglib.topology.fillers import FillerLike, resolve

__all__ = ("Topology",)

_NODE_DTYPE = np.int32


class Topology:
    """Undirected simple graph stored as a canonical edge list.

    Attributes:
        n: Number of nodes.
        src: Edge source nodes (src < dst).
        dst: Edge destination nodes (src < dst).
    """

    __slots__ = ("_csr_cache", "dst", "n", "src")

    def __init__(self, n: int, src: npt.ArrayLike, dst: npt.ArrayLike) -> None:
        if n < 0:
            raise ValueError("n must be non-negative.")
        self.n = int(n)

        s = np.asarray(src, dtype=_NODE_DTYPE)
        d = np.asarray(dst, dtype=_NODE_DTYPE)

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

        self.src: npt.NDArray[np.int32] = s
        self.dst: npt.NDArray[np.int32] = d
        self._csr_cache: tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]] | None = None

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
            return cls(0, np.empty(0, _NODE_DTYPE), np.empty(0, _NODE_DTYPE))
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

    def edges(self) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Return `(src, dst)` with `src[k] < dst[k]`."""
        return self.src, self.dst

    def degrees(self) -> npt.NDArray[np.int32]:
        """Return the degree of each node as an int32 array of length `n`."""
        deg = np.zeros(self.n, dtype=_NODE_DTYPE)
        if self.num_edges:
            np.add.at(deg, self.src, 1)
            np.add.at(deg, self.dst, 1)
        return deg

    def neighbors(self, u: int) -> npt.NDArray[np.int32]:
        """Return the sorted neighbor indices of node `u`."""
        if not (0 <= u < self.n):
            raise IndexError(f"Node {u} out of range [0, {self.n}).")
        indptr, indices = self._csr()
        return indices[indptr[u] : indptr[u + 1]]

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

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _csr(self) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Build and cache a CSR adjacency representation for O(degree) neighbor lookup."""
        if self._csr_cache is not None:
            return self._csr_cache

        if self.num_edges == 0:
            self._csr_cache = (np.zeros(self.n + 1, dtype=_NODE_DTYPE), np.empty(0, dtype=_NODE_DTYPE))
            return self._csr_cache

        u = np.concatenate((self.src, self.dst)).astype(np.int64)
        v = np.concatenate((self.dst, self.src)).astype(np.int64)
        order = np.lexsort((v, u))
        u, v = u[order], v[order]

        counts = np.bincount(u, minlength=self.n).astype(_NODE_DTYPE)
        indptr = np.empty(self.n + 1, dtype=_NODE_DTYPE)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])

        self._csr_cache = (indptr, v.astype(_NODE_DTYPE))
        return self._csr_cache
