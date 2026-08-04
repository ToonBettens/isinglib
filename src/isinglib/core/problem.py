from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from dataclasses import replace as _dataclasses_replace
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from isinglib.core import evaluate
from isinglib.utils.defaults import ATOL, RTOL
from isinglib.utils.dtypes import (
    FloatArray,
    FloatDType,
    ensure_float_array,
    ensure_float_dtype,
    is_scalar_like,
)

if TYPE_CHECKING:
    from isinglib.topology.topology import FillerLike, Topology


__all__ = ("Problem",)


@dataclass(frozen=True, eq=False)
class Problem:
    """An Ising problem instance: `E(s) = -0.5 sᵀ j s - hᵀ s + c`.

    Attributes:
        j: Symmetric (n, n) coupling matrix with zero diagonal.
        h: (n,) bias vector.
        c: Scalar energy offset.
        dtype: Authoritative floating dtype for all numerical fields.
        meta: Arbitrary provenance: name, source, ground-truth energy, etc.
    """

    j: FloatArray
    h: FloatArray | None = None
    c: float = 0.0
    dtype: FloatDType | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        dtype = ensure_float_dtype(self.dtype)
        j = ensure_float_array(self.j, dtype=dtype, copy=True)
        h = (
            ensure_float_array(self.h, dtype=dtype, copy=True)
            if self.h is not None
            else np.zeros(j.shape[0], dtype=dtype)
        )
        j.setflags(write=False)
        h.setflags(write=False)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "j", j)
        object.__setattr__(self, "h", h)
        object.__setattr__(self, "c", float(self.c))
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))
        self.validate()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of spins."""
        return int(self.j.shape[0])

    @property
    def num_interactions(self) -> int:
        """Number of nonzero couplings (upper-triangle entries of `j`)."""
        return int(np.count_nonzero(np.triu(self.j, k=1)))

    @cached_property
    def id(self) -> str:
        """16-character hash of `(j, h, c)`, ignoring `dtype` and `meta`.

        A stable, hashable, serializable identity for this problem's content.
        Normalized to float64 before hashing, so equal-valued problems of
        different `dtype` get equal ids too.
        """
        h = self.h
        assert h is not None  # always set by __post_init__
        m = hashlib.sha1(usedforsecurity=False)
        m.update(np.ascontiguousarray(self.j, dtype=np.float64).tobytes())
        m.update(np.ascontiguousarray(h, dtype=np.float64).tobytes())
        m.update(np.float64(self.c).tobytes())
        return m.hexdigest()[:16]

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise if `j` or `h` violate the Ising problem invariants."""
        j, h = self.j, self.h
        assert h is not None  # always set by __post_init__
        if j.ndim != 2 or j.shape[0] != j.shape[1]:
            raise ValueError(f"j must be a square 2-D array; got shape {j.shape}.")
        if not np.allclose(j, j.T, rtol=RTOL, atol=ATOL):
            raise ValueError("j must be symmetric.")
        if not np.allclose(np.diag(j), 0.0, rtol=RTOL, atol=ATOL):
            raise ValueError("j must have zero diagonal.")
        if h.ndim != 1 or h.shape[0] != j.shape[0]:
            raise ValueError(f"h must be 1-D with length {j.shape[0]}; got shape {h.shape}.")

    def astype(self, dtype: np.dtype) -> Problem:
        """Return a copy cast to `dtype`. Returns `self` if already that dtype."""
        dtype = ensure_float_dtype(dtype)
        return self if dtype == self.dtype else self.replace(dtype=dtype)

    def energy(self, s: npt.ArrayLike) -> float:
        """Return `E(s) = -0.5 sᵀ j s - hᵀ s + c` for spin vector `s`."""
        sv = np.asarray(s, dtype=self.dtype)
        h = self.h
        assert h is not None  # always set by __post_init__
        return evaluate.energy(self.j, h, self.c, sv)

    def replace(self, **changes: object) -> Problem:
        """Return a copy with the given fields replaced, the rest are kept as-is."""
        return _dataclasses_replace(self, **changes)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    # Unhashable on purpose: use `id` as a dict/set key instead of the object itself
    __hash__ = None

    def __eq__(self, other: object) -> bool:
        """Compare by content only: `j`, `h`, `c`.

        `dtype` and `meta` are excluded: two problems built from the same
        numbers are the same problem regardless of what's recorded about them.
        """
        if not isinstance(other, Problem):
            return NotImplemented
        assert self.h is not None and other.h is not None  # added for type-cleanliness, guaranteed by __post_init__
        return (
            self.c == other.c
            and np.array_equal(self.j, other.j)
            and np.array_equal(self.h, other.h)
        )

    def __repr__(self) -> str:
        return (
            f"Problem(n={self.n}, dtype={self.dtype}, "
            f"c={self.c:g}, id={self.id!r})"
        )

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    @classmethod
    def from_topology(
        cls,
        topology: Topology,
        coupling: FillerLike,
        bias: FillerLike = 0.0,
        *,
        meta: dict | None = None,
        dtype: np.dtype | None = None,
    ) -> Problem:
        """Construct from a `Topology` and weight specifications.

        Args:
            topology: Graph structure defining which spins interact.
            coupling: Edge weights for `j`: scalar or `callable(n_edges) -> array`.
            bias: Node biases for `h`: scalar or `callable(n_nodes) -> array`.
            meta: Provenance metadata stored in `Problem.meta`.
            dtype: Target float dtype (default float64).
        """
        dtype = ensure_float_dtype(dtype)
        j = topology.fill(coupling).astype(dtype)

        if is_scalar_like(bias):
            h = np.full(topology.n, float(bias), dtype=dtype)  # type: ignore[arg-type]
        elif callable(bias):
            h = ensure_float_array(bias(topology.n), dtype=dtype)
        else:
            raise TypeError("bias must be a scalar or callable(n_nodes) -> array.")

        return cls(j=j, h=h, dtype=dtype, meta=meta or {})

    # ------------------------------------------------------------------
    # QUBO
    # ------------------------------------------------------------------

    @classmethod
    def from_qubo(
        cls,
        q: npt.ArrayLike,
        offset: float = 0.0,
        *,
        meta: dict | None = None,
        dtype: np.dtype | None = None,
    ) -> Problem:
        """Construct from QUBO form `E(x) = xᵀQx + offset`, with `x ∈ {0, 1}ⁿ`.

        Only the upper triangle of `q` (including the diagonal) is used.

        Args:
            q: (n, n) QUBO matrix.
            offset: Additive constant in the QUBO objective.
            meta: Provenance metadata stored in `Problem.meta`.
            dtype: Target float dtype (default float64).
        """
        dtype = ensure_float_dtype(dtype)
        q_arr = ensure_float_array(np.triu(np.asarray(q)), dtype=dtype)
        n = q_arr.shape[0]

        rows, cols = np.triu_indices(n, k=1)

        j = np.zeros((n, n), dtype=dtype)
        j[rows, cols] = -q_arr[rows, cols] / 4
        j[cols, rows] = j[rows, cols]

        h = j.sum(axis=1) - np.diag(q_arr) / 2
        c = float(offset) - float(h.sum()) + float(j[rows, cols].sum())

        return cls(j=j, h=h, c=c, dtype=dtype, meta=meta or {})

    def to_qubo(self) -> tuple[FloatArray, float]:
        """Convert to QUBO form `E(x) = xᵀQx + offset`, with `x ∈ {0, 1}ⁿ`.

        The substitution `s_i = 2x_i - 1` maps every Ising spin to a binary
        variable. The returned `Q` is upper triangular.

        Returns:
            q: Upper-triangular (n, n) QUBO matrix.
            offset: Additive constant such that `E_ising(s) = E_qubo(x) + offset`
                when `s_i = 2x_i - 1`.
        """
        n = self.n
        h = self.h
        assert h is not None  # always set by __post_init__
        rows, cols = np.triu_indices(n, k=1)

        q = np.zeros((n, n), dtype=self.dtype)
        q[rows, cols] = -4 * self.j[rows, cols]
        q[np.arange(n), np.arange(n)] = 2 * self.j.sum(axis=1) - 2 * h

        offset = float(self.c + h.sum() - self.j[rows, cols].sum())
        return q, offset
