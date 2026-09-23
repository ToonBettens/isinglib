from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

from isinglib import _evaluate
from isinglib.dtypes import (
    MAX_FLOAT_DTYPE,
    FloatArray,
    FloatDType,
    ensure_float_array,
    ensure_float_dtype,
)

__all__ = ("Problem",)

RTOL = 1e-12
ATOL = 1e-12


@dataclass(frozen=True, eq=False, init=False)
class Problem:
    """An Ising problem instance: `E(s) = -0.5 sᵀ j s - hᵀ s + c`.

    Instances are immutable and compare by content.

    Attributes:
        j: Symmetric (n, n) coupling matrix with zero diagonal.
        h: (n,) bias vector.
        c: Scalar energy offset.
    """

    j: FloatArray
    h: FloatArray
    c: float

    def __init__(
        self,
        j: npt.ArrayLike,
        h: npt.ArrayLike | None = None,
        c: float = 0.0,
        dtype: npt.DTypeLike | None = None,
    ) -> None:
        """Build an Ising problem instance.

        Args:
            j: Symmetric (n, n) coupling matrix with zero diagonal.
            h: (n,) bias vector; zeros if not given.
            c: Scalar energy offset.
            dtype: Target float dtype (default float64).
        """
        resolved = ensure_float_dtype(dtype)
        couplings = ensure_float_array(j, dtype=resolved, copy=True)
        biases = (
            ensure_float_array(h, dtype=resolved, copy=True)
            if h is not None
            else ensure_float_array(np.zeros(couplings.shape[:1]), dtype=resolved)
        )
        offset = float(c)

        # Normalize -0.0 onto 0.0
        couplings += 0.0
        biases += 0.0
        offset += 0.0

        couplings.setflags(write=False)
        biases.setflags(write=False)
        object.__setattr__(self, "j", couplings)
        object.__setattr__(self, "h", biases)
        object.__setattr__(self, "c", offset)
        self._validate()

    def _validate(self) -> None:
        """Raise if `j` or `h` violate the Ising problem invariants."""
        j, h = self.j, self.h
        if j.ndim != 2 or j.shape[0] != j.shape[1]:
            raise ValueError(f"j must be a square 2-D array; got shape {j.shape}.")
        if not np.allclose(j, j.T, rtol=RTOL, atol=ATOL):
            raise ValueError("j must be symmetric.")
        if not np.allclose(np.diag(j), 0.0, rtol=RTOL, atol=ATOL):
            raise ValueError("j must have zero diagonal.")
        if h.ndim != 1 or h.shape[0] != j.shape[0]:
            raise ValueError(f"h must be 1-D with length {j.shape[0]}; got shape {h.shape}.")
        if not np.isfinite(j).all():
            raise ValueError("j must be finite; got NaN or infinity.")
        if not np.isfinite(h).all():
            raise ValueError("h must be finite; got NaN or infinity.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of spins."""
        return int(self.j.shape[0])

    @property
    def num_interactions(self) -> int:
        """Number of nonzero couplings."""
        return int(np.count_nonzero(self.j) // 2)

    @property
    def dtype(self) -> FloatDType:
        """Authoritative floating dtype for this problem (dtype of `j`)."""
        return self.j.dtype

    @cached_property
    def fingerprint(self) -> str:
        """16-character content digest of `(j, h, c)`, ignoring dtype."""
        m = hashlib.sha1(usedforsecurity=False)
        m.update(np.ascontiguousarray(self.j, dtype=MAX_FLOAT_DTYPE))
        m.update(np.ascontiguousarray(self.h, dtype=MAX_FLOAT_DTYPE))
        m.update(np.array(self.c, dtype=MAX_FLOAT_DTYPE))
        return m.hexdigest()[:16]

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def astype(self, dtype: npt.DTypeLike) -> Problem:
        """Return a copy cast to `dtype`. Returns `self` if already that dtype."""
        resolved = ensure_float_dtype(dtype)
        return self if resolved == self.dtype else Problem(self.j, self.h, self.c, resolved)

    def energy(self, s: npt.ArrayLike) -> float:
        """Return `E(s) = -0.5 sᵀ j s - hᵀ s + c` for spin vector `s`."""
        return _evaluate.energy(self.j, self.h, self.c, np.asarray(s, dtype=self.dtype))

    def replace(
        self,
        *,
        j: npt.ArrayLike | None = None,
        h: npt.ArrayLike | None = None,
        c: float | None = None,
    ) -> Problem:
        """Return a copy with the given fields replaced, the rest kept as-is."""
        return Problem(
            self.j if j is None else j,
            self.h if h is None else h,
            self.c if c is None else c,
            self.dtype,
        )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __hash__(self) -> int:
        """Hash the content digest."""
        return hash(self.fingerprint)

    def __eq__(self, other: object) -> bool:
        """Compare by content only: `j`, `h`, `c`, ignoring dtype. """
        if not isinstance(other, Problem):
            return NotImplemented
        return (
            self.c == other.c
            and np.array_equal(self.j, other.j)
            and np.array_equal(self.h, other.h)
        )

    def __copy__(self) -> Problem:
        """Immutable, so a copy is the object itself."""
        return self

    def __deepcopy__(self, _memo: dict[int, object]) -> Problem:
        """Immutable, so a deep copy is the object itself."""
        return self

    def __repr__(self) -> str:
        return f"Problem(n={self.n}, dtype={self.dtype}, c={self.c:g}, fingerprint={self.fingerprint!r})"
