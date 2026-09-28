from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

from isinglib._evaluate import energy
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
        h: (n,) bias vector (the external field).
        c: Scalar energy offset.
    """

    j: FloatArray
    h: FloatArray
    c: float

    def __init__(
        self,
        j: npt.ArrayLike,
        h: npt.ArrayLike = 0.0,
        c: float = 0.0,
        dtype: npt.DTypeLike | None = None,
    ) -> None:
        """Build an Ising problem instance.

        Args:
            j: Symmetric (n, n) coupling matrix with zero diagonal.
            h: Bias: a scalar for every spin, or an (n,) vector.
            c: Scalar energy offset.
            dtype: Target float dtype (default float64).
        """
        resolved = ensure_float_dtype(dtype)
        couplings = self._coerce_j(j, resolved)
        biases = self._coerce_h(h, couplings.shape[0], resolved)
        cte = self._coerce_c(c)
        object.__setattr__(self, "j", couplings)
        object.__setattr__(self, "h", biases)
        object.__setattr__(self, "c", cte)

    @staticmethod
    def _coerce_j(j: npt.ArrayLike, dtype: FloatDType) -> FloatArray:
        couplings = ensure_float_array(j, dtype=dtype, copy=True)
        couplings += 0.0  # Normalize -0.0 onto 0.0
        if couplings.ndim != 2 or couplings.shape[0] != couplings.shape[1]:
            raise ValueError(f"j must be a square 2-D array; got shape {couplings.shape}.")
        if not np.isfinite(couplings).all():
            raise ValueError("j must be finite; got NaN or infinity.")
        if not np.allclose(couplings, couplings.T, rtol=RTOL, atol=ATOL):
            raise ValueError("j must be symmetric.")
        if not np.allclose(np.diag(couplings), 0.0, rtol=RTOL, atol=ATOL):
            raise ValueError("j must have zero diagonal.")
        couplings.setflags(write=False)
        return couplings

    @staticmethod
    def _coerce_h(h: npt.ArrayLike, n: int, dtype: FloatDType) -> FloatArray:
        biases = ensure_float_array(h, dtype=dtype, copy=True)
        if biases.ndim == 0:
            biases = np.full(n, biases, dtype=dtype)
        biases += 0.0  # Normalize -0.0 onto 0.0
        if biases.shape != (n,):
            raise ValueError(f"h must be a scalar or 1-D with length {n}; got shape {biases.shape}.")
        if not np.isfinite(biases).all():
            raise ValueError("h must be finite; got NaN or infinity.")
        biases.setflags(write=False)
        return biases

    @staticmethod
    def _coerce_c(c: float) -> float:
        return float(c) + 0.0  # Normalize -0.0 onto 0.0

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
        return energy(self.j, self.h, self.c, np.asarray(s, dtype=self.dtype))

    def replace(
        self,
        *,
        j: npt.ArrayLike | None = None,
        h: npt.ArrayLike | None = None,
        c: float | None = None,
    ) -> Problem:
        """Return a problem with the given fields replaced, the rest kept as-is."""
        if j is not None:
            return Problem(j, self.h if h is None else h, self.c if c is None else c, self.dtype)
        if h is None and c is None:
            return self
        # Bypass __init__: the kept arrays are already validated and read-only.
        problem = object.__new__(Problem)
        object.__setattr__(problem, "j", self.j)
        object.__setattr__(problem, "h", self.h if h is None else self._coerce_h(h, self.n, self.dtype))
        object.__setattr__(problem, "c", self.c if c is None else self._coerce_c(c))
        return problem

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

