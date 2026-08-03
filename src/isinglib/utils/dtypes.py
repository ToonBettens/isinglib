from __future__ import annotations

from typing import Any, TypeIs

import numpy as np
import numpy.typing as npt

from isinglib.utils.defaults import DEFAULT_FLOAT_DTYPE

__all__ = (
    "FloatArray",
    "FloatDType",
    "ScalarLike",
    "ensure_float_array",
    "ensure_float_dtype",
    "is_scalar_like",
)


# Type aliases. These document intent at zero runtime cost.
# Scalars in this library are plain Python ``float``.
# Spin/continuous states are plain ``np.ndarray``.
type FloatArray = npt.NDArray[np.floating]
type FloatDType = np.dtype[np.floating]
type ScalarLike = int | float | np.floating


def is_scalar_like(x: Any) -> TypeIs[ScalarLike]:
    """Return True if ``x`` is a real-number scalar (and not a bool)."""
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float, np.floating)):
        return True
    return isinstance(x, np.ndarray) and x.ndim == 0 and np.issubdtype(x.dtype, np.floating)


def ensure_float_dtype(
    dtype: npt.DTypeLike | None,
    default: npt.DTypeLike = DEFAULT_FLOAT_DTYPE,
) -> FloatDType:
    """Resolve a dtype candidate to a concrete floating dtype.

    Args:
        dtype: Requested dtype, or None to use ``default``.
        default: Fallback dtype when ``dtype`` is None.

    Returns:
        A floating ``np.dtype``.

    Raises:
        TypeError: If the resolved dtype is not a floating type.
    """
    resolved = np.dtype(default if dtype is None else dtype)
    if not np.issubdtype(resolved, np.floating):
        raise TypeError(f"dtype {dtype!r} (resolved {resolved}) must be a floating dtype.")
    return resolved


def ensure_float_array(
    arr: npt.ArrayLike,
    dtype: npt.DTypeLike | None = None,
    *,
    copy: bool = False,
) -> FloatArray:
    """Coerce an array-like to a floating ndarray of the resolved dtype.

    Args:
        arr: Input array-like.
        dtype: Target dtype, or None to use the default.
        copy: If True, always return an array that doesn't alias ``arr``'s
            memory — plain ``np.asarray`` returns ``arr`` itself unchanged
            when it already has the resolved dtype. Trust boundaries that
            need to guarantee independence from caller-owned memory (e.g.
            before freezing an array read-only) should pass ``copy=True``.
    """
    resolved = ensure_float_dtype(dtype)
    try:
        if copy:
            return np.array(arr, dtype=resolved, copy=True)
        return np.asarray(arr, dtype=resolved)
    except Exception as e:
        raise TypeError(f"Cannot coerce {type(arr).__name__} to dtype {resolved}.") from e
