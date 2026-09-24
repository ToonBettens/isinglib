from __future__ import annotations

from typing import Any, TypeIs

import numpy as np
import numpy.typing as npt

__all__ = (
    "DEFAULT_FLOAT_DTYPE",
    "DEFAULT_INDEX_DTYPE",
    "MAX_FLOAT_DTYPE",
    "FloatArray",
    "FloatDType",
    "IndexArray",
    "ScalarLike",
    "ensure_float_array",
    "ensure_float_dtype",
    "ensure_index_array",
    "is_scalar_like",
)

MAX_FLOAT_DTYPE = np.dtype(np.float64)  # Widest dtype the library supports (fixed).

DEFAULT_FLOAT_DTYPE = np.dtype(np.float64)  # Authoritative default floating dtype.
DEFAULT_INDEX_DTYPE = np.dtype(np.int32)  # Authoritative dtype for indices.


# Type aliases.
type ScalarLike = int | float | np.floating
type FloatDType = np.dtype[np.floating]
type FloatArray = npt.NDArray[np.floating]
type IndexArray = npt.NDArray[np.integer]


def is_scalar_like(x: Any) -> TypeIs[ScalarLike]:
    """Return True if `x` is a real-number scalar (and not a bool)."""
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
        dtype: Requested dtype, or None to use `default`.
        default: Fallback dtype when `dtype` is None.
    """
    resolved = np.dtype(default if dtype is None else dtype)
    if not np.issubdtype(resolved, np.floating):
        raise TypeError(f"dtype {dtype!r} (resolved {resolved}) must be a floating dtype.")
    if resolved.itemsize > MAX_FLOAT_DTYPE.itemsize:
        raise TypeError(f"dtype {dtype!r} (resolved {resolved}) is wider than {MAX_FLOAT_DTYPE}.")
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
        copy: If True, guarantee output is memory-independent from `arr`.
    """
    resolved = ensure_float_dtype(dtype)
    try:
        if np.iscomplexobj(arr):
            raise TypeError(f"Cannot coerce complex input to {resolved}; numpy would silently drop the imaginary part.")
        if copy:
            return np.array(arr, dtype=resolved, copy=True)
        return np.asarray(arr, dtype=resolved)
    except TypeError:
        raise
    except Exception as e:
        raise TypeError(f"Cannot coerce {type(arr).__name__} to dtype {resolved}.") from e


def ensure_index_array(
    arr: npt.ArrayLike,
    dtype: npt.DTypeLike | None = None,
    *,
    copy: bool = False,
) -> IndexArray:
    """Coerce an array-like to an integer ndarray of the resolved dtype.

    Args:
        arr: Input array-like.
        dtype: Target dtype, or None to use `DEFAULT_INDEX_DTYPE`.
        copy: If True, guarantee output is memory-independent from `arr`.
    """
    resolved = np.dtype(DEFAULT_INDEX_DTYPE if dtype is None else dtype)
    if not np.issubdtype(resolved, np.integer):
        raise TypeError(f"dtype {dtype!r} (resolved {resolved}) must be an integer dtype.")
    try:
        probe = np.asarray(arr)
    except Exception as e:
        raise TypeError(f"Cannot coerce {type(arr).__name__} to dtype {resolved}.") from e
    if probe.size and not (np.issubdtype(probe.dtype, np.integer) or probe.dtype == np.bool_):
        raise TypeError(f"Cannot coerce {probe.dtype} input to {resolved}; pass an integer array.")
    return np.array(probe, dtype=resolved, copy=True) if copy else probe.astype(resolved, copy=False)
