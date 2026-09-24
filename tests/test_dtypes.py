"""Coercion helpers in `isinglib.dtypes`."""

from __future__ import annotations

import numpy as np
import pytest

from isinglib.dtypes import DEFAULT_INDEX_DTYPE, ensure_index_array

# ------------------------------------------------------------------
# ensure_index_array: accepted input
# ------------------------------------------------------------------


def test_python_ints_use_the_default_dtype() -> None:
    out = ensure_index_array([0, 1, 2])
    assert out.dtype == DEFAULT_INDEX_DTYPE
    assert list(out) == [0, 1, 2]


def test_wider_integers_are_narrowed() -> None:
    out = ensure_index_array(np.array([0, 1], dtype=np.int64))
    assert out.dtype == DEFAULT_INDEX_DTYPE


def test_booleans_are_accepted() -> None:
    assert list(ensure_index_array([False, True])) == [0, 1]


def test_empty_input_is_accepted_whatever_its_dtype() -> None:
    """np.asarray([]) is float64 by default; there are no values to be wrong about."""
    out = ensure_index_array([])
    assert out.size == 0
    assert out.dtype == DEFAULT_INDEX_DTYPE


def test_explicit_dtype_is_honoured() -> None:
    assert ensure_index_array([1, 2], dtype=np.int64).dtype == np.int64


def test_copy_is_independent_of_its_input() -> None:
    src = np.array([1, 2], dtype=DEFAULT_INDEX_DTYPE)
    out = ensure_index_array(src, copy=True)
    src[0] = 99
    assert out[0] == 1


# ------------------------------------------------------------------
# ensure_index_array: rejected input
# ------------------------------------------------------------------


@pytest.mark.parametrize("arr", [[0.0, 1.0, 2.0], [0.9, 1.9], np.array([1.0 + 2.0j])])
def test_rejects_non_integer_input_without_guessing(arr: object) -> None:
    """No automatic float coercion, exact or not: the caller decides how a float
    array becomes indices, this does not guess on its behalf."""
    with pytest.raises(TypeError, match="pass an integer array"):
        ensure_index_array(arr)  # ty: ignore[invalid-argument-type]


def test_rejects_a_non_integer_target_dtype() -> None:
    with pytest.raises(TypeError, match="integer dtype"):
        ensure_index_array([1, 2], dtype=np.float64)
