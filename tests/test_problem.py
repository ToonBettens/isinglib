from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

from isinglib import Problem

# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------


def test_defaults_zero_bias_and_offset() -> None:
    p = Problem(j=np.zeros((3, 3)))
    assert p.n == 3
    assert np.array_equal(p.h, np.zeros(3))
    assert p.c == 0.0
    assert p.dtype == np.dtype(np.float64)


def test_constructor_copies_its_inputs() -> None:
    j = np.array([[0.0, 1.0], [1.0, 0.0]])
    h = np.array([1.0, 2.0])
    p = Problem(j=j, h=h)
    j[0, 1] = j[1, 0] = 99.0
    h[0] = 99.0
    assert p.j[0, 1] == 1.0
    assert p.h[0] == 1.0


def test_negative_zero_is_canonicalized() -> None:
    a = Problem(j=np.array([[0.0, 1.0], [1.0, 0.0]]), h=np.array([-0.0, 1.0]), c=-0.0)
    b = Problem(j=np.array([[-0.0, 1.0], [1.0, -0.0]]), h=np.array([0.0, 1.0]), c=0.0)
    assert a == b
    assert a.fingerprint == b.fingerprint
    assert hash(a) == hash(b)
    assert not np.signbit(a.h).any() and not np.signbit(a.j).any()


# ------------------------------------------------------------------
# Immutability
# ------------------------------------------------------------------


def test_frozen() -> None:
    p = Problem(j=np.zeros((2, 2)))
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.c = 5.0  # type: ignore


def test_stored_arrays_are_read_only() -> None:
    p = Problem(j=np.zeros((2, 2)), h=np.zeros(2))
    with pytest.raises(ValueError):
        p.j[0, 1] = 1.0
    with pytest.raises(ValueError):
        p.h[0] = 1.0


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "j",
    [
        np.array([[0.0, 1.0, 0.0]]),            # not square
        np.array([[0.0, 1.0], [2.0, 0.0]]),     # not symmetric
        np.array([[1.0, 0.0], [0.0, 0.0]]),     # nonzero diagonal
    ],
)
def test_validate_rejects_bad_coupling(j: np.ndarray) -> None:
    with pytest.raises(ValueError):
        Problem(j=j)


def test_bias_length_must_match() -> None:
    with pytest.raises(ValueError):
        Problem(j=np.zeros((2, 2)), h=np.zeros(3))


@pytest.mark.parametrize("h", [np.array([np.nan, 1.0]), np.array([np.inf, 1.0])])
def test_rejects_non_finite_bias(h: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite"):
        Problem(j=np.zeros((2, 2)), h=h)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_rejects_non_finite_coupling(bad: float) -> None:
    """Reported as non-finite, not as asymmetric: NaN != NaN fails symmetry first."""
    with pytest.raises(ValueError, match="finite"):
        Problem(j=np.array([[0.0, bad], [bad, 0.0]]))


def test_rejects_dtype_wider_than_float64() -> None:
    if np.dtype(np.longdouble).itemsize <= np.dtype(np.float64).itemsize:
        pytest.skip("longdouble is not wider than float64 on this platform")
    with pytest.raises(TypeError, match="wider"):
        Problem(j=np.zeros((2, 2)), dtype=np.longdouble)


# ------------------------------------------------------------------
# Properties
# ------------------------------------------------------------------


def test_n_reports_the_spin_count() -> None:
    assert Problem(j=np.zeros((7, 7))).n == 7


@pytest.mark.parametrize(
    ("j", "expected"),
    [
        (np.zeros((3, 3)), 0),
        (np.array([[0.0, 1.0], [1.0, 0.0]]), 1),
        (np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 2.0], [0.0, 2.0, 0.0]]), 2),
    ],
)
def test_num_interactions_counts_upper_triangle(j: np.ndarray, expected: int) -> None:
    assert Problem(j=j).num_interactions == expected


def test_dtype_is_derived_from_j() -> None:
    p = Problem(j=np.zeros((2, 2)), dtype=np.float32)
    assert p.dtype == p.j.dtype == np.float32
    assert "dtype" not in {f.name for f in dataclasses.fields(p)}


def test_fingerprint_ignores_dtype() -> None:
    j = np.array([[0.0, 1.0], [1.0, 0.0]])
    p1 = Problem(j=j)
    p2 = Problem(j=j, dtype=np.dtype(np.float32))
    p3 = Problem(j=2 * j)
    # Same content, different dtype → same fingerprint (and equal).
    assert p1.fingerprint == p2.fingerprint
    assert p1 == p2
    # Different content → different fingerprint (and not equal).
    assert p1.fingerprint != p3.fingerprint
    assert p1 != p3


# ------------------------------------------------------------------
# Public methods
# ------------------------------------------------------------------


def test_astype_returns_self_when_dtype_already_matches() -> None:
    p = Problem(j=np.zeros((2, 2)))
    assert p.astype(np.float64) is p


def test_astype_casts_and_preserves_content() -> None:
    p = Problem(j=np.array([[0.0, 0.5], [0.5, 0.0]]), h=np.array([1.0, 2.0]), c=3.0)
    q = p.astype(np.float32)
    assert q is not p
    assert q.dtype == np.float32
    assert q == p  # 0.5, 1.0, 2.0, 3.0 are all exactly representable


def test_energy_matches_explicit_formula() -> None:
    j = np.array([[0.0, 1.0], [1.0, 0.0]])
    h = np.array([0.5, -0.5])
    p = Problem(j=j, h=h, c=2.0)
    s = np.array([1.0, -1.0])
    # E = -0.5 sᵀ j s - hᵀ s + c = -0.5*(-2) - (0.5*1 + -0.5*-1) + 2 = 1 - 1 + 2 = 2
    assert p.energy(s) == pytest.approx(2.0)


def test_energy_accepts_any_array_like() -> None:
    p = Problem(j=np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert p.energy([1.0, -1.0]) == p.energy(np.array([1.0, -1.0]))


def test_replace_keeps_untouched_fields() -> None:
    p = Problem(j=np.array([[0.0, 1.0], [1.0, 0.0]]), h=np.array([1.0, 2.0]), c=3.0)
    q = p.replace(c=5.0)
    assert q.c == 5.0
    assert np.array_equal(q.j, p.j) and np.array_equal(q.h, p.h)


# ------------------------------------------------------------------
# Dunder methods
# ------------------------------------------------------------------


def test_hash_agrees_with_equality() -> None:
    j = np.array([[0.0, 1.0], [1.0, 0.0]])
    p1 = Problem(j=j, h=np.array([0.5, -0.5]), c=1.0)
    p2 = Problem(j=j, h=np.array([0.5, -0.5]), c=1.0)
    assert p1 == p2 and hash(p1) == hash(p2)
    assert {p1, p2} == {p1}
    assert {p1: "a"}[p2] == "a"


def test_eq_with_a_non_problem_is_false() -> None:
    p = Problem(j=np.zeros((2, 2)))
    assert p != 5
    assert p != "not a problem"


def test_copy_and_deepcopy_return_the_same_object() -> None:
    p = Problem(j=np.array([[0.0, 1.0], [1.0, 0.0]]), h=np.array([1.0, 2.0]), c=3.0)
    assert copy.copy(p) is p
    assert copy.deepcopy(p) is p
    assert not copy.deepcopy(p).j.flags.writeable


def test_repr_names_the_shape_and_fingerprint() -> None:
    p = Problem(j=np.zeros((3, 3)))
    text = repr(p)
    assert "n=3" in text
    assert p.fingerprint in text


# ------------------------------------------------------------------
# Input coercion
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expected_dtype"),
    [
        ({"j": [[0.0, 1.0], [1.0, 0.0]]}, np.float64),                    # nested lists
        ({"j": np.array([[0, 1], [1, 0]])}, np.float64),                  # int array
        ({"j": np.zeros((2, 2)), "h": [1.0, 2.0]}, np.float64),           # list bias
        ({"j": np.zeros((2, 2)), "c": 3}, np.float64),                    # int offset
        ({"j": np.zeros((2, 2)), "dtype": "float32"}, np.float32),        # dtype by name
    ],
)
def test_accepts_any_array_like_input(kwargs: dict, expected_dtype: type) -> None:
    p = Problem(**kwargs)
    assert p.n == 2
    assert p.dtype == expected_dtype
    assert isinstance(p.c, float)


@pytest.mark.parametrize("n", [0, 1])
def test_degenerate_sizes_are_valid(n: int) -> None:
    p = Problem(j=np.zeros((n, n)))
    assert p.n == n
    assert p.num_interactions == 0
    assert p.h.shape == (n,)
    assert p.energy(np.ones(n)) == 0.0


@pytest.mark.parametrize(
    "j",
    [
        "not a matrix",                       # not array-like at all
        [[0.0, 1.0], [1.0]],                  # ragged
        np.zeros((2, 2), dtype=complex),      # would silently lose the imaginary part
    ],
)
def test_rejects_uncoercible_input(j: object) -> None:
    with pytest.raises(TypeError):
        Problem(j=j)  # ty: ignore[invalid-argument-type]
