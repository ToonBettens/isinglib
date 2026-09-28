"""`Solution` as a plain attribute bag: nothing about its shape is enforced."""

from __future__ import annotations

import numpy as np

from isinglib import Solution

# ------------------------------------------------------------------
# It is genuinely just a namespace
# ------------------------------------------------------------------


def test_fields_are_whatever_you_pass() -> None:
    sol = Solution(spins=np.array([1.0, -1.0]), energy=-2.0, anything_at_all=42)
    assert np.array_equal(sol.spins, [1.0, -1.0])
    assert sol.energy == -2.0
    assert sol.anything_at_all == 42


def test_no_field_is_required() -> None:
    sol = Solution()
    assert vars(sol) == {}


def test_is_an_ordinary_mutable_object() -> None:
    sol = Solution(energy=0.0)
    sol.energy = 5.0
    sol.new_field = "anything"
    assert sol.energy == 5.0
    assert sol.new_field == "anything"


def test_an_unset_field_is_absent_not_none() -> None:
    """The point of not enforcing a shape: `keep_trajectory=False` means no
    attribute at all, not a placeholder value."""
    sol = Solution(energy=0.0)
    assert not hasattr(sol, "trajectory")


# ------------------------------------------------------------------
# Equality: identity, not content
# ------------------------------------------------------------------


def test_equal_content_does_not_compare_equal() -> None:
    """A bare SimpleNamespace compares by __dict__ content, which would crash on
    an array field the same way a dataclass's auto-eq would. Two runs landing on
    the same state are still two different runs, so identity is the right
    notion anyway -- this is a deliberate override, not an oversight."""
    a = Solution(spins=np.array([1.0, -1.0]), energy=-2.0)
    b = Solution(spins=np.array([1.0, -1.0]), energy=-2.0)
    assert a == a
    assert a != b


def test_is_hashable() -> None:
    sol = Solution(energy=0.0)
    assert len({sol}) == 1
