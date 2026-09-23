from __future__ import annotations

import numpy as np
import pytest

from isinglib import Problem
from isinglib.topology import complete, fillers

# ------------------------------------------------------------------
# resolve
# ------------------------------------------------------------------


def test_resolve_scalar() -> None:
    assert np.array_equal(fillers.resolve(2.5, 4), np.full(4, 2.5))


def test_resolve_callable() -> None:
    assert np.array_equal(fillers.resolve(lambda n: np.arange(n), 4), [0.0, 1.0, 2.0, 3.0])


def test_resolve_respects_dtype() -> None:
    assert fillers.resolve(1.0, 3, dtype=np.float32).dtype == np.float32


def test_resolve_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        fillers.resolve(lambda n: np.zeros(n + 1), 4)


def test_resolve_rejects_non_filler() -> None:
    with pytest.raises(TypeError):
        fillers.resolve("nope", 4)  # ty: ignore[invalid-argument-type]


# ------------------------------------------------------------------
# distributions
# ------------------------------------------------------------------


def test_constant() -> None:
    assert np.array_equal(fillers.constant(3.0)(5), np.full(5, 3.0))


def test_uniform_within_bounds() -> None:
    values = fillers.uniform(0.1, 0.9, rng=0)(100)
    assert values.shape == (100,)
    assert np.all((values >= 0.1) & (values < 0.9))


def test_gaussian_shape_and_seeding() -> None:
    a = fillers.gaussian(2.0, rng=0)(50)
    b = fillers.gaussian(2.0, rng=0)(50)
    assert a.shape == (50,)
    assert np.array_equal(a, b)


def test_filler_draws_fresh_values_on_each_call() -> None:
    """The generator is seeded once at construction, so reuse must not replay."""
    filler = fillers.gaussian(rng=0)
    assert not np.array_equal(filler(20), filler(20))


def test_pm_one_values() -> None:
    assert set(np.unique(fillers.pm_one(rng=0)(100))) <= {-1.0, 1.0}
    assert np.all(fillers.pm_one(p=1.0, rng=0)(10) == 1.0)
    assert np.all(fillers.pm_one(p=0.0, rng=0)(10) == -1.0)


def test_choice_draws_from_the_given_set() -> None:
    values = fillers.choice([-1.0, 0.0, 1.0], rng=0)(100)
    assert set(np.unique(values)) <= {-1.0, 0.0, 1.0}


@pytest.mark.parametrize(
    "make",
    [
        lambda: fillers.uniform(1.0, 0.0),
        lambda: fillers.gaussian(-1.0),
        lambda: fillers.pm_one(1.5),
        lambda: fillers.choice([]),
    ],
)
def test_invalid_parameters_rejected(make) -> None:
    with pytest.raises(ValueError):
        make()


# ------------------------------------------------------------------
# integration with Topology / Problem
# ------------------------------------------------------------------


def test_fill_with_named_filler() -> None:
    j = complete(8).fill(fillers.pm_one(rng=0))
    assert np.array_equal(j, j.T)
    assert np.all(np.diag(j) == 0.0)
    assert set(np.unique(j)) <= {-1.0, 0.0, 1.0}


def test_gaussian_coupling_builds_a_valid_problem() -> None:
    p = Problem(complete(8).fill(fillers.gaussian(rng=3)))
    assert not np.all(p.j == p.j[0, 1])  # not all identical


def test_filler_supplies_the_bias_vector() -> None:
    t = complete(5)
    p = Problem(t.fill(0.0), fillers.resolve(fillers.gaussian(rng=4), t.n))
    assert not np.all(p.h == 0.0)
