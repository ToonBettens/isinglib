from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from isinglib import Problem, generators
from isinglib.topology import Topology, complete, grid, star

# (name, call): every distribution generator, on any topology.
DISTRIBUTIONS: list[tuple[str, Callable[..., Problem]]] = [
    ("constant", lambda t: generators.constant(t, 2.0)),
    ("uniform", lambda t: generators.uniform(t, rng=0)),
    ("gaussian", lambda t: generators.gaussian(t, rng=0)),
    ("pm_one", lambda t: generators.pm_one(t, rng=0)),
    ("choice", lambda t: generators.choice(t, [-1.0, 0.5, 1.0], rng=0)),
]


@pytest.fixture(params=DISTRIBUTIONS, ids=lambda d: d[0])
def distribution(request: pytest.FixtureRequest) -> Callable[..., Problem]:
    return request.param[1]


# ------------------------------------------------------------------
# Shared contract
# ------------------------------------------------------------------


def test_couplings_live_only_on_the_topology(distribution: Callable[..., Problem]) -> None:
    t = grid(3, 3)
    p = distribution(t)
    off_edge = np.ones((t.n, t.n), dtype=bool)
    off_edge[t.src, t.dst] = off_edge[t.dst, t.src] = False
    assert np.all(p.j[off_edge] == 0.0)


def test_generated_problems_have_zero_bias(distribution: Callable[..., Problem]) -> None:
    assert np.all(distribution(star(5)).h == 0.0)


def test_a_graph_without_edges_gives_zero_couplings(distribution: Callable[..., Problem]) -> None:
    assert np.all(distribution(Topology(4, [], [])).j == 0.0)


# ------------------------------------------------------------------
# Distributions
# ------------------------------------------------------------------


def test_constant_puts_the_value_on_every_edge() -> None:
    t = complete(4)
    p = generators.constant(t, -1.0)
    assert np.all(p.j[t.src, t.dst] == -1.0)


def test_uniform_within_bounds() -> None:
    t = complete(15)
    values = generators.uniform(t, 0.1, 0.9, rng=0).j[t.src, t.dst]
    assert np.all((values >= 0.1) & (values < 0.9))


def test_gaussian_is_reproducible_from_a_seed() -> None:
    assert generators.gaussian(complete(8), rng=3) == generators.gaussian(complete(8), rng=3)


def test_a_shared_generator_draws_fresh_values() -> None:
    rng = np.random.default_rng(0)
    assert generators.gaussian(complete(8), rng=rng) != generators.gaussian(complete(8), rng=rng)


def test_pm_one_values() -> None:
    t = complete(10)
    assert set(np.unique(generators.pm_one(t, rng=0).j[t.src, t.dst])) <= {-1.0, 1.0}
    assert np.all(generators.pm_one(t, p=1.0, rng=0).j[t.src, t.dst] == 1.0)
    assert np.all(generators.pm_one(t, p=0.0, rng=0).j[t.src, t.dst] == -1.0)


def test_choice_draws_from_the_given_set() -> None:
    t = complete(10)
    values = generators.choice(t, [-1.0, 0.5, 1.0], rng=0).j[t.src, t.dst]
    assert set(np.unique(values)) <= {-1.0, 0.5, 1.0}


@pytest.mark.parametrize(
    ("make", "kwargs"),
    [
        (generators.uniform, {"low": 1.0, "high": 0.0}),
        (generators.gaussian, {"sigma": -1.0}),
        (generators.pm_one, {"p": 1.5}),
        (generators.choice, {"values": []}),
    ],
)
def test_invalid_parameters_rejected(make: Callable[..., Problem], kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        make(complete(4), **kwargs)


# ------------------------------------------------------------------
# Named models
# ------------------------------------------------------------------


def test_sherrington_kirkpatrick_is_complete_with_variance_one_over_n() -> None:
    n = 200
    p = generators.sherrington_kirkpatrick(n, rng=0)
    couplings = p.j[np.triu_indices(n, k=1)]
    assert np.all(couplings != 0.0)
    assert np.var(couplings) == pytest.approx(1.0 / n, rel=0.05)
    assert np.all(p.h == 0.0)


@pytest.mark.parametrize(("gaussian_couplings", "values"), [(False, {-1.0, 1.0}), (True, None)])
def test_edwards_anderson_on_the_periodic_cubic_lattice(gaussian_couplings: bool, values: set[float] | None) -> None:
    p = generators.edwards_anderson(4, gaussian_couplings=gaussian_couplings, rng=0)
    lattice = grid(4, 4, 4, periodic=True)
    assert p.n == 64
    assert np.count_nonzero(p.j) == 2 * lattice.num_edges
    couplings = p.j[lattice.src, lattice.dst]
    if values is not None:
        assert set(np.unique(couplings)) == values
    else:
        assert len(np.unique(couplings)) == lattice.num_edges


def test_edwards_anderson_follows_the_dimension() -> None:
    assert generators.edwards_anderson(5, d=2, rng=0).n == 25
