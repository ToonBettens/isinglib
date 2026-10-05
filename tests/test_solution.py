"""`Solution` as a plain attribute bag: nothing about its shape is enforced."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from isinglib import Problem, Solution
from isinglib.solvers import run_batch, simulated_annealing

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


# ------------------------------------------------------------------
# run_batch: one stream per run
# ------------------------------------------------------------------


def draw(problem: Problem, *, rng: np.random.Generator, **kwargs: Any) -> Solution:
    """A stand-in solver that reports its first random draw and its kwargs."""
    return Solution(draw=rng.random(), kwargs=kwargs)


def test_batch_returns_one_indexed_solution_per_run(small_problem: Problem) -> None:
    sols = run_batch(draw, small_problem, 5, rng=0)
    assert [s.run_index for s in sols] == [0, 1, 2, 3, 4]


def test_batch_forwards_kwargs_to_every_run(small_problem: Problem) -> None:
    sols = run_batch(draw, small_problem, 3, rng=0, n_sweeps=7)
    assert all(s.kwargs == {"n_sweeps": 7} for s in sols)


def test_runs_get_distinct_streams(small_problem: Problem) -> None:
    """The point of spawning: passing one seed to every run would make them identical."""
    draws = [s.draw for s in run_batch(draw, small_problem, 10, rng=0)]
    assert len(set(draws)) == 10


@pytest.mark.parametrize("make_rng", [lambda: 0, lambda: np.random.default_rng(0)], ids=["int", "generator"])
def test_seeded_batch_is_reproducible(small_problem: Problem, make_rng: Callable[[], Any]) -> None:
    first = run_batch(draw, small_problem, 4, rng=make_rng())
    second = run_batch(draw, small_problem, 4, rng=make_rng())
    assert [s.draw for s in first] == [s.draw for s in second]


def test_a_generator_advances_between_batches(small_problem: Problem) -> None:
    rng = np.random.default_rng(0)
    first, second = run_batch(draw, small_problem, 3, rng=rng), run_batch(draw, small_problem, 3, rng=rng)
    assert [s.draw for s in first] != [s.draw for s in second]


def test_a_single_run_reproduces_on_its_own(make_problem: Callable[..., Problem]) -> None:
    """As documented: run k is the solver on the k-th spawned seed, no replay of runs 0..k-1."""
    p = make_problem(20, seed=1)
    sols = run_batch(simulated_annealing, p, 4, rng=0, n_sweeps=5)
    alone = simulated_annealing(p, rng=np.random.default_rng(np.random.SeedSequence(0).spawn(4)[2]), n_sweeps=5)
    assert np.array_equal(alone.spins, sols[2].spins)
    assert alone.energy == sols[2].energy


def test_empty_batch(small_problem: Problem) -> None:
    assert run_batch(draw, small_problem, 0, rng=0) == []


def test_negative_runs_rejected(small_problem: Problem) -> None:
    with pytest.raises(ValueError, match="n_runs"):
        run_batch(draw, small_problem, -1)
