from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any

import numpy as np

from isinglib.problem import Problem

__all__ = ("Solution", "run_batch")


class Solution(types.SimpleNamespace):
    """A solver run's output: whatever fields the solver decided to report.

    `run_batch` adds `run_index`, the run's position in its batch.
    """

    __eq__ = object.__eq__
    __ne__ = object.__ne__
    __hash__ = object.__hash__


def run_batch(
    solver: Callable[..., Solution],
    problem: Problem,
    n_runs: int,
    *,
    rng: np.random.Generator | int | None = None,
    **kwargs: Any,
) -> list[Solution]:
    """Run `solver` on `problem` `n_runs` times, each run on its own random stream.

    The streams are spawned from `rng`, so a seeded batch is reproducible.

    Args:
        solver: A solver taking `rng`, e.g. `simulated_annealing`.
        problem: The problem every run solves.
        n_runs: Number of runs.
        rng: Root random generator or seed.
        **kwargs: Passed unchanged to every run.

    Returns:
        One `Solution` per run, in order, each with `run_index` set.
    """
    if n_runs < 0:
        raise ValueError("n_runs must be non-negative.")
    root = rng if isinstance(rng, np.random.Generator) else np.random.SeedSequence(rng)
    solutions = []
    for index, seed in enumerate(root.spawn(n_runs)):
        solution = solver(problem, rng=np.random.default_rng(seed), **kwargs)
        solution.run_index = index
        solutions.append(solution)
    return solutions
