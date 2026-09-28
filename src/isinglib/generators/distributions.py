from __future__ import annotations

import numpy as np
import numpy.typing as npt

from isinglib.dtypes import ensure_float_array
from isinglib.problem import Problem
from isinglib.topology.topology import Topology

__all__ = ("choice", "constant", "gaussian", "pm_one", "uniform")


def constant(topology: Topology, value: float = 1.0) -> Problem:
    """Every coupling equal to `value`.

    Args:
        topology: The coupling graph.
        value: The coupling on every edge.
    """
    return Problem(topology.fill(value))


def uniform(
    topology: Topology,
    low: float = -1.0,
    high: float = 1.0,
    *,
    rng: np.random.Generator | int | None = None,
) -> Problem:
    """Couplings drawn uniformly from `[low, high)`.

    Args:
        topology: The coupling graph.
        low: Lower bound (inclusive).
        high: Upper bound (exclusive).
        rng: Random generator or seed.
    """
    if high < low:
        raise ValueError("high must be greater than or equal to low.")
    _rng = np.random.default_rng(rng)
    return Problem(topology.fill(_rng.uniform(low, high, topology.num_edges)))


def gaussian(
    topology: Topology,
    sigma: float = 1.0,
    *,
    mean: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> Problem:
    """Couplings drawn from a normal distribution.

    Args:
        topology: The coupling graph.
        sigma: Standard deviation; must be non-negative.
        mean: Distribution mean.
        rng: Random generator or seed.
    """
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative.")
    _rng = np.random.default_rng(rng)
    return Problem(topology.fill(_rng.normal(mean, sigma, topology.num_edges)))


def pm_one(
    topology: Topology,
    p: float = 0.5,
    *,
    rng: np.random.Generator | int | None = None,
) -> Problem:
    """Couplings of -1 or +1, the ±J spin glass.

    Args:
        topology: The coupling graph.
        p: Probability of a coupling being +1.
        rng: Random generator or seed.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1].")
    _rng = np.random.default_rng(rng)
    return Problem(topology.fill(np.where(_rng.random(topology.num_edges) < p, 1.0, -1.0)))


def choice(
    topology: Topology,
    values: npt.ArrayLike,
    *,
    p: npt.ArrayLike | None = None,
    rng: np.random.Generator | int | None = None,
) -> Problem:
    """Couplings drawn from a discrete set, e.g. the weight levels a chip can represent.

    Args:
        topology: The coupling graph.
        values: The values to draw from, e.g. `[-1, 0, 1]`.
        p: Probability of each value; uniform if not given.
        rng: Random generator or seed.
    """
    pool = ensure_float_array(values)
    if pool.ndim != 1 or pool.size == 0:
        raise ValueError("values must be a non-empty 1-D sequence.")
    probabilities = None if p is None else ensure_float_array(p)
    _rng = np.random.default_rng(rng)
    return Problem(topology.fill(_rng.choice(pool, size=topology.num_edges, p=probabilities)))
