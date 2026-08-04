from __future__ import annotations

import numpy as np
import numpy.typing as npt

from isinglib.core.problem import Problem
from isinglib.topology.topology import FillerLike, Topology
from isinglib.utils.dtypes import ensure_float_array, ensure_float_dtype, is_scalar_like

__all__ = ("planted_solution",)


def planted_solution(
    topology: Topology,
    coupling: FillerLike,
    bias: FillerLike = 0.0,
    *,
    planted: npt.ArrayLike | None = None,
    rng: np.random.Generator | int | None = None,
    meta: dict | None = None,
    dtype: np.dtype | None = None,
) -> Problem:
    """Build a Problem with a known ground state, planted by gauge transform.

    `coupling` and `bias` are *magnitude* specifications and must be
    non-negative (e.g. `lambda n: rng.uniform(0.1, 1.0, n)`); the sign of
    each coupling and bias term is chosen so that it is individually
    maximized in the energy's favor at `planted`. Since every term is
    simultaneously maximized by the same configuration, their sum is too —
    making `planted` a global ground state by construction. This is just a
    ferromagnet written in the gauge where "all aligned" means `planted`
    instead of all-ones.

    `planted` is guaranteed to be *a* ground state, but not necessarily the
    unique one: a node with zero total incident coupling and zero bias can
    have either spin value at no energy cost, and an all-zero `bias` leaves
    the usual `s -> -s` global flip degeneracy intact.

    Args:
        topology: Graph structure defining which spins interact.
        coupling: Edge coupling magnitudes: scalar or callable(n_edges) -> array. Must be non-negative.
        bias: Node bias magnitudes: scalar or callable(n_nodes) -> array. Must be non-negative.
        planted: The spin configuration to plant, valued in {-1, +1}. Drawn uniformly at random if not given.
        rng: Random generator or seed, used only when `planted` is None.
        meta: Provenance metadata, merged with the planting info this function always records.
        dtype: Target float dtype (default float64).

    Returns:
        A Problem whose ground state is `planted`, with
        `meta["planted_spins"]` and `meta["ground_truth_energy"]` set.
    """
    dtype = ensure_float_dtype(dtype)

    if planted is None:
        t = np.random.default_rng(rng).choice(np.array([-1.0, 1.0], dtype=dtype), size=topology.n)
    else:
        t = np.asarray(planted, dtype=dtype)
        if t.shape != (topology.n,):
            raise ValueError(f"planted must have shape ({topology.n},), got {t.shape}.")
        if not np.all(np.isin(t, (-1.0, 1.0))):
            raise ValueError("planted must contain only -1.0 and 1.0.")

    j_mag = topology.fill(coupling)
    if np.any(j_mag < 0.0):
        raise ValueError("coupling must be non-negative; sign is chosen by planting.")
    j = j_mag * np.outer(t, t)

    if is_scalar_like(bias):
        b_mag = np.full(topology.n, float(bias))  # type: ignore[arg-type]
    elif callable(bias):
        b_mag = ensure_float_array(bias(topology.n))
    else:
        raise TypeError("bias must be a scalar or callable(n_nodes) -> array.")
    if np.any(b_mag < 0.0):
        raise ValueError("bias must be non-negative; sign is chosen by planting.")
    h = b_mag * t

    problem_meta = dict(meta or {})
    problem_meta["planted_spins"] = t.tolist()
    problem_meta["ground_truth_energy"] = -0.5 * float(np.sum(j_mag)) - float(np.sum(b_mag))

    return Problem(j=j, h=h, dtype=dtype, meta=problem_meta)
