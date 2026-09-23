from __future__ import annotations

import numpy as np
import numpy.typing as npt

from isinglib.dtypes import ensure_float_array, ensure_float_dtype
from isinglib.problem import Problem
from isinglib.topology.fillers import FillerLike, resolve
from isinglib.topology.topology import Topology

__all__ = ("planted_solution",)


def planted_solution(
    topology: Topology,
    coupling: FillerLike,
    bias: FillerLike = 0.0,
    *,
    planted: npt.ArrayLike,
    dtype: npt.DTypeLike | None = None,
) -> Problem:
    """Build a Problem with a known ground state, planted by gauge transform.

    `coupling` and `bias` are *magnitude* specifications and must be
    non-negative (e.g. `fillers.uniform(0.1, 1.0)`); the sign of each coupling
    and bias term is chosen so that it is individually maximized in the energy's
    favor at `planted`. Since every term is simultaneously maximized by the same
    configuration, their sum is too — making `planted` a global ground state by
    construction. This is just a ferromagnet written in the gauge where "all
    aligned" means `planted` instead of all-ones.

    `planted` is guaranteed to be *a* ground state, but not necessarily the
    unique one: a node with zero total incident coupling and zero bias can
    have either spin value at no energy cost, and an all-zero `bias` leaves
    the usual `s -> -s` global flip degeneracy intact.

    The spins are supplied by the caller rather than drawn here, so nothing the
    function knows is thrown away — draw them with `states.random_spins(n)` if
    you want them random. The ground-state energy is then simply
    `problem.energy(planted)`.

    Args:
        topology: Graph structure defining which spins interact.
        coupling: Edge coupling magnitudes: scalar or `callable(n_edges) -> array`.
            Must be non-negative.
        bias: Node bias magnitudes: scalar or `callable(n_nodes) -> array`.
            Must be non-negative.
        planted: The spin configuration to plant, valued in {-1, +1}.
        dtype: Target float dtype (default float64).
    """
    resolved = ensure_float_dtype(dtype)

    spins = ensure_float_array(planted, resolved)
    if spins.shape != (topology.n,):
        raise ValueError(f"planted must have shape ({topology.n},), got {spins.shape}.")
    if not np.all(np.isin(spins, (-1.0, 1.0))):
        raise ValueError("planted must contain only -1.0 and 1.0.")

    coupling_magnitudes = topology.fill(coupling)
    if np.any(coupling_magnitudes < 0.0):
        raise ValueError("coupling must be non-negative; sign is chosen by planting.")

    bias_magnitudes = resolve(bias, topology.n, dtype=resolved)
    if np.any(bias_magnitudes < 0.0):
        raise ValueError("bias must be non-negative; sign is chosen by planting.")

    return Problem(
        coupling_magnitudes * np.outer(spins, spins),
        bias_magnitudes * spins,
        dtype=resolved,
    )
