from __future__ import annotations

import numpy as np
import numpy.typing as npt

from isinglib.problem import Problem
from isinglib.states import ensure_spins
from isinglib.topology.topology import Topology

__all__ = ("mattis",)


def mattis(
    topology: Topology,
    state: npt.ArrayLike,
    strength: float = 1.0,
) -> Problem:
    """Mattis model: every coupling points towards `state`, `j_ij = strength * s_i * s_j`.

    Every coupling is satisfied at `state` at once, so the problem is
    unfrustrated and `state` and `-state` are its ground states.

    Args:
        topology: The coupling graph.
        state: The ground state the couplings point towards, valued in {-1, +1}.
        strength: Magnitude of every coupling (must be non-negative).
    """
    s = ensure_spins(state, topology.n)
    if strength < 0.0:
        raise ValueError("strength must be non-negative; the signs come from state.")
    return Problem(topology.fill(strength) * np.outer(s, s))
