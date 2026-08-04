"""Pure graph structure: the Topology class and its generators.

Decoupled from weights on purpose — see `Problem.from_topology` to attach
coupling/bias and get a full `Problem`.
"""

from isinglib.topology.generators import (
    barabasi_albert,
    complete,
    cycle,
    erdos_renyi,
    grid,
    king,
    path,
    ring,
    star,
    watts_strogatz,
)
from isinglib.topology.topology import Topology

__all__ = (
    "Topology",
    "barabasi_albert",
    "complete",
    "cycle",
    "erdos_renyi",
    "grid",
    "king",
    "path",
    "ring",
    "star",
    "watts_strogatz",
)
