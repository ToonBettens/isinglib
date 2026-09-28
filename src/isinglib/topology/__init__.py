"""Topologies: coupling graphs, regular and random, filled into `j` via `fillers`."""

from isinglib.topology import fillers
from isinglib.topology.regular import (
    chimera,
    complete,
    cycle,
    grid,
    king,
    mobius_ladder,
    path,
    ring,
    star,
)
from isinglib.topology.stochastic import (
    barabasi_albert,
    erdos_renyi,
    watts_strogatz,
)
from isinglib.topology.topology import Topology

__all__ = (
    "Topology",
    "barabasi_albert",
    "chimera",
    "complete",
    "cycle",
    "erdos_renyi",
    "fillers",
    "grid",
    "king",
    "mobius_ladder",
    "path",
    "ring",
    "star",
    "watts_strogatz",
)
