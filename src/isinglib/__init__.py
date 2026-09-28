"""Ising problems, generators and solvers for Ising machine research."""

from importlib.metadata import version

from isinglib import generators, io, mappings, solvers, states, topology
from isinglib.problem import Problem
from isinglib.solvers.base import Solution
from isinglib.topology.topology import Topology

__version__ = version("isinglib")

__all__ = (
    "Problem",
    "Solution",
    "Topology",
    "__version__",
    "generators",
    "io",
    "mappings",
    "solvers",
    "states",
    "topology",
)
