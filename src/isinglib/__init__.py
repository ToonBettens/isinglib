"""isinglib — a pure-Python toolkit for Ising-machine research.

Problem generation, benchmark loading, and solvers (algorithmic + ODE) over a
single immutable ``Problem`` type. No visualization, persistence, or circuit
dependencies — those live in sibling repos (see ``private/architecture.md``).
"""

from isinglib.core import Problem, Solution, Solver
from isinglib.solvers.algorithmic import (
    ExhaustiveSolver,
    GreedySolver,
    SimulatedAnnealingSolver,
    TabuSearchSolver,
)
from isinglib.topology import Topology

__all__ = (
    "ExhaustiveSolver",
    "GreedySolver",
    "Problem",
    "SimulatedAnnealingSolver",
    "Solution",
    "Solver",
    "TabuSearchSolver",
    "Topology",
)
