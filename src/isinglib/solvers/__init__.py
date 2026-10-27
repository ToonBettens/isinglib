"""Solvers: plain functions, `Problem` in, `Solution` out."""

from isinglib.solvers.base import Solution, run_batch
from isinglib.solvers.branch_and_bound import branch_and_bound
from isinglib.solvers.exhaustive import exhaustive
from isinglib.solvers.local_search import local_search
from isinglib.solvers.parallel_tempering import parallel_tempering
from isinglib.solvers.simulated_annealing import simulated_annealing
from isinglib.solvers.simulated_bifurcation import max_stable_dt, simulated_bifurcation
from isinglib.solvers.tabu_search import tabu_search

__all__ = (
    "Solution",
    "branch_and_bound",
    "exhaustive",
    "local_search",
    "max_stable_dt",
    "parallel_tempering",
    "run_batch",
    "simulated_annealing",
    "simulated_bifurcation",
    "tabu_search",
)
