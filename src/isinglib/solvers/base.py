from __future__ import annotations

from abc import ABC, abstractmethod

from isinglib.problem import Problem
from isinglib.solution import Solution

__all__ = ("Solver",)


class Solver(ABC):
    """Contract every solver must implement.

    Hyperparameters belong on `__init__`; `solve` takes only a `Problem`
    and returns exactly one `Solution`.

    Solvers hold no mutable run state: given an integer seed, `solve` is a pure
    function of its configuration and the problem, so repeated calls return the
    same solution and one instance is safe to share across parallel workers.
    Run-to-run variation is therefore something a caller supplies — by reseeding,
    or (once supported) by varying the initial state — never something a solver
    accumulates between calls.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, stable identifier recorded in `Solution.solver_name`."""
        ...

    @abstractmethod
    def solve(self, problem: Problem) -> Solution:
        """Solve `problem` and return a single `Solution`."""
        ...
