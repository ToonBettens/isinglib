from __future__ import annotations

from abc import ABC, abstractmethod

from isinglib.core.problem import Problem
from isinglib.core.solution import Solution

__all__ = ("Solver",)


class Solver(ABC):
    """Contract every solver must implement.

    Hyperparameters belong on `__init__`; `solve` takes only a `Problem`
    and returns exactly one `Solution`.
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
