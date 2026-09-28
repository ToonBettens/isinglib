from __future__ import annotations

import types

__all__ = ("Solution",)


class Solution(types.SimpleNamespace):
    """A solver run's output: whatever fields the solver decided to report."""
    __eq__ = object.__eq__
    __ne__ = object.__ne__
    __hash__ = object.__hash__
