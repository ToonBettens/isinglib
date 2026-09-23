"""Problem generators: parameters (and often a `Topology`) in, `Problem` out.

One of the three ways a `Problem` comes into existence, alongside `mappings`
(an instance of another problem, re-encoded) and benchmark loading (a curated
instance, which lives outside this library).
"""

from isinglib.generators.planted import planted_solution

__all__ = ("planted_solution",)
