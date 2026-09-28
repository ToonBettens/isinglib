"""Generators: parameters in; a `Problem` out."""

from isinglib.generators.distributions import (
    choice,
    constant,
    gaussian,
    pm_one,
    uniform,
)
from isinglib.generators.planted import mattis
from isinglib.generators.spinglass import edwards_anderson, sherrington_kirkpatrick

__all__ = (
    "choice",
    "constant",
    "edwards_anderson",
    "gaussian",
    "mattis",
    "pm_one",
    "sherrington_kirkpatrick",
    "uniform",
)
