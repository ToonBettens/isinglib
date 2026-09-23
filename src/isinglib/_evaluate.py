from __future__ import annotations

from typing import overload

import numpy as np

__all__ = (
    "effective_field",
    "energy",
    "spin_flip_effective_field_update",
    "spin_flip_energy_update",
)


@overload
def effective_field(j: np.ndarray, h: np.ndarray, s: np.ndarray, *, i: None = None) -> np.ndarray: ...
@overload
def effective_field(j: np.ndarray, h: np.ndarray, s: np.ndarray, *, i: int) -> float: ...
def effective_field(
    j: np.ndarray,
    h: np.ndarray,
    s: np.ndarray,
    *,
    i: int | None = None,
) -> np.ndarray | float:
    """Effective field `h_eff = j @ s + h`.

    Args:
        j: (n, n) coupling matrix.
        h: (n,) bias vector.
        s: (n,) spin state.
        i: If given, return only `h_eff[i]` as a float instead of the full array.
    """
    if i is None:
        return j @ s + h
    return float(j[i] @ s + h[i])


def energy(
    j: np.ndarray,
    h: np.ndarray,
    c: float,
    s: np.ndarray,
    *,
    h_eff: np.ndarray | None = None,
) -> float:
    """`E(s) = -0.5 sᵀ j s - hᵀ s + c`.

    Pass `h_eff` (from `effective_field`) if already available,
    to avoid recomputing it.
    """
    if h_eff is None:
        h_eff = effective_field(j, h, s)
    return float(-0.5 * (s @ (h_eff + h)) + c)


@overload
def spin_flip_energy_update(s: np.ndarray, h_eff: np.ndarray, *, i: None = None) -> np.ndarray: ...
@overload
def spin_flip_energy_update(s: np.ndarray, h_eff: np.ndarray, *, i: int) -> float: ...
def spin_flip_energy_update(
    s: np.ndarray,
    h_eff: np.ndarray,
    *,
    i: int | None = None,
) -> np.ndarray | float:
    """ΔE from flipping spin(s), given `h_eff` evaluated at the pre-flip state.

    Args:
        s: (n,) spin state, before any flip.
        h_eff: Effective field at `s` (from `effective_field`).
        i: If given, return the scalar ΔE for flipping only spin `i`.
           Otherwise return the `(n,)` array of ΔE for flipping each spin
           independently (holding all others fixed).
    """
    if i is None:
        return 2.0 * s * h_eff
    return float(2.0 * s[i] * h_eff[i])


def spin_flip_effective_field_update(j: np.ndarray, s: np.ndarray, i: int) -> np.ndarray:
    """Delta to add to `h_eff` after flipping spin `i`.

    `s[i]` must still hold its pre-flip value when this is called.
    """
    return -2.0 * s[i] * j[:, i]
