"""Suffix -> backend dispatch for `io.read` / `io.write`."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

from isinglib.io import hdf5, json, npz

if TYPE_CHECKING:
    from isinglib.problem import Problem

__all__ = ("read", "write")

type Format = Literal["json", "npz", "hdf5"]

_BACKENDS: dict[Format, ModuleType] = {"json": json, "npz": npz, "hdf5": hdf5}
_SUFFIXES: dict[str, Format] = {
    ".json": "json",
    ".npz": "npz",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
}


def _backend_for(path: str | Path, format: Format | None) -> ModuleType:
    if format is not None:
        try:
            return _BACKENDS[format]
        except KeyError:
            raise ValueError(
                f"Unknown format {format!r}; expected one of {sorted(_BACKENDS)}."
            ) from None

    suffix = Path(path).suffix.lower()
    try:
        return _BACKENDS[_SUFFIXES[suffix]]
    except KeyError:
        raise ValueError(
            f"Cannot infer a format from suffix {suffix or '(none)'!r}; "
            f"expected one of {sorted(_SUFFIXES)}, or pass format=."
        ) from None


def read(path: str | Path, *, format: Format | None = None, **kwargs: Any) -> Problem:
    """Read a Problem, choosing the backend from the file suffix.

    Args:
        path: Path to read from.
        format: Override the suffix-based choice.
        **kwargs: Forwarded to the backend's `read` (e.g. `key=` for hdf5).
    """
    return _backend_for(path, format).read(path, **kwargs)


def write(
    problem: Problem, path: str | Path, *, format: Format | None = None, **kwargs: Any
) -> None:
    """Write a Problem, choosing the backend from the file suffix.

    Args:
        problem: The Problem to serialise.
        path: Path to write to.
        format: Override the suffix-based choice.
        **kwargs: Forwarded to the backend's `write` (e.g. `layout=`, `key=`, `overwrite=`).
    """
    _backend_for(path, format).write(problem, path, **kwargs)
