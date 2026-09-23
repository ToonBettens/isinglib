from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from isinglib.problem import Problem


def read(path: str | Path, *, dtype: np.dtype | None = None) -> Problem:
    """Load a Problem from a JSON file written by `write`.

    Args:
        path: Path to the `.json` file.
        dtype: If provided, overrides the dtype stored in the file.
    """
    with Path(path).open() as f:
        data = json.load(f)
    return Problem(
        j=np.array(data["j"]),
        h=np.array(data["h"]),
        c=data.get("c", 0.0),
        dtype=np.dtype(data["dtype"]) if dtype is None else dtype,
    )


def write(problem: Problem, path: str | Path, *, indent: int = 2) -> None:
    """Serialise a Problem to a JSON file.

    All numerical fields are stored as nested lists; `dtype` is preserved.
    The file can be round-tripped through `read`.

    Args:
        problem: The Problem to serialise.
        path: Destination file path.
        indent: JSON indentation level for human-readable output.
    """
    data = {
        "dtype": str(problem.dtype),
        "c": problem.c,
        "j": problem.j.tolist(),
        "h": problem.h.tolist(),
    }
    with Path(path).open() as f:
        json.dump(data, f, indent=indent)
