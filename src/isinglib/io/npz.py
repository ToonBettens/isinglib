from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np

from isinglib.core.problem import Problem
from isinglib.io._sparse import choose_encoding, from_edges, to_edges


def read(path: str | Path) -> Problem:
    """Load a Problem from an `.npz` file written by `write`.

    The encoding (sparse or dense) used to store the file is recorded
    inside it, so this always reconstructs correctly without being told.

    Args:
        path: Path to the `.npz` file.
    """
    with np.load(path, allow_pickle=False) as data:
        encoding = str(data["encoding"])
        dtype = np.dtype(str(data["dtype"]))
        c = float(data["c"])
        meta = json.loads(str(data["meta"]))

        if encoding == "sparse":
            n = int(data["n"])
            j = from_edges(n, data["src"], data["dst"], data["weight"], dtype)
            h = data["h"]
        elif encoding == "dense":
            j = data["j"]
            h = data["h"]
        else:
            raise ValueError(f"Unknown npz encoding {encoding!r}.")

    return Problem(j=j, h=h, c=c, dtype=dtype, meta=meta)


def write(
    problem: Problem,
    path: str | Path,
    *,
    encoding: Literal["auto", "sparse", "dense"] = "auto",
) -> None:
    """Serialise a Problem to a compressed `.npz` file.

    Args:
        problem: The Problem to serialise.
        path: Destination file path.
        encoding: "sparse" stores only the upper-triangle edges of `j`;
            `"dense"` stores `j` as-is. `"auto"` (default)
            picks based on `j`'s actual density.
    """
    h = problem.h
    assert h is not None  # always set by Problem.__post_init__
    meta_json = json.dumps(dict(problem.meta))

    if encoding == "auto":
        encoding = choose_encoding(problem)

    if encoding == "sparse":
        src, dst, weight = to_edges(problem)
        np.savez_compressed(
            path,
            encoding="sparse",
            dtype=str(problem.dtype),
            c=problem.c,
            meta=meta_json,
            n=problem.n,
            src=src,
            dst=dst,
            weight=weight,
            h=h,
        )
    elif encoding == "dense":
        np.savez_compressed(
            path,
            encoding="dense",
            dtype=str(problem.dtype),
            c=problem.c,
            meta=meta_json,
            j=problem.j,
            h=h,
        )
    else:
        raise ValueError(f"Unknown encoding {encoding!r}; expected 'auto', 'sparse', or 'dense'.")
