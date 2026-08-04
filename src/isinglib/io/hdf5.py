from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np

from isinglib.core.problem import Problem
from isinglib.io._sparse import choose_encoding, from_edges, to_edges


def read(
    path: str | Path,
    *,
    key: str = "problem",
    meta: dict | None = None,
    dtype: np.dtype | None = None,
) -> Problem:
    """Load a Problem from an HDF5 file written by `write`.

    Args:
        path: Path to the `.h5` / `.hdf5` file.
        key: HDF5 group path within the file (e.g. `"experiment/run1"`).
        meta: If provided, replaces the metadata stored in the file.
        dtype: If provided, overrides the dtype stored in the file.
    """
    import h5py

    with h5py.File(path, "r") as f:
        grp = f[key]
        c = float(grp.attrs["c"])
        file_dtype = np.dtype(grp.attrs["dtype"])
        file_meta = json.loads(grp.attrs.get("meta", "{}"))
        encoding = grp.attrs.get("encoding", "dense")

        if encoding == "sparse":
            n = int(grp.attrs["n"])
            j = from_edges(n, grp["src"][:], grp["dst"][:], grp["weight"][:], file_dtype)
        elif encoding == "dense":
            j = grp["j"][:]
        else:
            raise ValueError(f"Unknown hdf5 encoding {encoding!r}.")
        h = grp["h"][:]

    return Problem(
        j=j,
        h=h,
        c=c,
        dtype=file_dtype if dtype is None else dtype,
        meta=meta if meta is not None else file_meta,
    )


def write(
    problem: Problem,
    path: str | Path,
    *,
    key: str = "problem",
    compression: str | None = "gzip",
    encoding: Literal["auto", "sparse", "dense"] = "auto",
) -> None:
    """Serialise a Problem into an HDF5 group.

    Opens the file in append mode, so multiple problems can coexist in one
    file under different keys. An existing group at `key` is overwritten.

    Args:
        problem: The Problem to serialise.
        path: Destination file path.
        key: HDF5 group path within the file (e.g. `"experiment/run1"`).
        compression: HDF5 compression filter for array datasets. `None` disables compression.
        encoding: "sparse" stores only the upper-triangle edges of `j`;
            `"dense"` stores `j` as-is. `"auto"` (default)
            picks based on `j`'s actual density.
    """
    import h5py

    if encoding == "auto":
        encoding = choose_encoding(problem)

    with h5py.File(path, "a") as f:
        if key in f:
            del f[key]
        grp = f.create_group(key)
        if encoding == "sparse":
            src, dst, weight = to_edges(problem)
            grp.create_dataset("src", data=src, compression=compression)
            grp.create_dataset("dst", data=dst, compression=compression)
            grp.create_dataset("weight", data=weight, compression=compression)
            grp.attrs["n"] = problem.n
        elif encoding == "dense":
            grp.create_dataset("j", data=problem.j, compression=compression)
        else:
            raise ValueError(f"Unknown encoding {encoding!r}; expected 'auto', 'sparse', or 'dense'.")
        grp.create_dataset("h", data=problem.h, compression=compression)
        grp.attrs["c"] = problem.c
        grp.attrs["dtype"] = str(problem.dtype)
        grp.attrs["meta"] = json.dumps(dict(problem.meta))
        grp.attrs["encoding"] = encoding
