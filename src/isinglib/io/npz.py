from __future__ import annotations

from pathlib import Path

import numpy as np

from isinglib.io._record import (
    FORMAT_VERSION,
    LayoutOptions,
    Record,
    check_version,
    from_record,
    to_record,
)
from isinglib.problem import Problem

__all__ = ("read", "write")


def read(path: str | Path) -> Problem:
    """Load a Problem from a `.npz` file.

    Args:
        path: Path to the `.npz` file.
    """
    with np.load(path, allow_pickle=False) as data:
        version = int(data["format_version"]) if "format_version" in data else None
        check_version(version)

        sparse = str(data["layout"]) == "sparse"
        record = Record(
            j=(data["src"], data["dst"], data["weight"]) if sparse else data["j"],
            dtype=str(data["dtype"]),
            n=int(data["n"]),
            c=float(data["c"]),
            h=data["h"],
            version=version if version is not None else FORMAT_VERSION,
        )
    return from_record(record)


def write(
    problem: Problem, path: str | Path, *, layout: LayoutOptions = "auto", overwrite: bool = False
) -> None:
    """Serialise a Problem to a `.npz` file, at exactly `path`.

    Args:
        problem: The Problem to serialise.
        path: Destination file path.
        layout: `"sparse"`, `"dense"` or `"auto"`.
        overwrite: If False (default), raise if `path` already exists.
    """
    path = Path(path)
    if not overwrite and path.exists():
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it.")
    record = to_record(problem, layout)
    with path.open("wb") as f:  # prevent np.savez_compressed from adding .npz to file-paths lacking it
        if isinstance(record.j, tuple):
            src, dst, weight = record.j
            np.savez_compressed(
                f,
                format_version=record.version,
                layout=record.layout,
                dtype=record.dtype,
                n=record.n,
                c=record.c,
                h=record.h,
                src=src,
                dst=dst,
                weight=weight,
            )
        else:
            np.savez_compressed(
                f,
                format_version=record.version,
                layout=record.layout,
                dtype=record.dtype,
                n=record.n,
                c=record.c,
                h=record.h,
                j=record.j,
            )
