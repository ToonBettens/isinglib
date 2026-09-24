from __future__ import annotations

from pathlib import Path
from typing import Any

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


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
                    "The hdf5 backend needs h5py, which is an optional dependency. "
                    "Install it with: pip install 'isinglib[io]'"
                ) from e
    return h5py


def read(path: str | Path, *, key: str = "problem") -> Problem:
    """Load a Problem from a HDF5 file.

    Args:
        path: Path to the `.h5` / `.hdf5` file.
        key: HDF5 group path within the file (e.g. `"experiment/run1"`).
    """
    h5py = _import_h5py()

    with h5py.File(path, "r") as f:
        grp = f[key]
        version = grp.attrs.get("format_version")
        check_version(int(version) if version is not None else None)

        sparse = grp.attrs.get("layout", "dense") == "sparse"
        record = Record(
            j=(grp["src"][:], grp["dst"][:], grp["weight"][:]) if sparse else grp["j"][:],
            dtype=str(grp.attrs["dtype"]),
            n=int(grp.attrs["n"]) if "n" in grp.attrs else len(grp["h"]),
            c=float(grp.attrs["c"]),
            h=grp["h"][:],
            version=int(version) if version is not None else FORMAT_VERSION,
        )
    return from_record(record)


def write(
    problem: Problem,
    path: str | Path,
    *,
    key: str = "problem",
    compression: str | None = "gzip",
    layout: LayoutOptions = "auto",
    overwrite: bool = False,
) -> None:
    """Serialise a Problem into an HDF5 group.

    Args:
        problem: The Problem to serialise.
        path: Destination file path.
        key: HDF5 group path within the file (e.g. `"experiment/run1"`).
        compression: Compression filter for array datasets; `None` disables it.
        layout: `"sparse"`, `"dense"` or `"auto"`.
        overwrite: If False (default), raise if `key` already exists in the file.
    """
    h5py = _import_h5py()
    record = to_record(problem, layout)

    with h5py.File(path, "a") as f:
        if key in f:
            if not overwrite:
                raise FileExistsError(
                    f"Key {key!r} already exists in {path}; pass overwrite=True to replace it."
                )
            del f[key]
        grp = f.create_group(key)
        if isinstance(record.j, tuple):
            src, dst, weight = record.j
            grp.create_dataset("src", data=src, compression=compression)
            grp.create_dataset("dst", data=dst, compression=compression)
            grp.create_dataset("weight", data=weight, compression=compression)
        else:
            grp.create_dataset("j", data=record.j, compression=compression)
        grp.create_dataset("h", data=record.h, compression=compression)
        grp.attrs["format_version"] = record.version
        grp.attrs["layout"] = record.layout
        grp.attrs["dtype"] = record.dtype
        grp.attrs["n"] = record.n
        grp.attrs["c"] = record.c
