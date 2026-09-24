from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from isinglib.dtypes import ensure_index_array
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
    """Load a Problem from a `.json` file.

    Args:
        path: Path to the `.json` file.
    """
    with Path(path).open("r") as f:
        data = json.load(f)

    version = data.get("format_version")
    check_version(version)

    layout = data.get("layout", "dense")
    if layout == "sparse":
        triples = data["edges"]
        src = ensure_index_array([e[0] for e in triples])
        dst = ensure_index_array([e[1] for e in triples])
        weight = np.array([e[2] for e in triples], dtype=np.dtype(data["dtype"]))
        stored = (src, dst, weight)
    else:
        stored = np.array(data["j"])

    record = Record(
        j=stored,
        dtype=data["dtype"],
        n=int(data["n"]),
        c=float(data.get("c", 0.0)),
        h=np.array(data["h"]),
        version=version if version is not None else FORMAT_VERSION,
    )
    return from_record(record)


def write(
    problem: Problem,
    path: str | Path,
    *,
    layout: LayoutOptions = "auto",
    indent: int = 2,
    overwrite: bool = False,
) -> None:
    """Serialise a Problem to a JSON file.

    Args:
        problem: The Problem to serialise.
        path: Destination file path.
        layout: `"sparse"`, `"dense"` or `"auto"`.
        indent: JSON indentation level.
        overwrite: If False (default), raise if `path` already exists.
    """
    path = Path(path)
    if not overwrite and path.exists():
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it.")
    record = to_record(problem, layout)
    data: dict[str, Any] = {
        "format_version": record.version,
        "layout": record.layout,
        "dtype": record.dtype,
        "n": record.n,
        "c": record.c,
    }
    if isinstance(record.j, tuple):
        src, dst, weight = record.j
        data["edges"] = [[int(u), int(v), float(w)] for u, v, w in zip(src, dst, weight, strict=True)]
    else:
        data["j"] = record.j.tolist()
    data["h"] = record.h.tolist()

    text = json.dumps(data, indent=indent)
    if isinstance(record.j, tuple):
        text = _compact_edges(text, indent)
    path.write_text(text + "\n")


def _compact_edges(text: str, indent: int) -> str:
    """Put each `[src, dst, weight]` on one line. """
    close = "\n" + " " * indent + "]"
    start = text.index('"edges": [')
    end = text.index(close, start) + len(close)
    block = re.sub(r"\[\s+([^][]+?)\s+\]", lambda m: "[" + " ".join(m.group(1).split()) + "]", text[start:end])
    return text[:start] + block + text[end:]
