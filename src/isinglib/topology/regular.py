from __future__ import annotations

import numpy as np

from isinglib.topology.topology import Topology

__all__ = (
    "complete",
    "cycle",
    "grid",
    "king",
    "path",
    "ring",
    "star",
)


def complete(n: int) -> Topology:
    """Complete graph K_n: every pair of nodes is connected."""
    if n <= 0:
        raise ValueError("n must be positive.")
    iu, ju = np.triu_indices(n, k=1)
    return Topology(n, iu, ju)


def star(n: int) -> Topology:
    """Star graph S_n: hub at node 0 connected to all other nodes."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if n == 1:
        return Topology(1, [], [])
    src = np.zeros(n - 1, dtype=np.int32)
    dst = np.arange(1, n, dtype=np.int32)
    return Topology(n, src, dst)


def path(n: int) -> Topology:
    """Path graph P_n: nodes connected in a line."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if n == 1:
        return Topology(1, [], [])
    src = np.arange(n - 1, dtype=np.int32)
    return Topology(n, src, src + 1)


def cycle(n: int) -> Topology:
    """Cycle graph C_n: closed path where every node has degree 2."""
    return ring(n, k=2)


def ring(n: int, k: int = 2) -> Topology:
    """k-regular ring lattice: each node connected to k/2 neighbours on each side.

    Args:
        n: Number of nodes.
        k: Degree of each node (must be even).
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if k % 2 != 0:
        raise ValueError("k must be even.")
    if n == 1 or k == 0:
        return Topology(n, [], [])
    edge_set: set[tuple[int, int]] = set()
    for i in range(n):
        for d in range(1, k // 2 + 1):
            a, b = i, (i + d) % n
            edge_set.add((min(a, b), max(a, b)))
    edges = np.array(sorted(edge_set), dtype=np.int32)
    return Topology(n, edges[:, 0], edges[:, 1])


def grid(rows: int, cols: int, *, periodic: bool = False) -> Topology:
    """2-D grid graph with 4-connectivity (von Neumann neighbourhood).

    Args:
        rows: Number of rows.
        cols: Number of columns.
        periodic: If True, wrap edges along both axes (toroidal boundary).
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive.")
    n = rows * cols

    def idx(r: int, c: int) -> int:
        return r * cols + c

    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            i = idx(r, c)
            if periodic or c + 1 < cols:
                j = idx(r, (c + 1) % cols)
                edges.append((min(i, j), max(i, j)))
            if periodic or r + 1 < rows:
                j = idx((r + 1) % rows, c)
                edges.append((min(i, j), max(i, j)))

    if not edges:
        return Topology(n, [], [])
    e = np.array(edges, dtype=np.int32)
    return Topology(n, e[:, 0], e[:, 1])


def king(rows: int, cols: int, *, periodic: bool = False) -> Topology:
    """2-D grid graph with 8-connectivity (Moore neighbourhood).

    Args:
        rows: Number of rows.
        cols: Number of columns.
        periodic: If True, wrap edges along both axes (toroidal boundary).
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive.")
    n = rows * cols

    def idx(r: int, c: int) -> int:
        return r * cols + c

    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            i = idx(r, c)
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                nr, nc = (r + dr) % rows, (c + dc) % cols
                if periodic or (0 <= r + dr < rows and 0 <= c + dc < cols):
                    j = idx(nr, nc)
                    edges.append((min(i, j), max(i, j)))

    if not edges:
        return Topology(n, [], [])
    e = np.array(edges, dtype=np.int32)
    return Topology(n, e[:, 0], e[:, 1])
