from __future__ import annotations

import itertools

import numpy as np

from isinglib.dtypes import DEFAULT_INDEX_DTYPE, IndexArray
from isinglib.topology.topology import Topology

__all__ = (
    "chimera",
    "complete",
    "cycle",
    "grid",
    "king",
    "mobius_ladder",
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
    src = np.zeros(n - 1, dtype=DEFAULT_INDEX_DTYPE)
    dst = np.arange(1, n, dtype=DEFAULT_INDEX_DTYPE)
    return Topology(n, src, dst)


def path(n: int) -> Topology:
    """Path graph P_n: nodes connected in a line."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if n == 1:
        return Topology(1, [], [])
    src = np.arange(n - 1, dtype=DEFAULT_INDEX_DTYPE)
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
    edges = np.array(sorted(edge_set), dtype=DEFAULT_INDEX_DTYPE)
    return Topology(n, edges[:, 0], edges[:, 1])


def mobius_ladder(n: int) -> Topology:
    """Möbius ladder M_n: cycle C_n plus a rung from every node to its opposite.

    Args:
        n: Number of nodes (even, at least 4).
    """
    if n < 4 or n % 2 != 0:
        raise ValueError("n must be even and at least 4.")
    nodes = np.arange(n, dtype=DEFAULT_INDEX_DTYPE)
    half = nodes[: n // 2]
    src = np.concatenate([nodes, half])
    dst = np.concatenate([(nodes + 1) % n, half + n // 2])
    return Topology(n, src, dst)


def chimera(m: int, n: int | None = None, t: int = 4) -> Topology:
    """D-Wave Chimera graph C(m, n, t): an `m x n` grid of K_{t,t} unit cells.

    Each cell has a vertical and a horizontal shore of `t` nodes, fully
    connected to each other. Vertical nodes also couple to the same node in the
    cells above and below, horizontal nodes to those left and right.

    Args:
        m: Number of cell rows.
        n: Number of cell columns; defaults to `m`.
        t: Nodes per shore.
    """
    n = m if n is None else n
    if m <= 0 or n <= 0 or t <= 0:
        raise ValueError("m, n and t must be positive.")
    idx = np.arange(m * n * 2 * t, dtype=DEFAULT_INDEX_DTYPE).reshape(m, n, 2, t)
    vertical, horizontal = idx[:, :, 0, :], idx[:, :, 1, :]

    in_cell_src = np.repeat(vertical, t, axis=-1)
    in_cell_dst = np.tile(horizontal, (1, 1, t))

    src = np.concatenate([in_cell_src.ravel(), vertical[:-1].ravel(), horizontal[:, :-1].ravel()])
    dst = np.concatenate([in_cell_dst.ravel(), vertical[1:].ravel(), horizontal[:, 1:].ravel()])
    return Topology(m * n * 2 * t, src, dst)


def grid(*dims: int, periodic: bool = False) -> Topology:
    """Grid graph: nodes on a lattice of shape `dims`, coupled to their nearest neighbour along every axis.

    `grid(rows, cols)` is the 2-D grid, `grid(L, L, L)` the simple cubic
    lattice. Nodes are numbered row-major.

    Args:
        *dims: Nodes along each axis; one or more.
        periodic: If True, wrap every axis, giving every node degree `2d` (needs every side `>= 3`).
    """
    return _lattice(dims, np.eye(len(dims), dtype=DEFAULT_INDEX_DTYPE), periodic)


def king(*dims: int, periodic: bool = False) -> Topology:
    """King's graph: nodes on a lattice of shape `dims`, coupled to every node one step away diagonally too.

    Two nodes are coupled when they differ by at most 1 in every coordinate.
    8 for `king(rows, cols)`, 26 in 3-D. Nodes are numbered row-major.

    Args:
        *dims: Nodes along each axis; one or more.
        periodic: If True, wrap every axis, giving every node degree `2d` (needs every side `>= 3`).
    """
    steps = np.array(list(itertools.product((-1, 0, 1), repeat=len(dims))), dtype=DEFAULT_INDEX_DTYPE)
    first_nonzero = steps[np.arange(len(steps)), np.argmax(steps != 0, axis=1)]
    return _lattice(dims, steps[first_nonzero == 1], periodic)


def _lattice(dims: tuple[int, ...], steps: IndexArray, periodic: bool) -> Topology:
    """Couple every node of a row-major lattice of shape `dims` to the node each of `steps` away."""
    if not dims or any(size <= 0 for size in dims):
        raise ValueError("a lattice needs one or more positive dimensions.")
    if periodic and any(size < 3 for size in dims):
        raise ValueError("a periodic lattice needs every side >= 3.")
    shape = np.array(dims)[:, None]
    coords = np.indices(dims).reshape(len(dims), -1)
    nodes = np.arange(coords.shape[1], dtype=DEFAULT_INDEX_DTYPE)
    src, dst = [], []
    for step in steps:
        target = coords + step[:, None]
        if periodic:
            keep = np.ones(nodes.size, dtype=bool)
            target %= shape
        else:
            keep = np.all((target >= 0) & (target < shape), axis=0)
        src.append(nodes[keep])
        dst.append(np.ravel_multi_index(tuple(target[:, keep]), dims))
    return Topology(nodes.size, np.concatenate(src), np.concatenate(dst))
