from __future__ import annotations

import numpy as np

from isinglib.topology.topology import Topology

__all__ = (
    "barabasi_albert",
    "complete",
    "cycle",
    "erdos_renyi",
    "grid",
    "king",
    "path",
    "ring",
    "star",
    "watts_strogatz",
)


# ------------------------------------------------------------------
# Regular graph topologies
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Stochastic graph topologies
# ------------------------------------------------------------------

def erdos_renyi(
    n: int,
    p: float = 0.5,
    *,
    rng: np.random.Generator | None = None,
) -> Topology:
    """Erdős-Rényi G(n, p): each edge included independently with probability ``p``.

    Args:
        n: Number of nodes.
        p: Edge inclusion probability in [0, 1].
        rng: Random generator or seed for reproducibility.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1].")
    rng = np.random.default_rng(rng)
    iu, ju = np.triu_indices(n, k=1)
    mask = rng.random(iu.size) < p
    return Topology(n, iu[mask], ju[mask])


def barabasi_albert(
    n: int,
    m: int = 2,
    *,
    rng: np.random.Generator | None = None,
) -> Topology:
    """Barabási-Albert preferential-attachment graph.

    Grows from an initial clique of ``m+1`` nodes; each new node attaches
    to ``m`` existing nodes with probability proportional to their degree.

    Args:
        n: Total number of nodes.
        m: Number of edges added per new node (1 ≤ m < n).
        rng: Random generator or seed for reproducibility.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if not (1 <= m < n):
        raise ValueError("m must satisfy 1 <= m < n.")
    rng = np.random.default_rng(rng)

    edges: list[tuple[int, int]] = [(u, v) for u in range(m) for v in range(u + 1, m + 1)]
    nbrs: list[set[int]] = [set() for _ in range(n)]
    for u, v in edges:
        nbrs[u].add(v)
        nbrs[v].add(u)

    for new in range(m + 1, n):
        targets: set[int] = set()
        while len(targets) < m:
            u, v = edges[rng.integers(len(edges))]
            chosen = u if rng.random() < 0.5 else v
            if chosen == new or chosen in targets or chosen in nbrs[new]:
                continue
            a, b = (new, chosen) if new < chosen else (chosen, new)
            edges.append((a, b))
            nbrs[new].add(chosen)
            nbrs[chosen].add(new)
            targets.add(chosen)

    e = np.array(edges, dtype=np.int32)
    return Topology(n, e[:, 0], e[:, 1])


def watts_strogatz(
    n: int,
    k: int = 4,
    p: float = 0.1,
    *,
    rng: np.random.Generator | None = None,
) -> Topology:
    """Watts-Strogatz small-world graph: ring lattice with random rewiring.

    Args:
        n: Number of nodes.
        k: Initial degree of each node in the ring lattice (must be even).
        p: Probability of rewiring each edge.
        rng: Random generator or seed for reproducibility.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if k % 2 != 0 or not (0 < k < n):
        raise ValueError("k must be even and satisfy 0 < k < n.")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1].")
    rng = np.random.default_rng(rng)

    edge_set: set[tuple[int, int]] = set()
    nbrs: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for d in range(1, k // 2 + 1):
            j = (i + d) % n
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in edge_set:
                edge_set.add((a, b))
                nbrs[a].add(b)
                nbrs[b].add(a)

    for i in range(n):
        for d in range(1, k // 2 + 1):
            j = (i + d) % n
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in edge_set or rng.random() >= p:
                continue
            edge_set.remove((a, b))
            nbrs[a].discard(b)
            nbrs[b].discard(a)
            forbidden = nbrs[i] | {i}
            candidates = np.setdiff1d(np.arange(n), list(forbidden))
            if candidates.size == 0:
                edge_set.add((a, b))
                nbrs[a].add(b)
                nbrs[b].add(a)
                continue
            new_j = int(rng.choice(candidates))
            c2, d2 = (i, new_j) if i < new_j else (new_j, i)
            if (c2, d2) not in edge_set:
                edge_set.add((c2, d2))
                nbrs[c2].add(d2)
                nbrs[d2].add(c2)
            else:
                edge_set.add((a, b))
                nbrs[a].add(b)
                nbrs[b].add(a)

    if not edge_set:
        return Topology(n, [], [])
    e = np.array(sorted(edge_set), dtype=np.int32)
    return Topology(n, e[:, 0], e[:, 1])
