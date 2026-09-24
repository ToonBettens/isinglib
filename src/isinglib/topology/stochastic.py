from __future__ import annotations

import numpy as np

from isinglib.dtypes import DEFAULT_INDEX_DTYPE
from isinglib.topology.topology import Topology

__all__ = (
    "barabasi_albert",
    "erdos_renyi",
    "watts_strogatz",
)


def erdos_renyi(
    n: int,
    p: float = 0.5,
    *,
    rng: np.random.Generator | None = None,
) -> Topology:
    """Erdős-Rényi G(n, p): each edge included independently with probability `p`.

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

    Grows from an initial clique of `m+1` nodes. Each new node attaches
    to `m` existing nodes with probability proportional to their degree.

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

    e = np.array(edges, dtype=DEFAULT_INDEX_DTYPE)
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
    e = np.array(sorted(edge_set), dtype=DEFAULT_INDEX_DTYPE)
    return Topology(n, e[:, 0], e[:, 1])
