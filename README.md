# isinglib

A pure-Python library for research on Ising machines It provides problem
generation and a set of solvers over a single immutable `Problem` type, so that
different solving approaches (combinatorial and continuous solvers) can be
compared on the same footing.

Currently implemented: the `Problem` and `Topology` types, graph generators,
and four algorithmic solvers (exhaustive, greedy, simulated annealing, tabu
search). Continuous solvers, benchmark loaders, and landscape/result analysis
tools are planned but not yet built.

## Energy convention

```
E(s) = -0.5 * s^T j s - h^T s + c
```

`j` is the symmetric, zero-diagonal coupling matrix; `h` is the bias vector;
`c` is a scalar offset. Spin states take values in {-1, +1}.

## Example

```python
import numpy as np
from isinglib import Problem, SimulatedAnnealingSolver
from isinglib.topology import erdos_renyi

rng = np.random.default_rng(0)

# Build a random graph, then attach random couplings to get a Problem.
topology = erdos_renyi(20, p=0.3, rng=rng)
problem = Problem.from_topology(topology, coupling=rng.standard_normal)

solution = SimulatedAnnealingSolver(rng=rng).solve(problem)
print(solution.spins)
print(solution.energy)
```

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check
uv run ty check
```
