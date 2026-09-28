# isinglib

A pure-Python library for research on Ising machines It provides problem
generation and a set of solvers over a single immutable `Problem` type, so that
different solving approaches (combinatorial and continuous solvers) can be
compared on the same footing.

Currently implemented: the `Problem` and `Topology` types, graph generators,
and nine solvers: exact (exhaustive, branch and bound), local search (steepest
descent, first improvement), stochastic search (simulated annealing, tabu
search, parallel tempering) and simulated bifurcation (ballistic, discrete).
Continuous solvers, benchmark loaders, and landscape/result analysis tools are
planned but not yet built.

## Energy convention

```
E(s) = -0.5 * s^T j s - h^T s + c
```

`j` is the symmetric, zero-diagonal coupling matrix; `h` is the bias vector;
`c` is a scalar offset. Spin states take values in {-1, +1}.

## Examples

Solvers are plain functions: `Problem` in, `Solution` out.

```python
import numpy as np
from isinglib import Problem
from isinglib.generators import planted_solution
from isinglib.solvers import exhaustive, simulated_annealing, tabu_search
from isinglib.states import random_spins
from isinglib.topology import erdos_renyi, fillers, grid

rng = np.random.default_rng(0)
```

**Build a `Problem` directly from arrays**, when you already have the couplings:

```python
j = np.array([[0.0, 1.0], [1.0, 0.0]])
h = np.array([0.5, -0.5])
problem = Problem(j, h)
solution = exhaustive(problem)
```

**Combine a `Topology` with a filler** to get random couplings on a random graph:

```python
topology = erdos_renyi(20, p=0.3, rng=rng)
j = topology.fill(fillers.uniform(-1.0, 1.0, rng=rng))
problem = Problem(j)
solution = simulated_annealing(problem, rng=rng)
```

**Or on a regular lattice**, with a fixed-magnitude, random-sign filler:

```python
lattice = grid(4, 4, periodic=True)
j = lattice.fill(fillers.pm_one(rng=rng))
problem = Problem(j)
solution = tabu_search(problem, rng=rng)
```

**Plant a known ground state**, for checking a solver against the right answer:

```python
planted = random_spins(9, rng=rng)
problem = planted_solution(
    erdos_renyi(9, p=0.4, rng=rng),
    coupling=fillers.uniform(0.1, 1.0, rng=rng),
    bias=0.1,
    planted=planted,
)
solution = exhaustive(problem)
assert np.isclose(solution.energy, problem.energy(planted))
```

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check
uv run ty check
```
