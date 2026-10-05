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

Generators turn parameters, usually a `Topology`, into a `Problem`. Solvers
are plain functions: `Problem` in, `Solution` out.

```python
import numpy as np
from isinglib import Problem, generators
from isinglib.solvers import exhaustive, simulated_annealing, tabu_search
from isinglib.states import random_spins
from isinglib.topology import erdos_renyi, grid, king

rng = np.random.default_rng(0)
```

**The Sherrington-Kirkpatrick model**, the canonical spin glass:

```python
problem = generators.sherrington_kirkpatrick(100, rng=rng)
solution = simulated_annealing(problem, rng=rng)
```

**Random couplings on a random graph**:

```python
problem = generators.uniform(erdos_renyi(20, p=0.3, rng=rng), -10.0, 10.0, rng=rng)
solution = simulated_annealing(problem, rng=rng)
```

**The 3-D Edwards-Anderson spin glass**, the classic hard short-range benchmark:

```python
problem = generators.edwards_anderson(6, rng=rng)
solution = simulated_annealing(problem, rng=rng)
```

**A ±J spin glass on a torus, with a random bias**:

```python
lattice = grid(4, 4, periodic=True)
problem = generators.pm_one(lattice, rng=rng).replace(h=rng.normal(0.0, 0.1, lattice.n))
solution = tabu_search(problem, rng=rng)
```

**A ferromagnet with a uniform bias**:

```python
problem = generators.constant(king(3, 3), 1.0).replace(h=0.5)
solution = exhaustive(problem)
```

**Custom couplings**: `Topology.fill` takes a scalar, one value per edge, or
a callable handed the edge count, and `Problem` takes the arrays directly:

```python
problem = Problem(grid(3, 3).fill(lambda m: rng.exponential(1.0, m)), h=0.1)
lattice = grid(3, 3)
problem = Problem(lattice.fill(np.linspace(0.1, 1.0, lattice.num_edges)))
```

**Plant a known ground state** with the Mattis model, whose couplings all point
towards a chosen state, for checking a solver against the right answer:

```python
topology = erdos_renyi(9, p=0.4, rng=rng)
state = random_spins(9, rng=rng)
problem = generators.mattis(topology, state)
problem = problem.replace(h=0.1 * state)  # makes `state` the unique ground state
solution = exhaustive(problem)
assert np.array_equal(solution.spins, state)
```

## Development

```bash
uv sync --all-extras
uv run ruff format
uv run ruff check
uv run ty check
uv run pytest
```
