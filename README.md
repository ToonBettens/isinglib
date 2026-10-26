# isinglib

A pure-Python library for research on Ising machines. Every instance is one
immutable `Problem`, and every solver is a plain function taking a `Problem`
and returning a `Solution`.

```
E(s) = -0.5 sᵀ j s - hᵀ s + c,    s ∈ {-1, +1}ⁿ
```

`j` is symmetric with a zero diagonal, `h` holds the fields, and `c` is an offset: the
mean energy over random states.

| Module | What it holds |
|---|---|
| `Problem` | an instance `(j, h, c)`: energy, content fingerprint, normalized residual |
| `topology` | complete, star, path, cycle, ring, Möbius ladder, chimera; grid and king lattices in any dimension, open or periodic; Erdős–Rényi, Barabási–Albert, Watts–Strogatz |
| `generators` | couplings on a topology: constant, uniform, Gaussian, ±1, a choice of values; Sherrington–Kirkpatrick and Edwards–Anderson spin glasses; Mattis planted ground states |
| `solvers` | exact: exhaustive, branch and bound · local: steepest descent, first improvement · stochastic: simulated annealing, parallel tempering, tabu search · dynamical: simulated bifurcation, ballistic and discrete · `run_batch` for repeated runs |
| `mappings` | max-cut and QUBO to and from Ising; an auxiliary spin that folds `h` into couplings |
| `io` | JSON, NPZ and HDF5 behind one `read` / `write` |

Literature benchmarks ship separately in
[isinglib-benchmarks](https://github.com/ToonBettens/isinglib-benchmarks): BiqMac and
G-set, with their best known energies and, where known, the states that reach them.

## A tour

**Build an instance and solve it four ways.** Each solver reports what it found and
how long it took, plus its own diagnostics.

```python
from isinglib import generators
from isinglib.solvers import parallel_tempering, simulated_annealing, simulated_bifurcation, tabu_search

problem = generators.edwards_anderson(6, rng=0)  # 3-D ±1 spin glass, 216 spins
for solver in (simulated_annealing, parallel_tempering, tabu_search, simulated_bifurcation):
    solution = solver(problem, rng=0)
    print(f"{solution.solver:22} E = {solution.energy:7.1f}  in {solution.elapsed_s:.2f} s")
```

**Repeat runs** on independent random streams, reproducible from one seed:

```python
from isinglib.solvers import run_batch

runs = run_batch(simulated_annealing, problem, 20, rng=0, n_sweeps=300)
energies = [run.energy for run in runs]
```

**Check a solver against a known answer.** The Mattis model plants a ground state, and
the exact solvers confirm it:

```python
import numpy as np
from isinglib.solvers import branch_and_bound
from isinglib.states import random_spins
from isinglib.topology import erdos_renyi

state = random_spins(20, rng=1)
planted = generators.mattis(erdos_renyi(20, p=0.3, rng=1), state).replace(h=0.1 * state)
assert np.array_equal(branch_and_bound(planted).spins, state)
```

**Measure quality on a literature benchmark.** The normalized residual is 0 at the best
known energy and 1 at a random state, so it means the same on every instance, unlike
the usual gap, which changes when a constant is added to the energy:

```python
import isinglib_benchmarks as bm

g05 = bm.load("biqmac", "rudy/g05_100.0")  # dense max-cut, proven optimum
best = bm.best_energy("biqmac", "rudy/g05_100.0")
residual = g05.normalized_residual(tabu_search(g05, rng=0).energy, best)
```

**Solve other problem classes.** Mappings encode them exactly, so energies carry over:

```python
from isinglib.mappings import maxcut, qubo

rng = np.random.default_rng(2)
w = np.triu(rng.integers(0, 2, (30, 30)), 1).astype(float)
w += w.T  # an unweighted graph
solution = tabu_search(maxcut.encode(w), rng=0)
assert maxcut.objective(w, maxcut.decode(solution.spins)) == -solution.energy  # the cut size

q = np.triu(rng.normal(size=(12, 12)))  # minimize xᵀQx over x ∈ {0, 1}¹²
x = qubo.decode(branch_and_bound(qubo.encode(q)).spins)
```

**Know a solver's limits.** Simulated bifurcation's reference step `dt = 1.25` is
unstable where the couplings share a sign, as in unweighted max-cut;
`max_stable_dt` computes each instance's limit from its spectrum:

```python
from isinglib.solvers import max_stable_dt

limit = max_stable_dt(g05)  # 0.94: below the default
solution = simulated_bifurcation(g05, dt=0.9 * limit, rng=0)
```

**Store instances** in a portable format; the extension picks the backend:

```python
from isinglib import io

io.write(problem, "ea6.npz")
assert io.read("ea6.npz") == problem  # problems compare by content
```

## Development

```bash
uv sync --all-extras
uv run ruff format
uv run ruff check
uv run ty check
uv run pytest
```
