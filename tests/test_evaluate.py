from __future__ import annotations

import numpy as np
import pytest

from isinglib import Problem, _evaluate


def _flipped(s: np.ndarray, i: int) -> np.ndarray:
    out = s.copy()
    out[i] = -out[i]
    return out


@pytest.fixture
def case(small_problem: Problem) -> tuple[Problem, np.ndarray]:
    """A problem and a spin state to evaluate it at."""
    rng = np.random.default_rng(0)
    return small_problem, np.where(rng.random(small_problem.n) < 0.5, 1.0, -1.0)


# ------------------------------------------------------------------
# effective_field
# ------------------------------------------------------------------


def test_effective_field_matches_the_explicit_product(case: tuple[Problem, np.ndarray]) -> None:
    p, s = case
    assert np.allclose(_evaluate.effective_field(p.j, p.h, s), p.j @ s + p.h)


def test_effective_field_at_one_index_matches_the_full_vector(
    case: tuple[Problem, np.ndarray],
) -> None:
    p, s = case
    full = _evaluate.effective_field(p.j, p.h, s)
    for i in range(p.n):
        single = _evaluate.effective_field(p.j, p.h, s, i=i)
        assert isinstance(single, float)
        assert single == pytest.approx(full[i])


# ------------------------------------------------------------------
# energy
# ------------------------------------------------------------------


def test_energy_matches_the_hamiltonian(case: tuple[Problem, np.ndarray]) -> None:
    p, s = case
    expected = -0.5 * float(s @ p.j @ s) - float(p.h @ s) + p.c
    assert _evaluate.energy(p.j, p.h, p.c, s) == pytest.approx(expected)


def test_energy_reuses_a_supplied_effective_field(case: tuple[Problem, np.ndarray]) -> None:
    p, s = case
    h_eff = _evaluate.effective_field(p.j, p.h, s)
    assert _evaluate.energy(p.j, p.h, p.c, s, h_eff=h_eff) == pytest.approx(
        _evaluate.energy(p.j, p.h, p.c, s)
    )


# ------------------------------------------------------------------
# spin_flip_energy_update
# ------------------------------------------------------------------


def test_spin_flip_energy_update_equals_the_recomputed_difference(
    case: tuple[Problem, np.ndarray],
) -> None:
    p, s = case
    h_eff = _evaluate.effective_field(p.j, p.h, s)
    deltas = _evaluate.spin_flip_energy_update(s, h_eff)
    for i in range(p.n):
        assert deltas[i] == pytest.approx(p.energy(_flipped(s, i)) - p.energy(s))


def test_spin_flip_energy_update_at_one_index_matches_the_full_vector(
    case: tuple[Problem, np.ndarray],
) -> None:
    p, s = case
    h_eff = _evaluate.effective_field(p.j, p.h, s)
    full = _evaluate.spin_flip_energy_update(s, h_eff)
    for i in range(p.n):
        single = _evaluate.spin_flip_energy_update(s, h_eff, i=i)
        assert isinstance(single, float)
        assert single == pytest.approx(full[i])


# ------------------------------------------------------------------
# spin_flip_effective_field_update
# ------------------------------------------------------------------


def test_spin_flip_effective_field_update_reaches_the_flipped_field(
    case: tuple[Problem, np.ndarray],
) -> None:
    p, s = case
    h_eff = _evaluate.effective_field(p.j, p.h, s)
    for i in range(p.n):
        delta = _evaluate.spin_flip_effective_field_update(p.j, s, i)
        expected = _evaluate.effective_field(p.j, p.h, _flipped(s, i))
        assert np.allclose(h_eff + delta, expected)


def test_incremental_updates_track_a_full_recomputation(
    case: tuple[Problem, np.ndarray],
) -> None:
    p, s = case
    state = s.copy()
    h_eff = _evaluate.effective_field(p.j, p.h, state)
    running = _evaluate.energy(p.j, p.h, p.c, state, h_eff=h_eff)

    for i in range(p.n):
        running += _evaluate.spin_flip_energy_update(state, h_eff, i=i)
        h_eff = h_eff + _evaluate.spin_flip_effective_field_update(p.j, state, i)
        state[i] = -state[i]

        assert np.allclose(h_eff, _evaluate.effective_field(p.j, p.h, state))
        assert running == pytest.approx(p.energy(state))
