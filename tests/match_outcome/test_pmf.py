"""Rung-1 core primitives (TF-53 Task 2): exact Poisson-binomial PMF + outcome simplex."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from silly_kicks.match_outcome import goal_count_pmf, match_outcome_probabilities


def _brute_force_pmf(xgs: list[float]) -> np.ndarray:
    n = len(xgs)
    pmf = np.zeros(n + 1)
    for combo in product([0, 1], repeat=n):
        p = 1.0
        for bit, xg in zip(combo, xgs, strict=True):
            p *= xg if bit else (1 - xg)
        pmf[sum(combo)] += p
    return pmf


@pytest.mark.parametrize("seed", range(8))
def test_poisson_binomial_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 13))
    xgs = rng.uniform(0.01, 0.95, size=n).tolist()
    np.testing.assert_allclose(goal_count_pmf(xgs), _brute_force_pmf(xgs), atol=1e-12)


def test_pmf_edges():
    np.testing.assert_array_equal(goal_count_pmf([]), np.array([1.0]))
    np.testing.assert_allclose(goal_count_pmf([0.3]), np.array([0.7, 0.3]))
    assert abs(goal_count_pmf([0.2, 0.5, 0.8]).sum() - 1.0) < 1e-12


def test_outcome_probs_sum_to_one_and_symmetric():
    home = goal_count_pmf([0.4, 0.2, 0.1])
    away = goal_count_pmf([0.3, 0.3])
    ph, pd_, pa = match_outcome_probabilities(home, away)
    assert abs(ph + pd_ + pa - 1.0) < 1e-12
    # symmetric inputs -> symmetric outputs; draw = sum of matched-score products
    s = goal_count_pmf([0.5, 0.25])
    ph2, pd2, pa2 = match_outcome_probabilities(s, s)
    assert abs(ph2 - pa2) < 1e-12
    assert abs(pd2 - float((s * s).sum())) < 1e-12


def test_single_shot_each():
    # one 0.5 shot each: draw = 0.5*0.5 (0-0) + 0.5*0.5 (1-1) = 0.5; win = loss = 0.25
    ph, pd_, pa = match_outcome_probabilities(goal_count_pmf([0.5]), goal_count_pmf([0.5]))
    assert abs(pd_ - 0.5) < 1e-12 and abs(ph - 0.25) < 1e-12 and abs(pa - 0.25) < 1e-12
