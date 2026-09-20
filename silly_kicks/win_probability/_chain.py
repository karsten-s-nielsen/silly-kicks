"""Forward Markov chain on ``score_diff`` + the backward-DP ``Pwin`` table.

The self-contained in-game core (approach (b)): the interval-hazard GLM supplies per-minute scoring
probabilities; this module propagates a distribution over ``score_diff`` and reads the win/draw/loss
outcome directly. It replaces -- does NOT reuse -- the TF-53 ``goal_count_pmf`` convolution and
``match_outcome_probabilities`` simplex (team independence is exactly the approximation the chain
removes). Hazards are callables ``(score_diff:int, minutes_remaining:float) -> float``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Hazard = Callable[[int, float], float]


def _diffs_axis(K: int) -> np.ndarray:
    return np.arange(-K, K + 1)


def step_matrix(p_home: np.ndarray, p_away: np.ndarray, K: int) -> np.ndarray:
    """Row-stochastic ``score_diff`` transition on ``[-K, K]`` for one interval.

    ``+1`` (home scores, away doesn't) w.p. ``p_home*(1-p_away)``; ``-1`` (away scores) w.p.
    ``p_away*(1-p_home)``; ``0`` (neither, or both -- a net-zero step) otherwise. Mass that would leave
    the lattice is absorbed at the edge (negligible, pinned by the edge-mass / expected-goals gates).
    """
    n = 2 * K + 1
    up = p_home * (1.0 - p_away)
    dn = p_away * (1.0 - p_home)
    stay = 1.0 - up - dn
    M = np.zeros((n, n), dtype="float64")
    idx = np.arange(n)
    M[idx, idx] = stay
    M[idx[:-1], idx[:-1] + 1] = up[:-1]
    M[idx[1:], idx[1:] - 1] = dn[1:]
    M[0, 0] += dn[0]  # absorbing pad edges
    M[-1, -1] += up[-1]
    return M


def win_prob_table(hazard_home: Hazard, hazard_away: Hazard, *, n_intervals: int, K: int) -> np.ndarray:
    """``Pwin[d_idx, m] = P(final score_diff > 0 | current diff, m intervals remain)`` (backward DP)."""
    diffs = _diffs_axis(K)
    Pwin = np.zeros((2 * K + 1, n_intervals + 1), dtype="float64")
    Pwin[:, 0] = (diffs > 0).astype("float64")  # terminal: already decided
    for m in range(1, n_intervals + 1):
        ph = np.array([hazard_home(int(d), float(m)) for d in diffs])
        pa = np.array([hazard_away(int(d), float(m)) for d in diffs])
        M = step_matrix(ph, pa, K)
        Pwin[:, m] = M @ Pwin[:, m - 1]
    return Pwin


def outcome_table(
    hazard_home: Hazard, hazard_away: Hazard, *, n_intervals: int, K: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(Pwin, Pdraw, Ploss)`` each ``(2K+1, n_intervals+1)`` by backward DP.

    ``P*[d_idx, m]`` = probability of that outcome given the current ``score_diff`` and ``m`` intervals
    remaining. Backs the per-match O(1) per-action lookup (SPEC-12): one table per
    ``(base_strength, home, man_advantage)`` value, indexed by ``(score_diff, minutes_remaining)``.
    """
    diffs = _diffs_axis(K)
    Pw = np.zeros((2 * K + 1, n_intervals + 1), dtype="float64")
    Pd = np.zeros_like(Pw)
    Pl = np.zeros_like(Pw)
    Pw[:, 0] = (diffs > 0).astype("float64")
    Pd[:, 0] = (diffs == 0).astype("float64")
    Pl[:, 0] = (diffs < 0).astype("float64")
    for m in range(1, n_intervals + 1):
        ph = np.array([hazard_home(int(d), float(m)) for d in diffs])
        pa = np.array([hazard_away(int(d), float(m)) for d in diffs])
        M = step_matrix(ph, pa, K)
        Pw[:, m] = M @ Pw[:, m - 1]
        Pd[:, m] = M @ Pd[:, m - 1]
        Pl[:, m] = M @ Pl[:, m - 1]
    return Pw, Pd, Pl


def _outcome_planes(hazard_home: Hazard, hazard_away: Hazard, *, n_intervals: int, K: int):
    diffs = _diffs_axis(K)
    W = (diffs > 0).astype("float64")
    D = (diffs == 0).astype("float64")
    L = (diffs < 0).astype("float64")
    for m in range(1, n_intervals + 1):
        ph = np.array([hazard_home(int(d), float(m)) for d in diffs])
        pa = np.array([hazard_away(int(d), float(m)) for d in diffs])
        M = step_matrix(ph, pa, K)
        W, D, L = M @ W, M @ D, M @ L
    return W, D, L


def outcome_from_start(
    hazard_home: Hazard, hazard_away: Hazard, *, start_diff: int, n_intervals: int, K: int
) -> tuple[float, float, float]:
    """Single ``(win, draw, loss)`` roll from ``start_diff`` (special case of the table)."""
    W, D, L = _outcome_planes(hazard_home, hazard_away, n_intervals=n_intervals, K=K)
    i = start_diff + K
    return float(W[i]), float(D[i]), float(L[i])


def expected_total_goals(
    hazard_home: Hazard, hazard_away: Hazard, *, n_intervals: int, K: int, start_diff: int = 0
) -> float:
    """Expected TOTAL goals over the match (hazard-path integral).

    The ``score_diff`` chain does not track total goals (a both-score minute is net-0), so this
    forward-propagates the diff distribution and accumulates ``Σ_m Σ_d dist_m[d]·(p_home+p_away)``.
    Backs the expected-goals gate (mis-scaled-hazard guard).
    """
    diffs = _diffs_axis(K)
    dist = np.zeros(2 * K + 1)
    dist[start_diff + K] = 1.0
    total = 0.0
    for m in range(n_intervals, 0, -1):  # m = intervals remaining at this step
        ph = np.array([hazard_home(int(d), float(m)) for d in diffs])
        pa = np.array([hazard_away(int(d), float(m)) for d in diffs])
        total += float(dist @ (ph + pa))
        dist = step_matrix(ph, pa, K) @ dist
    return total
