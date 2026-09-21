import numpy as np

from silly_kicks.win_probability._chain import (
    expected_total_goals,
    outcome_from_start,
    step_matrix,
    win_prob_table,
)


def _const_haz(p):
    return lambda d, m: p


def test_step_matrix_rows_sum_to_one():
    K = 5
    ph = np.full(2 * K + 1, 0.02)
    pa = np.full(2 * K + 1, 0.02)
    M = step_matrix(ph, pa, K)
    assert M.shape == (2 * K + 1, 2 * K + 1)
    assert np.allclose(M.sum(axis=1), 1.0)


def test_symmetric_teams_give_symmetric_winloss():
    w, draw, loss = outcome_from_start(_const_haz(0.03), _const_haz(0.03), start_diff=0, n_intervals=20, K=8)
    assert abs(w - loss) < 1e-12
    assert abs(w + draw + loss - 1.0) < 1e-12


def test_dead_state_leverage_zero():
    tab = win_prob_table(_const_haz(0.03), _const_haz(0.03), n_intervals=0, K=8)
    K = 8
    lead_idx = K + 3  # diff +3
    assert tab[lead_idx, 0] == 1.0
    assert tab[lead_idx + 1, 0] - tab[lead_idx, 0] == 0.0


def test_leverage_is_table_difference():
    K = 8
    m = 30
    d = 0
    tab = win_prob_table(_const_haz(0.03), _const_haz(0.03), n_intervals=m, K=K)
    lev = tab[K + d + 1, m] - tab[K + d, m]
    w1, _, _ = outcome_from_start(_const_haz(0.03), _const_haz(0.03), start_diff=d + 1, n_intervals=m, K=K)
    w0, _, _ = outcome_from_start(_const_haz(0.03), _const_haz(0.03), start_diff=d, n_intervals=m, K=K)
    assert abs(lev - (w1 - w0)) < 1e-12


def test_monotone_kernel_gives_nonnegative_leverage():
    # constant (score-independent) hazard IS stochastically monotone -> leverage >= 0 everywhere
    tab = win_prob_table(_const_haz(0.03), _const_haz(0.03), n_intervals=40, K=10)
    diffs = tab[1:, :] - tab[:-1, :]
    assert (diffs >= -1e-12).all()


def test_per_step_mass_conserved():
    # §9.1 mass-conservation, per-step literal (TF63-PLAN-07): every transition row sums to 1.
    K = 8
    pm = np.full(2 * K + 1, 0.03)
    M = step_matrix(pm, pm, K)
    assert np.allclose(M.sum(axis=1), 1.0)


def test_lattice_pad_K_absorbs_negligible_mass():
    # K sufficiency (mis-set-K guard, TF63-PLAN-02): at the realistic ~0.015/team/min rate a full-match
    # roll from 0-0 leaves negligible mass at the K=10 pad (~6 sigma out); a too-small K (e.g. 3) fails.
    K = 10
    dist = np.zeros(2 * K + 1)
    dist[K] = 1.0
    pm = np.full(2 * K + 1, 0.015)
    M = step_matrix(pm, pm, K)
    for _ in range(95):
        dist = M @ dist
    assert dist[0] + dist[-1] < 1e-4


def test_expected_total_goals_matches_hazard_sum():
    # mis-scaled-hazard guard (TF63-PLAN-02): expected TOTAL goals ~ Σ (p_home+p_away).
    eg = expected_total_goals(_const_haz(0.015), _const_haz(0.015), n_intervals=90, K=10)
    assert abs(eg - 90 * (0.015 + 0.015)) < 0.2  # ≈2.7; edge/coupling correction small
