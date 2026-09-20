import numpy as np
import pandas as pd

from silly_kicks.spadl import config as C
from silly_kicks.win_probability import (
    WinProbabilityModel,
    WinProbabilityParams,
    compute_win_probability,
    goal_leverage,
)
from silly_kicks.win_probability._state import derive_match_states

PASS = C.actiontype_id["pass"]
SUCC = C.result_id["success"]
_P = WinProbabilityParams.default()


def _fitted(actions, games):
    return WinProbabilityModel(params=_P).fit(actions, games=games)


def test_conservation_and_counts(sample_two_match_actions, sample_games):
    m = _fitted(sample_two_match_actions, sample_games)
    _, report = compute_win_probability(sample_two_match_actions, model=m, games=sample_games)
    assert report.n_matches_in == 2
    assert report.n_matches_scored == 2
    assert report.n_matches_excluded_not_two_teams == 0
    assert report.n_matches_scored + report.n_matches_excluded_not_two_teams == report.n_matches_in
    assert report.n_actions == len(sample_two_match_actions)


def test_all_actions_have_a_row(sample_two_match_actions, sample_games):
    m = _fitted(sample_two_match_actions, sample_games)
    samples, _ = compute_win_probability(sample_two_match_actions, model=m, games=sample_games)
    assert len(samples) == len(sample_two_match_actions)


def test_non_two_team_excluded_and_counted(sample_two_match_actions, sample_games):
    # add a one-team game g3 (only team A) -> excluded_not_two_teams, NaN metrics, counted.
    g3 = pd.DataFrame(
        {
            "game_id": ["g3", "g3"],
            "action_id": [0, 1],
            "period_id": [1, 1],
            "team_id": ["A", "A"],
            "time_seconds": [10, 20],
            "type_id": [PASS, PASS],
            "result_id": [SUCC, SUCC],
        }
    )
    actions = pd.concat([sample_two_match_actions, g3], ignore_index=True)
    m = _fitted(sample_two_match_actions, sample_games)
    samples, report = compute_win_probability(actions, model=m, games=sample_games)
    assert report.n_matches_in == 3
    assert report.n_matches_excluded_not_two_teams == 1
    g3_rows = samples[samples["game_id"] == "g3"]
    assert (g3_rows["win_prob_source"] == "excluded_not_two_teams").all()
    assert g3_rows["p_win"].isna().all()


def test_scored_probs_sum_to_one(sample_two_match_actions, sample_games):
    m = _fitted(sample_two_match_actions, sample_games)
    samples, _ = compute_win_probability(sample_two_match_actions, model=m, games=sample_games)
    scored = samples[samples["win_prob_source"] == "scored"].dropna(subset=["p_win"])
    s = scored["p_win"] + scored["p_draw"] + scored["p_loss"]
    assert np.allclose(s.to_numpy(), 1.0, atol=1e-9)


def test_table_lookup_matches_predict_outcome(sample_two_match_actions, sample_games):
    # SPEC-12: the per-match table lookup must equal a direct predict_outcome roll for an action's state.
    m = _fitted(sample_two_match_actions, sample_games)
    states = derive_match_states(sample_two_match_actions, games=sample_games, params=_P)
    samples, _ = compute_win_probability(sample_two_match_actions, model=m, games=sample_games)
    # pick g1 action_id 0 (0-0, early, team A home)
    st = states[(states["game_id"] == "g1") & (states["action_id"] == 0)].iloc[0]
    w, draw, loss = m.predict_outcome(
        score_diff=int(st["score_diff"]),
        minutes_remaining=float(st["minutes_remaining"]),
        base_strength=0.0,
        home=bool(st["home"]),
        man_advantage=int(st["man_advantage"]),
    )
    row = samples[(samples["game_id"] == "g1") & (samples["action_id"] == 0)].iloc[0]
    assert abs(row["p_win"] - w) < 1e-9 and abs(row["p_draw"] - draw) < 1e-9 and abs(row["p_loss"] - loss) < 1e-9


def test_goal_leverage_aligns(sample_two_match_actions, sample_games):
    m = _fitted(sample_two_match_actions, sample_games)
    samples, _ = compute_win_probability(sample_two_match_actions, model=m, games=sample_games)
    lev = goal_leverage(sample_two_match_actions, model=m, games=sample_games)
    assert len(lev) == len(sample_two_match_actions)
    np.testing.assert_allclose(lev.to_numpy(), samples["win_prob_leverage"].to_numpy(), equal_nan=True, atol=1e-12)


def test_purity_no_mutation(sample_two_match_actions, sample_games):
    m = _fitted(sample_two_match_actions, sample_games)
    snap = sample_two_match_actions.copy()
    compute_win_probability(sample_two_match_actions, model=m, games=sample_games)
    pd.testing.assert_frame_equal(sample_two_match_actions, snap)


def test_integer_game_id_works(sample_two_match_actions):
    # id-dtype: integer game_ids route through canonical_id_series without raising.
    a = sample_two_match_actions.copy()
    a["game_id"] = a["game_id"].map({"g1": 1, "g2": 2})
    games = pd.DataFrame({"game_id": [1, 2], "home_team_id": ["A", "A"]})
    m = _fitted(a, games)
    samples, report = compute_win_probability(a, model=m, games=games)
    assert report.n_matches_scored == 2
    assert len(samples) == len(a)
