import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as C
from silly_kicks.win_probability import WinProbabilityParams
from silly_kicks.win_probability._state import derive_match_states

SHOT = C.actiontype_id["shot"]
PASS = C.actiontype_id["pass"]
FOUL = C.actiontype_id["foul"]
SUCC = C.result_id["success"]
FAIL = C.result_id["fail"]
OG = C.result_id["owngoal"]
RED = C.result_id["red_card"]
YEL = C.result_id["yellow_card"]
BADTOUCH = C.actiontype_id["bad_touch"]

_GAMES = pd.DataFrame({"game_id": ["g1"], "home_team_id": ["A"]})
_P = WinProbabilityParams.default()


def _actions():
    # game g1, teams A(home) & B; period 1; times ascending.
    rows = [
        (10, "A", PASS, SUCC),
        (20, "A", SHOT, SUCC),  # A scores -> own_score becomes 1 for the NEXT A state
        (30, "B", SHOT, FAIL),
        (40, "B", PASS, SUCC),
        (50, "A", PASS, SUCC),  # pre-action state here: A 1-0
    ]
    return pd.DataFrame(
        {
            "game_id": ["g1"] * 5,
            "action_id": range(5),
            "period_id": [1] * 5,
            "team_id": [r[1] for r in rows],
            "time_seconds": [r[0] for r in rows],
            "type_id": [r[2] for r in rows],
            "result_id": [r[3] for r in rows],
        }
    )


def test_score_is_pre_action():
    st = derive_match_states(_actions(), games=_GAMES, params=_P)
    last_a = st[(st["team_id"] == "A") & (st["action_id"] == 4)].iloc[0]
    assert last_a["own_score"] == 1 and last_a["opp_score"] == 0  # A's own goal counted, B none
    scoring_row = st[st["action_id"] == 1].iloc[0]  # the shot itself: pre-action state is 0-0
    assert scoring_row["own_score"] == 0 and scoring_row["opp_score"] == 0


def test_owngoal_credits_opponent():
    a = _actions().copy()
    a.loc[len(a)] = ["g1", 5, 1, "A", 60, BADTOUCH, OG]  # A own goal -> B +1
    a.loc[len(a)] = ["g1", 6, 1, "B", 70, PASS, SUCC]  # trailing B action to read the post-OG state
    st = derive_match_states(a, games=_GAMES, params=_P)
    b_after = st[st["action_id"] == 6].iloc[0]
    assert b_after["own_score"] == 1 and b_after["opp_score"] == 1  # B: own 1 (the OG), opp 1 (A's shot)


def test_absolute_minute_rebuilds_across_periods():
    a = _actions().copy()
    a.loc[len(a)] = ["g1", 5, 2, "A", 100, PASS, SUCC]  # period 2, t=100s
    st = derive_match_states(a, games=_GAMES, params=_P)
    p2 = st[st["action_id"] == 5].iloc[0]
    assert p2["absolute_minute"] > 45 and p2["minutes_remaining"] < 45


def test_man_advantage_direct_red():
    a = _actions().copy()
    a.loc[len(a)] = ["g1", 5, 1, "B", 55, FOUL, RED]  # B down to 10
    a.loc[len(a)] = ["g1", 6, 1, "A", 60, PASS, SUCC]
    st = derive_match_states(a, games=_GAMES, params=_P)
    assert st[st["action_id"] == 6].iloc[0]["man_advantage"] == 1  # A perspective: +1


def test_home_flag():
    st = derive_match_states(_actions(), games=_GAMES, params=_P)
    assert bool(st[st["team_id"] == "A"]["home"].iloc[0]) is True
    assert bool(st[st["team_id"] == "B"]["home"].iloc[0]) is False


def test_missing_home_raises():
    with pytest.raises(ValueError):
        derive_match_states(_actions(), games=None, params=_P)


def test_pure_no_mutation():
    a = _actions()
    snap = a.copy()
    derive_match_states(a, games=_GAMES, params=_P)
    pd.testing.assert_frame_equal(a, snap)


def test_man_advantage_second_yellow_becomes_red():
    # TF63-PLAN-03: two yellows to the SAME player -> red -> man_advantage flips. (spec §7)
    a = _actions().copy()
    a.loc[len(a)] = ["g1", 5, 1, "B", 52, FOUL, YEL]
    a.loc[len(a)] = ["g1", 6, 1, "B", 54, FOUL, YEL]
    a.loc[len(a)] = ["g1", 7, 1, "A", 60, PASS, SUCC]
    a["player_id"] = [None, None, None, None, None, "p9", "p9", None]  # same B player both yellows
    st = derive_match_states(a, games=_GAMES, params=_P)
    assert st[st["action_id"] == 7].iloc[0]["man_advantage"] == 1  # B a man down after 2nd yellow


def test_id_based_scoring_matches_vaep_labels_names():
    # TF63-PLAN-04 / SPEC-08: the id-based goal rule (what _state uses) must equal vaep.labels' name-based
    # predicate on a fixture.
    from silly_kicks import spadl
    from silly_kicks.vaep import labels as vlabels

    a = _actions()
    a["bodypart_id"] = C.bodypart_id["foot"]  # add_names merges on type_id/result_id/bodypart_id
    named = spadl.add_names(a)
    id_goal = a["type_id"].isin([C.actiontype_id[n] for n in ("shot", "shot_penalty", "shot_freekick")]) & (
        a["result_id"] == SUCC
    )
    name_goal = vlabels._is_goal(named)
    assert (id_goal.to_numpy() == np.asarray(name_goal)).all()


def test_state_golden_snapshot():
    # TF63-PLAN-05: pins the output so the perf vectorization (§ NOTE) stays byte-identical.
    st = derive_match_states(_actions(), games=_GAMES, params=_P).sort_values("action_id").reset_index(drop=True)
    exp = pd.DataFrame(
        {
            "action_id": [0, 1, 2, 3, 4],
            "score_diff": [0, 0, -1, -1, 1],  # B (actions 2,3) sees -1; A (action 4) sees +1
            "own_score": [0, 0, 0, 0, 1],
            "opp_score": [0, 0, 1, 1, 0],
        }
    )
    got = st[["action_id", "score_diff", "own_score", "opp_score"]]
    pd.testing.assert_frame_equal(got, exp, check_dtype=False)
