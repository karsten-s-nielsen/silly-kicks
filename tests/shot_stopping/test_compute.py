from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.shot_stopping import SHOT_STOPPING_COLUMNS, ShotStoppingReport, compute_shot_stopping
from silly_kicks.spadl import config as spadlconfig

_SHOT = spadlconfig.actiontype_id["shot"]
_PEN = spadlconfig.actiontype_id["shot_penalty"]
_BAD_TOUCH = spadlconfig.actiontype_id["bad_touch"]
_PASS = spadlconfig.actiontype_id["pass"]
_SUCCESS = spadlconfig.result_id["success"]
_FAIL = spadlconfig.result_id["fail"]
_OWNGOAL = spadlconfig.result_id["owngoal"]


def _row(gid, pid, tid, tyid, resid, psxg, dgk, dgk_team, blocked: object = pd.NA):
    return {
        "game_id": gid,
        "period_id": pid,
        "team_id": tid,
        "type_id": tyid,
        "result_id": resid,
        "psxg": psxg,
        "defending_gk_player_id": dgk,
        "defending_gk_team_id": dgk_team,
        "shot_blocked": blocked,
    }


def _actions(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["shot_blocked"] = df["shot_blocked"].astype("boolean")
    df["defending_gk_player_id"] = df["defending_gk_player_id"].astype("object")
    df["defending_gk_team_id"] = df["defending_gk_team_id"].astype("object")
    return df


def test_gsaa_exact_over_known_psxg():
    # Team 10 shoots at keeper 99 (team 20). 3 on-target: psxg .2/.5/.8 (one goal, .5). GP = 1.5 - 1 = 0.5.
    rows = [
        _row(1, 1, 10, _SHOT, _FAIL, 0.2, 99, 20),
        _row(1, 1, 10, _SHOT, _SUCCESS, 0.5, 99, 20),
        _row(1, 1, 10, _SHOT, _FAIL, 0.8, 99, 20),
    ]
    out, rep = compute_shot_stopping(_actions(rows), psxg_column="psxg")
    assert list(out.columns) == list(SHOT_STOPPING_COLUMNS)
    r = out[out["player_id"] == 99].iloc[0]
    assert r["shots_faced"] == 3
    assert r["goals_conceded"] == 1
    assert r["psxg_faced"] == pytest.approx(1.5)
    assert r["goals_prevented"] == pytest.approx(0.5)  # GSAA
    assert r["team_id"] == 20  # AUTHORITATIVE, from defending_gk_team_id (not inferred)
    assert isinstance(rep, ShotStoppingReport)
    assert (rep.n_shots_faced, rep.n_shots_attributed, rep.n_shots_unattributed) == (3, 3, 0)


def test_team_id_comes_from_resolver_not_opponent_inference():
    # A shot row carrying a NOISY / wrong team_id (shooter mislabeled) must NOT change the keeper's team:
    # team_id comes from defending_gk_team_id, not "the other team in the actions".
    rows = [_row(1, 1, 777, _SHOT, _FAIL, 0.3, 99, 20)]  # shooter team_id is garbage; keeper team = 20
    out, _ = compute_shot_stopping(_actions(rows), psxg_column="psxg")
    assert out[out["player_id"] == 99].iloc[0]["team_id"] == 20


def test_mixed_dtype_keeper_id_does_not_fragment_gsaa():
    # ADR-019: a keeper stamped with mixed id dtypes across a match (int 88 and str "88") must NOT
    # fragment into two output rows with split GSAA -- the CANONICAL group key merges them, and the
    # RAW id is emitted via .first(). Under the old raw grouping this produced TWO rows (len == 2).
    rows = [
        _row(1, 1, 10, _SHOT, _FAIL, 0.4, 88, 20),  # keeper id int 88
        _row(1, 1, 10, _SHOT, _SUCCESS, 0.6, "88", 20),  # SAME keeper, id str "88"
    ]
    out, rep = compute_shot_stopping(_actions(rows), psxg_column="psxg")
    assert len(out) == 1  # ONE keeper row, not two
    r = out.iloc[0]
    assert r["shots_faced"] == 2
    assert r["psxg_faced"] == pytest.approx(1.0)
    assert r["goals_prevented"] == pytest.approx(0.0)  # 1.0 psxg - 1 goal
    assert r["team_id"] == 20  # raw team via .first()
    assert rep.n_shots_attributed == 2


def test_on_target_gate_is_psxg_presence():
    rows = [
        _row(1, 1, 10, _SHOT, _FAIL, np.nan, 99, 20),  # off target (psxg NaN) -> excluded
        _row(1, 1, 10, _SHOT, _FAIL, 0.3, 99, 20),  # on target
    ]
    out, rep = compute_shot_stopping(_actions(rows), psxg_column="psxg")
    assert out[out["player_id"] == 99].iloc[0]["shots_faced"] == 1
    assert rep.n_shots_faced == 1


def test_blocked_and_owngoal_and_shootout_excluded():
    rows = [
        _row(1, 1, 10, _SHOT, _FAIL, 0.4, 99, 20, blocked=True),  # blocked -> excluded
        _row(1, 1, 10, _BAD_TOUCH, _OWNGOAL, 0.9, 99, 20),  # own goal (bad_touch) -> excluded by is_shot
        _row(1, 5, 10, _PEN, _SUCCESS, 0.75, 99, 20),  # shootout (period 5) -> excluded
        _row(1, 1, 10, _SHOT, _FAIL, 0.3, 99, 20),  # the only counted shot
    ]
    out, rep = compute_shot_stopping(_actions(rows), psxg_column="psxg")
    assert out[out["player_id"] == 99].iloc[0]["shots_faced"] == 1
    assert rep.n_shots_faced == 1


def test_penalty_split():
    rows = [
        _row(1, 1, 10, _PEN, _SUCCESS, 0.79, 99, 20),  # in-play penalty, scored
        _row(1, 1, 10, _SHOT, _FAIL, 0.3, 99, 20),  # open-play save
    ]
    out, _ = compute_shot_stopping(_actions(rows), psxg_column="psxg")
    r = out[out["player_id"] == 99].iloc[0]
    assert r["shots_faced"] == 2 and r["goals_conceded"] == 1
    assert r["psxg_faced"] == pytest.approx(1.09)
    assert r["goals_prevented"] == pytest.approx(0.09)
    assert r["shots_faced_excl_penalties"] == 1
    assert r["goals_conceded_excl_penalties"] == 0
    assert r["psxg_faced_excl_penalties"] == pytest.approx(0.3)
    assert r["goals_prevented_excl_penalties"] == pytest.approx(0.3)


def test_unattributed_shot_counted_not_dropped():
    rows = [
        _row(1, 1, 10, _SHOT, _FAIL, 0.6, pd.NA, pd.NA),  # unattributed (no keeper / no team)
        _row(1, 1, 10, _SHOT, _FAIL, 0.3, 99, 20),  # attributed
    ]
    out, rep = compute_shot_stopping(_actions(rows), psxg_column="psxg")
    assert set(out["player_id"]) == {99}
    assert (rep.n_shots_faced, rep.n_shots_attributed, rep.n_shots_unattributed) == (2, 1, 1)


def test_mid_match_gk_change_attributes_per_keeper():
    rows = [
        _row(1, 1, 10, _SHOT, _SUCCESS, 0.7, 99, 20),  # period 1 -> keeper 99 (team 20)
        _row(1, 2, 10, _SHOT, _FAIL, 0.4, 98, 20),  # period 2 -> keeper 98 (team 20, post-change)
    ]
    out, _ = compute_shot_stopping(_actions(rows), psxg_column="psxg")
    assert set(out["player_id"]) == {98, 99}
    assert out[out["player_id"] == 99].iloc[0]["goals_conceded"] == 1
    assert out[out["player_id"] == 98].iloc[0]["goals_conceded"] == 0
    assert set(out["team_id"]) == {20}  # both keepers on team 20


def test_keeper_faced_only_a_penalty_has_zero_excl_companions():
    # a keeper who faced ONLY an in-play penalty -> the _excl_penalties companions take the fillna path
    # (0 shots / 0 goals / 0.0 psxg / 0.0 GP), NOT NA.
    rows = [_row(1, 1, 10, _PEN, _SUCCESS, 0.79, 99, 20)]
    out, _ = compute_shot_stopping(_actions(rows), psxg_column="psxg")
    r = out[out["player_id"] == 99].iloc[0]
    assert r["shots_faced"] == 1 and r["goals_conceded"] == 1
    assert r["shots_faced_excl_penalties"] == 0 and r["goals_conceded_excl_penalties"] == 0
    assert r["psxg_faced_excl_penalties"] == pytest.approx(0.0)
    assert r["goals_prevented_excl_penalties"] == pytest.approx(0.0)


def test_appearances_to_compute_chain_flips_at_the_sub():
    # spec §6.4: the FULL PR1->PR2 chain. A half-time GK change (team 20: keeper 99 in period 1, keeper
    # 98 in period 2) via a PR1 appearance table -> add_defending_gk_player_id stamps the per-action
    # keeper + team -> compute attributes each period's shot to the right keeper (both on team 20).
    from silly_kicks.keeper_identity import (
        KeeperSegment,
        add_defending_gk_player_id,
        build_keeper_appearances_from_segments,
        resolve_keeper_identities,
    )

    acts = pd.DataFrame(
        [
            # team-10 shots (defended by team 20). type_name REQUIRED (roster resolver).
            {
                "game_id": 1,
                "action_id": 0,
                "period_id": 1,
                "time_seconds": 100.0,
                "team_id": 10,
                "player_id": 1,
                "type_id": _SHOT,
                "type_name": "shot",
                "result_id": _SUCCESS,
                "psxg": 0.7,
                "shot_blocked": pd.NA,
            },
            {
                "game_id": 1,
                "action_id": 1,
                "period_id": 2,
                "time_seconds": 100.0,
                "team_id": 10,
                "player_id": 1,
                "type_id": _SHOT,
                "type_name": "shot",
                "result_id": _FAIL,
                "psxg": 0.4,
                "shot_blocked": pd.NA,
            },
            # team 20 (the DEFENDING team) must ACT in EACH period so the roster resolver SEEDS it.
            {
                "game_id": 1,
                "action_id": 2,
                "period_id": 1,
                "time_seconds": 50.0,
                "team_id": 20,
                "player_id": 9,
                "type_id": _PASS,
                "type_name": "pass",
                "result_id": _SUCCESS,
                "psxg": np.nan,
                "shot_blocked": pd.NA,
            },
            {
                "game_id": 1,
                "action_id": 3,
                "period_id": 2,
                "time_seconds": 50.0,
                "team_id": 20,
                "player_id": 9,
                "type_id": _PASS,
                "type_name": "pass",
                "result_id": _SUCCESS,
                "psxg": np.nan,
                "shot_blocked": pd.NA,
            },
        ]
    )
    acts["shot_blocked"] = acts["shot_blocked"].astype("boolean")
    kmap, _ = resolve_keeper_identities(acts, identity="roster", roster={10: 1, 20: 99})
    ap = pd.concat(
        [
            build_keeper_appearances_from_segments(
                [KeeperSegment(20, 99, "starting_xi", 1, 0.0, 1, float("inf"))], [1, 2], game_id=1
            ),
            build_keeper_appearances_from_segments(
                [KeeperSegment(20, 98, "sub_events", 2, 0.0, 2, float("inf"))], [1, 2], game_id=1
            ),
            build_keeper_appearances_from_segments(
                [KeeperSegment(10, 1, "starting_xi", 1, 0.0, 2, float("inf"))], [1, 2], game_id=1
            ),
        ],
        ignore_index=True,
    )
    stamped = add_defending_gk_player_id(acts, kmap, appearances=ap)
    out, _ = compute_shot_stopping(stamped, psxg_column="psxg")
    assert set(out["player_id"]) == {98, 99}
    assert out[out["player_id"] == 99].iloc[0]["goals_conceded"] == 1  # period-1 keeper conceded
    assert out[out["player_id"] == 98].iloc[0]["goals_conceded"] == 0
    assert set(out["team_id"]) == {20}


def test_report_conserves():
    rows = [_row(1, 1, 10, _SHOT, _FAIL, 0.3, 99, 20), _row(1, 1, 10, _SHOT, _FAIL, 0.6, pd.NA, pd.NA)]
    _, rep = compute_shot_stopping(_actions(rows), psxg_column="psxg")
    assert rep.n_shots_attributed + rep.n_shots_unattributed == rep.n_shots_faced


def test_missing_psxg_column_raises_with_canonical_message():
    rows = [_row(1, 1, 10, _SHOT, _FAIL, 0.3, 99, 20)]
    with pytest.raises(KeyError, match=r"ships no.*xG/PSxG"):
        compute_shot_stopping(_actions(rows), psxg_column="post_shot_xg")
