"""Sportec DFL ``Penalty`` mapping, direct-free-kick pairing and ``BallDeflection`` routing (spec §25).

C1 of the TF-46 Arm 1 Part 2 cycle. Events-only; tested on the committed IDSSE fixtures plus synthetic
DFL-bronze frames. See ``docs/superpowers/adrs/ADR-DRAFT-sportec-penalty-mapping.md``.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl import sportec
from silly_kicks.spadl._training_exclusion import training_exclusion_mask

_IDSSE = Path(__file__).parent.parent / "datasets" / "idsse"
IDSSE_MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]

SHOT_PENALTY = spadlconfig.actiontype_id["shot_penalty"]
SHOT_FREEKICK = spadlconfig.actiontype_id["shot_freekick"]
FREEKICK_SHORT = spadlconfig.actiontype_id["freekick_short"]
SHOT = spadlconfig.actiontype_id["shot"]
SUCCESS = spadlconfig.result_id["success"]
FAIL = spadlconfig.result_id["fail"]

# ---------------------------------------------------------------------------
# Fixture / synthetic-bronze helpers
# ---------------------------------------------------------------------------

_REQUIRED = ["match_id", "event_id", "event_type", "period", "timestamp_seconds", "player_id", "team", "x", "y"]
_DEFAULTS = {"match_id": "M", "player_id": "", "team": "home", "x": 50.0, "y": 34.0}


def _bronze(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal, valid DFL-bronze events frame from a list of partial row dicts.

    Fills the required columns and sensible defaults; assigns a unique ``event_id`` per row. Omits
    ``x_source_position`` / ``x_position_from_tracking`` so the input-convention validator skips.
    """
    df = pd.DataFrame(rows)
    if "event_id" not in df.columns:
        df["event_id"] = [f"e{i}" for i in range(len(df))]
    for col in _REQUIRED:
        if col not in df.columns:
            df[col] = _DEFAULTS.get(col)
        if col in _DEFAULTS:
            df[col] = df[col].where(df[col].notna(), _DEFAULTS[col])
    df["period"] = df["period"].astype(int)
    df["timestamp_seconds"] = df["timestamp_seconds"].astype(float)
    return df


def _finalwhistle(period: int, t: float, score: str) -> dict:
    return {"event_type": "FinalWhistle", "period": period, "timestamp_seconds": t, "whistle_final_result": score}


def _penalty(period: int, t: float, team: str, taker: str, pen_team: str, retaken: str = "false") -> dict:
    return {
        "event_type": "Penalty",
        "period": period,
        "timestamp_seconds": t,
        "team": team,
        "player_id": "",
        "x": 94.0,
        "y": 34.0,
        "shot_player": taker,
        "penalty_prospective_taker": taker,
        "penalty_team": pen_team,
        "shot_team": pen_team,
        "shot_type_of_shot": "rightLeg",
        "penalty_retaken_penalty": retaken,
    }


def _kickoff(period: int, t: float, team: str) -> dict:
    return {"event_type": "KickOff", "period": period, "timestamp_seconds": t, "team": team}


_CLU = {"home": "CLU_H", "away": "CLU_A"}


def _clu_play(period: int, t: float, team: str, player: str = "P9") -> dict:
    # A benign on-ball event that carries the role<->CLU bridge (team_id_native) the converter needs.
    return {
        "event_type": "Play",
        "period": period,
        "timestamp_seconds": t,
        "team": team,
        "team_id_native": _CLU[team],
        "player_id": player,
    }


def _owngoal(period: int, t: float, scorer_team: str, player: str, counts_for_clu: str, *, scorer_clu=None) -> dict:
    # A top-level DFL OwnGoal event. `scorer_team`/`scorer_clu` is the conceder; the goal counts for the
    # DFL-CLU id `counts_for_clu` (the field; resolved to a role via team_id_native).
    return {
        "event_type": "OwnGoal",
        "period": period,
        "timestamp_seconds": t,
        "team": scorer_team,
        "team_id_native": scorer_clu if scorer_clu is not None else _CLU.get(scorer_team),
        "player_id": player,
        "x": 2.0,
        "y": 34.0,
        "own_goal_team_own_goal_counts_for": counts_for_clu,
        "own_goal_type_of_shot": "rightLeg",
    }


def _convert_idsse_no_score(match: str):
    meta = json.loads((_IDSSE / f"{match}_meta.json").read_text(encoding="utf-8"))
    events = pd.read_parquet(_IDSSE / f"{match}_events_native.parquet").copy()
    events["whistle_final_result"] = pd.NA  # force steps 3 / 3m
    return sportec.convert_to_actions(
        events,
        home_team_id="home",
        home_team_start_left=meta["home_team_start_left"],
        home_team_start_left_extratime=meta.get("home_team_start_left_extratime"),
    )


def _shot(
    period: int,
    t: float,
    team: str,
    player: str,
    outcome: str,
    *,
    penalty_team: str | None = None,
    after_fk: bool = False,
) -> dict:
    row = {
        "event_type": "ShotAtGoal",
        "period": period,
        "timestamp_seconds": t,
        "team": team,
        "player_id": player,
        "x": 90.0,
        "y": 34.0,
        "shot_player": player,
        "shot_outcome_type": outcome,
    }
    if penalty_team is not None:
        row["penalty_team"] = penalty_team
    if after_fk:
        row["shot_after_free_kick"] = "true"
    return row


def _freekick(period: int, t: float, team: str, player: str) -> dict:
    return {
        "event_type": "FreeKick",
        "period": period,
        "timestamp_seconds": t,
        "team": team,
        "player_id": player,
        "x": 25.0,
        "y": 40.0,
    }


def _balldeflection(period: int, t: float, team: str, player: str, dtype: str, x: float, y: float) -> dict:
    return {
        "event_type": "BallDeflection",
        "period": period,
        "timestamp_seconds": t,
        "team": team,
        "player_id": player,
        "x": x,
        "y": y,
        "deflection_type": dtype,
        "deflection_player": player,
    }


def _convert(rows: list[dict]):
    return sportec.convert_to_actions(_bronze(rows), home_team_id="home", home_team_start_left=True)


def _convert_idsse(match: str):
    meta = json.loads((_IDSSE / f"{match}_meta.json").read_text(encoding="utf-8"))
    events = pd.read_parquet(_IDSSE / f"{match}_events_native.parquet")
    return sportec.convert_to_actions(
        events,
        home_team_id="home",
        home_team_start_left=meta["home_team_start_left"],
        home_team_start_left_extratime=meta.get("home_team_start_left_extratime"),
    )


def _shot_player_of_penalty_event(match: str) -> str:
    events = pd.read_parquet(_IDSSE / f"{match}_events_native.parquet")
    pen = events[events["event_type"] == "Penalty"]
    return str(pen.iloc[-1]["shot_player"])


# ---------------------------------------------------------------------------
# IDSSE real-fixture penalties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("match", ["J03WMX", "J03WOH", "J03WR9"])
def test_idsse_penalty_becomes_shot_penalty(match):
    actions, report = _convert_idsse(match)
    pen = actions[actions["type_id"] == SHOT_PENALTY]
    assert len(pen) == 1
    row = pen.iloc[-1]
    assert row["result_id"] == SUCCESS
    assert row["result_source"] == "native"
    assert row["penalty_result_method"] == "score_reconciliation"
    assert row["player_id"] == _shot_player_of_penalty_event(match)
    # the start is the rule-snapped penalty spot in one of the two attack orientations (x ~ 94 or ~ 11),
    # centred in y.
    assert abs(float(row["start_x"]) - 94.0) < 2.0 or abs(float(row["start_x"]) - 11.0) < 2.0
    assert abs(float(row["start_y"]) - 34.0) < 2.0
    assert report.penalty_result_method_counts.get("score_reconciliation", 0) >= 1


# ---------------------------------------------------------------------------
# Synthetic result-table cases
# ---------------------------------------------------------------------------


def test_all_missed_team_reconciles_to_fail():
    # final 0:0, one home penalty -> reconciled home penalty goals = 0 -> fail at step 2.
    actions, _ = _convert(
        [
            _kickoff(1, 0.0, "away"),
            _penalty(2, 100.0, "home", "P1", "CLU_H"),
            _finalwhistle(2, 3000.0, "0:0"),
        ]
    )
    row = actions[actions["type_id"] == SHOT_PENALTY].iloc[0]
    assert (row["result_id"], row["result_source"], row["penalty_result_method"]) == (
        FAIL,
        "native",
        "score_reconciliation",
    )


def test_elimination_2_pens_1_goal():
    # final 1:0, two home penalties; #1 proven scored by an away kick-off; #2 eliminated to fail.
    actions, _ = _convert(
        [
            _penalty(2, 100.0, "home", "P1", "CLU_H"),
            _kickoff(2, 110.0, "away"),
            _penalty(2, 200.0, "home", "P2", "CLU_H"),
            _finalwhistle(2, 3000.0, "1:0"),
        ]
    )
    pens = actions[actions["type_id"] == SHOT_PENALTY].sort_values("time_seconds")
    first, second = pens.iloc[0], pens.iloc[1]
    assert (first["result_id"], first["result_source"], first["penalty_result_method"]) == (
        SUCCESS,
        "native",
        "kickoff_order",
    )
    assert (second["result_id"], second["result_source"], second["penalty_result_method"]) == (
        FAIL,
        "native",
        "score_elimination",
    )


def test_elimination_3_pens_2_goals_one_proven():
    # final 2:0, three home penalties; #1 proven; R=1,U=2 -> #2,#3 unresolved (step 4).
    actions, _ = _convert(
        [
            _penalty(2, 100.0, "home", "P1", "CLU_H"),
            _kickoff(2, 110.0, "away"),
            _penalty(2, 200.0, "home", "P2", "CLU_H"),
            _penalty(2, 300.0, "home", "P3", "CLU_H"),
            _finalwhistle(2, 3000.0, "2:0"),
        ]
    )
    pens = actions[actions["type_id"] == SHOT_PENALTY].sort_values("time_seconds")
    assert pens.iloc[0]["penalty_result_method"] == "kickoff_order"
    for i in (1, 2):
        assert pens.iloc[i]["result_source"] == "inferred"
        assert pens.iloc[i]["penalty_result_method"] == "unresolved"


def test_elimination_3_pens_2_goals_two_proven():
    # final 2:0, three home penalties; #1 and #2 proven by kick-offs; #3 eliminated to fail.
    actions, _ = _convert(
        [
            _penalty(2, 100.0, "home", "P1", "CLU_H"),
            _kickoff(2, 110.0, "away"),
            _penalty(2, 200.0, "home", "P2", "CLU_H"),
            _kickoff(2, 210.0, "away"),
            _penalty(2, 300.0, "home", "P3", "CLU_H"),
            _finalwhistle(2, 3000.0, "2:0"),
        ]
    )
    pens = actions[actions["type_id"] == SHOT_PENALTY].sort_values("time_seconds")
    assert pens.iloc[0]["penalty_result_method"] == "kickoff_order"
    assert pens.iloc[1]["penalty_result_method"] == "kickoff_order"
    assert (pens.iloc[2]["result_id"], pens.iloc[2]["result_source"], pens.iloc[2]["penalty_result_method"]) == (
        FAIL,
        "native",
        "score_elimination",
    )


def test_equal_goals_and_penalties_resolve_at_step_2():
    # 2 penalties, 2 reconciled goals -> step 2 resolves BOTH success; step 3/3b never run (B-10).
    actions, _ = _convert(
        [
            _penalty(2, 100.0, "home", "P1", "CLU_H"),
            _kickoff(2, 110.0, "away"),
            _penalty(2, 200.0, "home", "P2", "CLU_H"),
            _finalwhistle(2, 3000.0, "2:0"),
        ]
    )
    pens = actions[actions["type_id"] == SHOT_PENALTY]
    assert set(pens["penalty_result_method"]) == {"score_reconciliation"}
    assert set(pens["result_id"]) == {SUCCESS}


def test_mixed_format_legacy_row_outside_counting_set():
    # Sub-case A: a legacy ShotAtGoal penalty MISS + 2 new-format Penalty, reconciled goals = 2.
    actions_a, _ = _convert(
        [
            _shot(2, 50.0, "home", "L1", "saved", penalty_team="CLU_H"),  # legacy penalty, missed
            _penalty(2, 100.0, "home", "P1", "CLU_H"),
            _penalty(2, 200.0, "home", "P2", "CLU_H"),
            _finalwhistle(2, 3000.0, "2:0"),
        ]
    )
    pens_a = actions_a[actions_a["type_id"] == SHOT_PENALTY].sort_values("time_seconds")
    legacy = pens_a.iloc[0]
    assert (legacy["result_id"], legacy["result_source"], legacy["penalty_result_method"]) == (
        FAIL,
        "native",
        "shot_outcome",
    )
    new_rows = pens_a.iloc[1:]
    assert set(new_rows["result_id"]) == {SUCCESS}
    assert set(new_rows["penalty_result_method"]) == {"score_reconciliation"}

    # Sub-case B: a legacy SCORED penalty + 1 new-format Penalty with reconciled = 0 (B-14: the legacy
    # success is already inside ShotAtGoal goals and is never double-counted in R).
    actions_b, _ = _convert(
        [
            _shot(2, 50.0, "home", "L1", "successful", penalty_team="CLU_H"),  # legacy penalty goal
            _penalty(2, 100.0, "home", "P1", "CLU_H"),
            _finalwhistle(2, 3000.0, "1:0"),
        ]
    )
    pens_b = actions_b[actions_b["type_id"] == SHOT_PENALTY].sort_values("time_seconds")
    legacy_b, new_b = pens_b.iloc[0], pens_b.iloc[1]
    assert (legacy_b["result_id"], legacy_b["result_source"], legacy_b["penalty_result_method"]) == (
        SUCCESS,
        "native",
        "shot_outcome",
    )
    assert (new_b["result_id"], new_b["result_source"], new_b["penalty_result_method"]) == (
        FAIL,
        "native",
        "score_reconciliation",
    )


def test_eliminate_helper_r_equals_u():
    elim2 = sportec._eliminate(r=2, unresolved=[7, 9])
    assert elim2 == {7: (SUCCESS, "score_elimination"), 9: (SUCCESS, "score_elimination")}
    elim0 = sportec._eliminate(r=0, unresolved=[7])
    assert elim0 == {7: (FAIL, "score_elimination")}
    assert sportec._eliminate(r=1, unresolved=[7, 9]) == {}


def test_no_kickoff_before_finalwhistle_is_not_a_miss():
    # J03WR9 shape: a scored penalty followed directly by FinalWhistle; reconciliation gives a goal.
    actions, _ = _convert(
        [
            _penalty(2, 3000.0, "home", "P1", "CLU_H"),
            _finalwhistle(2, 3005.0, "1:0"),
        ]
    )
    row = actions[actions["type_id"] == SHOT_PENALTY].iloc[0]
    assert (row["result_id"], row["result_source"], row["penalty_result_method"]) == (
        SUCCESS,
        "native",
        "score_reconciliation",
    )


def test_period_start_kickoff_never_counts():
    # 2 home penalties in period 1; the only later kick-off is the period-2 start -> step 3 cannot fire
    # (different period, and a period-start kick-off never counts). Score split -> step 4.
    actions, _ = _convert(
        [
            _kickoff(1, 0.0, "home"),
            _penalty(1, 100.0, "home", "P1", "CLU_H"),
            _penalty(1, 200.0, "home", "P2", "CLU_H"),
            _kickoff(2, 0.0, "away"),  # second-half start
            _finalwhistle(2, 3000.0, "1:0"),
        ]
    )
    pens = actions[actions["type_id"] == SHOT_PENALTY]
    assert "kickoff_order" not in set(pens["penalty_result_method"])
    assert set(pens["penalty_result_method"]) == {"unresolved"}


def test_continued_play_proves_a_miss():
    # Split (final 1:0, 2 home penalties). pen1 is followed by an on-ball Play before any kick-off -> miss
    # (continued_play, native). pen2 is followed by an opponent kick-off -> scored (kickoff_order, native).
    actions, _ = _convert(
        [
            _penalty(2, 100.0, "home", "P1", "CLU_H"),
            {"event_type": "Play", "period": 2, "timestamp_seconds": 105.0, "team": "home", "player_id": "P9"},
            _penalty(2, 200.0, "home", "P2", "CLU_H"),
            _kickoff(2, 210.0, "away"),
            _finalwhistle(2, 3000.0, "1:0"),
        ]
    )
    pens = actions[actions["type_id"] == SHOT_PENALTY].sort_values("time_seconds")
    assert (pens.iloc[0]["result_id"], pens.iloc[0]["result_source"], pens.iloc[0]["penalty_result_method"]) == (
        FAIL,
        "native",
        "continued_play",
    )
    assert pens.iloc[1]["penalty_result_method"] == "kickoff_order"


def test_var_disallowed_goal_restart_is_a_miss():
    # A VAR-disallowed goal restarts with a free kick, not a kick-off -> the FreeKick is on-ball before any
    # kick-off -> continued_play fail (the GoalDisallowed / VAR annotations do not count as on-ball).
    actions, _ = _convert(
        [
            _penalty(2, 100.0, "home", "P1", "CLU_H"),
            {"event_type": "GoalDisallowed", "period": 2, "timestamp_seconds": 102.0, "team": "home"},
            _freekick(2, 105.0, "away", "D1"),
            _penalty(2, 200.0, "home", "P2", "CLU_H"),
            _kickoff(2, 210.0, "away"),
            _finalwhistle(2, 3000.0, "1:0"),
        ]
    )
    pens = actions[actions["type_id"] == SHOT_PENALTY].sort_values("time_seconds")
    assert pens.iloc[0]["penalty_result_method"] == "continued_play"
    assert pens.iloc[0]["result_id"] == FAIL


def test_step4_r_over_u_all_success_inconsistent_and_warns():
    # reconciled penalty goals = 3 with 1 penalty -> R > U -> R' = 1 -> success, inferred, inconsistent, warn.
    with pytest.warns(UserWarning, match="penalty score inconsistent"):
        actions, _ = _convert(
            [
                _penalty(2, 100.0, "home", "P1", "CLU_H"),
                _finalwhistle(2, 3000.0, "3:0"),
            ]
        )
    pen_mask = actions["type_id"] == SHOT_PENALTY
    row = actions[pen_mask].iloc[0]
    assert (row["result_id"], row["result_source"], row["penalty_result_method"]) == (SUCCESS, "inferred", "unresolved")
    assert row["repair_kind"] == "penalty_score_inconsistent"
    excl = training_exclusion_mask(actions)
    assert bool(excl[pen_mask].iloc[0]) is True
    assert not excl[~pen_mask].any()


def test_step4_r_below_zero_all_fail_inconsistent():
    # 2 home ShotAtGoal goals + 1 penalty, final 1:0 -> reconciled = -1 -> R < 0 -> R' = 0 -> all fail,
    # inconsistent, warn.
    with pytest.warns(UserWarning, match="penalty score inconsistent"):
        actions, _ = _convert(
            [
                _shot(2, 40.0, "home", "S1", "successful"),
                _shot(2, 50.0, "home", "S2", "successful"),
                _penalty(2, 100.0, "home", "P1", "CLU_H"),
                _finalwhistle(2, 3000.0, "1:0"),
            ]
        )
    row = actions[actions["type_id"] == SHOT_PENALTY].iloc[0]
    assert (row["result_id"], row["result_source"]) == (FAIL, "inferred")
    assert row["repair_kind"] == "penalty_score_inconsistent"


def test_step4_split_earliest_r_prime_success():
    # 2 home penalties, both to a period end (no kick-off, no on-ball), final 1:0 -> R' = 1 -> the earliest
    # penalty success, the later fail; both inferred, consistent (penalty_outcome_unresolved).
    actions, _ = _convert(
        [
            _penalty(2, 100.0, "home", "P1", "CLU_H"),
            _penalty(2, 200.0, "home", "P2", "CLU_H"),
            _finalwhistle(2, 3000.0, "1:0"),
        ]
    )
    pens = actions[actions["type_id"] == SHOT_PENALTY].sort_values("time_seconds")
    assert (pens.iloc[0]["result_id"], pens.iloc[0]["result_source"]) == (SUCCESS, "inferred")
    assert (pens.iloc[1]["result_id"], pens.iloc[1]["result_source"]) == (FAIL, "inferred")
    assert set(pens["repair_kind"]) == {"penalty_outcome_unresolved"}
    assert set(pens["penalty_result_method"]) == {"unresolved"}


def test_legacy_shotatgoal_penalty_path_still_shot_penalty():
    actions, _ = _convert(
        [
            _shot(2, 50.0, "home", "L1", "successful", penalty_team="CLU_H"),
            _finalwhistle(2, 3000.0, "1:0"),
        ]
    )
    row = actions[actions["type_id"] == SHOT_PENALTY].iloc[0]
    assert (row["result_id"], row["result_source"], row["penalty_result_method"]) == (
        SUCCESS,
        "native",
        "shot_outcome",
    )


def test_direct_freekick_pairing_gives_one_shot_freekick():
    # Within 5 s: the standalone FreeKick pairs with the same player's direct shot -> one shot_freekick row
    # carrying the FreeKick's start; no freekick_* row.
    actions, _ = _convert(
        [
            _freekick(1, 100.0, "home", "P1"),
            _shot(1, 101.2, "home", "P1", "saved", after_fk=True),
            _finalwhistle(2, 3000.0, "0:0"),
        ]
    )
    assert (actions["type_id"] == SHOT_FREEKICK).sum() == 1
    assert (actions["type_id"] == FREEKICK_SHORT).sum() == 0
    assert (actions["type_id"] == spadlconfig.actiontype_id["freekick_crossed"]).sum() == 0
    fk_only, _ = _convert([_freekick(1, 100.0, "home", "P1"), _finalwhistle(2, 3000.0, "0:0")])
    fk_start = (fk_only.iloc[0]["start_x"], fk_only.iloc[0]["start_y"])
    row = actions[actions["type_id"] == SHOT_FREEKICK].iloc[0]
    assert (row["start_x"], row["start_y"]) == fk_start

    # At +6 s: too far to pair -> both the freekick_short and the shot_freekick survive.
    far, _ = _convert(
        [
            _freekick(1, 100.0, "home", "P1"),
            _shot(1, 106.0, "home", "P1", "saved", after_fk=True),
            _finalwhistle(2, 3000.0, "0:0"),
        ]
    )
    assert (far["type_id"] == SHOT_FREEKICK).sum() == 1
    assert (far["type_id"] == FREEKICK_SHORT).sum() == 1


# ---------------------------------------------------------------------------
# BallDeflection routing (§25.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "match,woodwork_type",
    [("J03WMX", "post_right"), ("J03WN1", "post_right"), ("J03WR9", "crossbar")],
)
def test_idsse_woodwork_deflection_sets_type_and_end(match, woodwork_type):
    actions, report = _convert_idsse(match)
    woodwork = actions[actions["shot_woodwork_type"].notna()]
    assert len(woodwork) == 1
    row = woodwork.iloc[0]
    assert row["shot_woodwork_type"] == woodwork_type
    # a placeholder end (end == start) is replaced by the deflection point.
    assert (row["end_x"], row["end_y"]) != (row["start_x"], row["start_y"])
    # the deflection never marks the shot blocked.
    assert bool(row["shot_blocked"]) is False
    assert report.deflection_counts.get("woodwork_paired", 0) >= 1


def test_idsse_referee_deflection_counted_no_row():
    actions, report = _convert_idsse("J03WPY")
    assert report.deflection_counts.get("deflection_referee", 0) == 1
    assert actions["shot_woodwork_type"].notna().sum() == 0


def test_unpaired_woodwork_deflection_warns_and_counts():
    with pytest.warns(UserWarning, match="deflection"):
        actions, report = _convert(
            [
                {"event_type": "Play", "period": 1, "timestamp_seconds": 50.0, "team": "home", "player_id": "P9"},
                _balldeflection(1, 100.0, "home", "P1", "postLeft", 105.0, 32.0),
                _finalwhistle(2, 3000.0, "0:0"),
            ]
        )
    assert report.deflection_counts.get("deflection_unpaired", 0) == 1
    assert actions["shot_woodwork_type"].notna().sum() == 0


# ---------------------------------------------------------------------------
# Own goal (top-level DFL OwnGoal event; decision 3)
# ---------------------------------------------------------------------------

_BAD_TOUCH = spadlconfig.actiontype_id["bad_touch"]
_OWNGOAL = spadlconfig.result_id["owngoal"]


def _own_goal_row(actions):
    og = actions[(actions["type_id"] == _BAD_TOUCH) & (actions["result_id"] == _OWNGOAL)]
    assert len(og) == 1
    return og.iloc[0]


def test_own_goal_credit_field_authoritative_agreement():
    # away concedes an own goal; the field counts_for = home's CLU. Row team_id = the conceder (away);
    # no conflict; body part from own_goal_type_of_shot.
    actions, report = _convert(
        [
            _clu_play(1, 40.0, "home"),
            _clu_play(1, 50.0, "away"),
            _owngoal(2, 100.0, "away", "D5", _CLU["home"]),
            _finalwhistle(2, 3000.0, "1:0"),
        ]
    )
    row = _own_goal_row(actions)
    assert row["team_id"] == "away"
    assert row["player_id"] == "D5"
    assert row["bodypart_id"] == spadlconfig.bodypart_id["foot"]  # rightLeg -> foot
    assert report.own_goal_attribution_conflicts == 0


def test_own_goal_credit_field_conflict_warns_and_counts():
    # native team says "home" scored the own goal, but the field credits home (counts_for = home CLU) ->
    # the conceder is away. The field is authoritative; the disagreement is a counted conflict.
    with pytest.warns(UserWarning, match="disagrees with the native team"):
        actions, report = _convert(
            [
                _clu_play(1, 40.0, "home"),
                _clu_play(1, 50.0, "away"),
                _owngoal(2, 100.0, "home", "D5", _CLU["home"]),  # native "home" contradicts counts_for=home
                _finalwhistle(2, 3000.0, "1:0"),
            ]
        )
    assert _own_goal_row(actions)["team_id"] == "away"  # field-resolved conceder
    assert report.own_goal_attribution_conflicts == 1


def test_own_goal_credit_field_absent_falls_back_to_native():
    # No counts_for field -> keep the native conceder role; credit is the opponent by result.
    og = _owngoal(2, 100.0, "away", "D5", _CLU["home"])
    del og["own_goal_team_own_goal_counts_for"]
    actions, report = _convert([_clu_play(1, 40.0, "home"), og, _finalwhistle(2, 3000.0, "1:0")])
    assert _own_goal_row(actions)["team_id"] == "away"
    assert report.own_goal_attribution_conflicts == 0


def test_own_goal_scorer_unknown_resolved_by_field():
    # native team is "unknown"; the field (counts_for = home CLU) resolves the conceder to away. Goal not lost.
    actions, _ = _convert(
        [
            _clu_play(1, 40.0, "home"),
            _clu_play(1, 50.0, "away"),
            _owngoal(2, 100.0, "unknown", "D5", _CLU["home"], scorer_clu=None),
            _finalwhistle(2, 3000.0, "1:0"),
        ]
    )
    assert _own_goal_row(actions)["team_id"] == "away"


def test_missed_penalty_resolves_fail_via_opponent_own_goal_at_step_2():
    # home's only goal is an AWAY own goal; home's penalty therefore reconciles to fail at step 2. Without
    # own-goal counting, step 2 would wrongly label the penalty native `success`.
    actions, _ = _convert(
        [
            _clu_play(1, 40.0, "home"),
            _clu_play(1, 50.0, "away"),
            _owngoal(2, 90.0, "away", "D5", _CLU["home"]),
            _penalty(2, 100.0, "home", "P1", "CLU_H"),
            _finalwhistle(2, 3000.0, "1:0"),
        ]
    )
    row = actions[actions["type_id"] == SHOT_PENALTY].iloc[0]
    assert (row["result_id"], row["result_source"], row["penalty_result_method"]) == (
        FAIL,
        "native",
        "score_reconciliation",
    )


def test_step2_own_goal_with_unknown_row_present_credits_correctly():
    # M9 regression: with an "unknown"-team row present, the own-goal credit must still resolve (via the
    # field + kick-off/role pair), so home's only goal is the away own goal and the penalty reconciles to
    # fail. The old opponent-from-all-labels path returned None here and mislabelled the penalty success.
    actions, _ = _convert(
        [
            {"event_type": "FinalWhistle", "period": 1, "timestamp_seconds": 1.0, "team": "unknown"},
            _clu_play(1, 40.0, "home"),
            _clu_play(1, 50.0, "away"),
            _owngoal(2, 90.0, "away", "D5", _CLU["home"]),
            _penalty(2, 100.0, "home", "P1", "CLU_H"),
            _finalwhistle(2, 3000.0, "1:0"),
        ]
    )
    row = actions[actions["type_id"] == SHOT_PENALTY].iloc[0]
    assert (row["result_id"], row["result_source"], row["penalty_result_method"]) == (
        FAIL,
        "native",
        "score_reconciliation",
    )


def _penalty_methods(actions) -> list:
    pens = actions[actions["type_id"] == SHOT_PENALTY].sort_values("time_seconds")
    return list(zip(pens["result_id"], pens["result_source"], pens["penalty_result_method"], strict=True))


# --- C-17 (option B): refuse only when an own goal's CREDIT is unknown, not when opp is None -------------


def test_step2_credit_from_field_reconciles_without_opponent():
    # C-17 B (reshaped B-12): counts_for = CLU_H maps to home, so the credit is provider-known even though the
    # conceder is native `unknown` and the opponent is unresolvable (one-team frame). The own goal counts for
    # home, so home's penalty reconciles to fail at step 2 -- no raise (C-3: the field resolves the credit).
    actions, _ = _convert(
        [
            _clu_play(1, 40.0, "home"),  # CLU_H <-> home bridge
            _owngoal(2, 90.0, "unknown", "D5", _CLU["home"]),  # counts_for = home; conceder native unknown
            _penalty(2, 100.0, "home", "P1", "CLU_H"),
            _finalwhistle(2, 3000.0, "1:0"),
        ]
    )
    assert _penalty_methods(actions) == [(FAIL, "native", "score_reconciliation")]


def test_step2_fails_loud_when_owngoal_credit_unknown_two_team():
    # The resolved-opponent path (B-12): both teams present, opponent resolves, but the own goal kept a native
    # `unknown` conceder with no counts_for, so its credit is unknown. The step-2 term would drop it silently;
    # the invariant refuses instead (C-17 B closes this path too, not just the unresolvable-opponent one).
    with pytest.raises(ValueError, match="unknown credit"):
        _convert(
            [
                _clu_play(1, 40.0, "home"),
                _clu_play(1, 50.0, "away"),
                _owngoal(2, 90.0, "unknown", "D5", "", scorer_clu=None),  # unknown conceder, no counts_for
                _penalty(2, 100.0, "home", "P1", "CLU_H"),
                _finalwhistle(2, 3000.0, "1:0"),
            ]
        )


def test_step2_fails_loud_when_owngoal_credit_unknown_one_team():
    # The unresolvable-opponent path: a single-team frame, own goal with an unknown credit (no counts_for) ->
    # refuse. Same invariant, the opp-None trigger.
    with pytest.raises(ValueError, match="unknown credit"):
        _convert(
            [
                _clu_play(1, 40.0, "home"),
                _owngoal(2, 90.0, "unknown", "D5", "", scorer_clu=None),
                _penalty(2, 100.0, "home", "P1", "CLU_H"),
                _finalwhistle(2, 3000.0, "1:0"),
            ]
        )


def test_step2_shot_outcome_owngoal_is_credited_not_refused():
    # C1-IMPL-18 regression: a ShotAtGoal with shot_outcome == "ownGoal" (the defensive path kept for other
    # feeds) is an own goal BY RESULT, so `_penalty_results` counts it. Its conceder (away) is a resolved
    # match team, so rule 2 credits home -- the match must reconcile, not refuse. The build mask (owngoal by
    # result) must match the consume mask; an earlier version built credit for OwnGoal events only and raised.
    actions, _ = _convert(
        [
            _shot(2, 90.0, "away", "D5", "ownGoal"),  # away scores into its own net -> credited to home
            _penalty(2, 100.0, "home", "P1", "CLU_H"),
            _finalwhistle(2, 3000.0, "1:0"),
        ]
    )
    assert _penalty_methods(actions) == [(FAIL, "native", "score_reconciliation")]


def test_step2_no_owngoal_single_team_frame_does_not_fail_loud():
    # The other side of the gate: with NO own goal there is nothing to credit, so a single-team frame
    # (opponent unresolvable) reconciles without raising. This is the common synthetic shape; it must stay
    # working.
    actions, _ = _convert(
        [
            _clu_play(1, 40.0, "home"),
            _penalty(2, 100.0, "home", "P1", "CLU_H"),
            _finalwhistle(2, 3000.0, "1:0"),
        ]
    )
    assert (actions["type_id"] == SHOT_PENALTY).sum() == 1


# --- C-16: steps 3 / 3m emit no opponent-dependent native result when the opponent is unresolvable --------


def test_opp_unresolvable_scored_then_missed_penalty_reconciles():
    # C's frame: home scores penalty 1 (the next kick-off carries the unresolvable label `unknown`), home
    # misses penalty 2, final 1:0. Pen1's "kick-off first" is not evidence (opponent unattributable) -> step 3
    # emits nothing and step 3b elects it success (score_elimination). Pen2 continued play -> native miss.
    actions, report = _convert(
        [
            _penalty(2, 100.0, "home", "P1", "CLU_H"),
            _kickoff(2, 110.0, "unknown"),  # restart after the scored penalty; team unresolvable
            _penalty(2, 200.0, "home", "P2", "CLU_H"),
            _clu_play(2, 210.0, "home"),  # on-ball after pen2 -> observed continued play
            _finalwhistle(2, 3000.0, "1:0"),
        ]
    )
    assert _penalty_methods(actions) == [
        (SUCCESS, "native", "score_elimination"),
        (FAIL, "native", "continued_play"),
    ]
    assert report.penalty_opponent_unresolved == 1


def test_opp_unresolvable_missed_then_scored_penalty_reconciles():
    # The reversed order: pen1 missed (continued play), pen2 scored (kick-off first, unresolvable -> step 3
    # emits nothing, step 3b elects pen2 success). The earliest-R' rule must not mislabel; step 3b fixes it.
    actions, _ = _convert(
        [
            _penalty(2, 100.0, "home", "P1", "CLU_H"),
            _clu_play(2, 110.0, "home"),  # on-ball after pen1 -> continued play
            _penalty(2, 200.0, "home", "P2", "CLU_H"),
            _kickoff(2, 210.0, "unknown"),  # restart after the scored penalty; team unresolvable
            _finalwhistle(2, 3000.0, "1:0"),
        ]
    )
    assert _penalty_methods(actions) == [
        (FAIL, "native", "continued_play"),
        (SUCCESS, "native", "score_elimination"),
    ]


def test_opp_unresolvable_no_final_score_penalty_goes_to_step4():
    # C's frame with no final score: pen2 continued play (native miss, observed), pen1's kick-off-first is not
    # evidence and there is no score to reconcile -> pen1 falls to step 4 (inferred, repair_kind set).
    actions, _ = _convert(
        [
            _penalty(2, 100.0, "home", "P1", "CLU_H"),
            _kickoff(2, 110.0, "unknown"),
            _penalty(2, 200.0, "home", "P2", "CLU_H"),
            _clu_play(2, 210.0, "home"),
        ]
    )
    pens = actions[actions["type_id"] == SHOT_PENALTY].sort_values("time_seconds")
    assert list(pens["penalty_result_method"]) == ["unresolved", "continued_play"]
    assert list(pens["result_source"]) == ["inferred", "native"]
    # pen1 carries the step-4 repair tag; pen2 (continued_play native) is a clean row with repair_kind
    # missing. repair_kind is object dtype, so the missing value is None or NaN by pandas version -- assert
    # NA-robustly, not against a single sentinel identity.
    assert pens["repair_kind"].iloc[0] == "penalty_outcome_unresolved"
    assert pd.isna(pens["repair_kind"].iloc[1])


def test_period_without_kickoff_emits_no_native_step3_result():
    # Kick-off premise gate: the period codes no KickOff at all, so an absent opponent kick-off proves
    # nothing. Steps 3 / 3m emit neither kickoff_order nor continued_play; the penalties fall to step 4.
    actions, _ = _convert(
        [
            _penalty(2, 100.0, "home", "P1", "CLU_H"),
            _clu_play(2, 110.0, "home"),
            _penalty(2, 200.0, "home", "P2", "CLU_H"),
            _clu_play(2, 210.0, "home"),
            _finalwhistle(2, 3000.0, "1:0"),
        ]
    )
    methods = set(actions.loc[actions["type_id"] == SHOT_PENALTY, "penalty_result_method"])
    assert "kickoff_order" not in methods
    assert "continued_play" not in methods


def test_opp_unresolvable_warns_and_counts():
    # The C-16 visibility signal: an unresolvable opponent for a team with penalties warns and increments
    # report.penalty_opponent_unresolved.
    with pytest.warns(UserWarning, match="opponent of"):
        _actions, report = _convert(
            [
                _penalty(2, 100.0, "home", "P1", "CLU_H"),
                _kickoff(2, 110.0, "unknown"),
                _finalwhistle(2, 3000.0, "0:0"),
            ]
        )
    assert report.penalty_opponent_unresolved == 1


@pytest.mark.parametrize("match", IDSSE_MATCHES)
def test_idsse_opponent_always_resolved(match):
    # On real DFL every period has a kick-off and both teams kick off, so the opponent always resolves and the
    # C-16 counter stays 0 (the degenerate-feed signal never fires on real data).
    _, report = _convert_idsse(match)
    assert report.penalty_opponent_unresolved == 0


@pytest.mark.parametrize("match", IDSSE_MATCHES)
def test_idsse_has_no_owngoal_rows(match):
    # The 7 IDSSE matches contain 0 OwnGoal events, so the decision-3 fix leaves IDSSE goal totals unchanged.
    actions, _ = _convert_idsse(match)
    assert (actions["result_id"] == _OWNGOAL).sum() == 0


# ---------------------------------------------------------------------------
# B-01 / C-1: an "unknown" team label must not break opponent resolution
# ---------------------------------------------------------------------------


def test_unknown_team_label_does_not_break_penalty_opponent():
    # Same shape as test_elimination_2_pens_1_goal, with an extra "unknown"-team event present. The opponent
    # is taken from the KickOff teams, so pen1 still resolves kickoff_order and pen2 score_elimination.
    actions, _ = _convert(
        [
            {"event_type": "FinalWhistle", "period": 1, "timestamp_seconds": 1.0, "team": "unknown"},
            _kickoff(2, 0.0, "home"),
            _penalty(2, 100.0, "home", "P1", "CLU_H"),
            _kickoff(2, 110.0, "away"),
            _penalty(2, 200.0, "home", "P2", "CLU_H"),
            _finalwhistle(2, 3000.0, "1:0"),
        ]
    )
    pens = actions[actions["type_id"] == SHOT_PENALTY].sort_values("time_seconds")
    assert pens.iloc[0]["penalty_result_method"] == "kickoff_order"
    assert pens.iloc[1]["penalty_result_method"] == "score_elimination"


@pytest.mark.parametrize("match", ["J03WMX", "J03WOH"])
def test_real_idsse_penalty_without_final_score_resolves_by_kickoff(match):
    # With the final score blanked, the real IDSSE penalty must resolve via step 3 (kickoff_order) on real
    # DFL data whose team labels include "unknown" (the premise test the C1 gap lacked).
    actions, _ = _convert_idsse_no_score(match)
    row = actions[actions["type_id"] == SHOT_PENALTY].iloc[-1]
    assert (row["result_id"], row["result_source"], row["penalty_result_method"]) == (
        SUCCESS,
        "native",
        "kickoff_order",
    )


def test_real_idsse_j03wr9_penalty_never_continued_play():
    actions, _ = _convert_idsse_no_score("J03WR9")
    pens = actions[actions["type_id"] == SHOT_PENALTY]
    assert "continued_play" not in set(pens["penalty_result_method"])


# ---------------------------------------------------------------------------
# Discriminating tests (B-03): each pairing/credit condition must be load-bearing
# ---------------------------------------------------------------------------


def _woodwork_shot(period, t, team, player):
    return _shot(period, t, team, player, "woodwork")


def test_deflection_other_player_does_not_pair():
    # the woodwork deflection's player differs from the shot's player -> no pairing.
    actions, report = _convert(
        [
            _woodwork_shot(1, 100.0, "home", "S1"),
            _balldeflection(1, 101.0, "home", "D9", "postLeft", 0.0, 32.0),  # different player
            _finalwhistle(2, 3000.0, "0:0"),
        ]
    )
    assert actions["shot_woodwork_type"].notna().sum() == 0
    assert report.deflection_counts.get("woodwork_paired", 0) == 0


def test_deflection_beyond_3s_does_not_pair():
    actions, report = _convert(
        [
            _woodwork_shot(1, 100.0, "home", "S1"),
            _balldeflection(1, 103.5, "home", "S1", "postLeft", 0.0, 32.0),  # > 3 s after the shot
            _finalwhistle(2, 3000.0, "0:0"),
        ]
    )
    assert actions["shot_woodwork_type"].notna().sum() == 0
    assert report.deflection_counts.get("woodwork_paired", 0) == 0


def test_deflection_non_woodwork_shot_does_not_pair():
    # the preceding shot is not shot_outcome_type == "woodwork" -> no pairing.
    actions, report = _convert(
        [
            _shot(1, 100.0, "home", "S1", "saved"),
            _balldeflection(1, 101.0, "home", "S1", "postLeft", 0.0, 32.0),
            _finalwhistle(2, 3000.0, "0:0"),
        ]
    )
    assert actions["shot_woodwork_type"].notna().sum() == 0
    assert report.deflection_counts.get("woodwork_paired", 0) == 0


def test_same_team_kickoff_does_not_prove_penalty():
    # the kick-off after the penalty is by the SAME team, not the opponent -> step 3 must not fire.
    # Both teams kick off (the away period-start kick-off at t=0), so _match_team_pair resolves the opponent
    # and _first_restart_signal reaches the opponent filter. Without this, match_teams = [home] and the filter
    # is never exercised (B-03 residual: the frame was one-team, so M6 "any team's kick-off counts" survived).
    actions, _ = _convert(
        [
            _kickoff(2, 0.0, "away"),  # period-start kick-off by the opponent -> two teams in the frame
            _penalty(2, 100.0, "home", "P1", "CLU_H"),
            _kickoff(2, 110.0, "home"),  # same team, not the conceding opponent
            _penalty(2, 200.0, "home", "P2", "CLU_H"),
            _finalwhistle(2, 3000.0, "1:0"),
        ]
    )
    pens = actions[actions["type_id"] == SHOT_PENALTY].sort_values("time_seconds")
    assert "kickoff_order" not in set(pens["penalty_result_method"])


def test_direct_freekick_other_player_does_not_pair():
    # a FreeKick + ShotAtGoal by a DIFFERENT player is not a direct free kick -> both rows survive.
    actions, _ = _convert(
        [
            _freekick(1, 100.0, "home", "P1"),
            _shot(1, 101.2, "home", "P2", "saved", after_fk=True),  # different player
            _finalwhistle(2, 3000.0, "0:0"),
        ]
    )
    assert (actions["type_id"] == FREEKICK_SHORT).sum() == 1
    assert (actions["type_id"] == SHOT_FREEKICK).sum() == 1


def test_retaken_penalty_yields_one_shot():
    # Two Penalty events; the first flagged retaken -> only the second is a shot; one retake counted.
    actions, report = _convert(
        [
            _penalty(2, 100.0, "home", "P1", "CLU_H", retaken="true"),
            _penalty(2, 101.0, "home", "P1", "CLU_H", retaken="false"),
            _finalwhistle(2, 3000.0, "1:0"),
        ]
    )
    pens = actions[actions["type_id"] == SHOT_PENALTY]
    assert len(pens) == 1
    assert report.penalty_retakes == 1
