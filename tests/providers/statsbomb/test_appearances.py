"""StatsBomb events-only keeper-appearance extractor (TF-59 PR1, spec §5.5).

Appearances are emitted PER ``(team, period)`` (the port's key), so an unsubbed keeper in a
two-period match has TWO rows. The happy-path test runs on the real committed match ``7584.json``
(Belgium vs Japan, WC2018): two Starting XI keepers (Courtois 3509, Kawashima 3175), four
substitutions NONE of which is a keeper -> each starter is decomposed into a period-1 AND a period-2
row, all open (end == +inf). ``synthetic_gk_sub.json`` exercises a single-period keeper sub;
``synthetic_gk_sub_halftime.json`` exercises a half-time keeper change (period-2 t=0, the
degenerate no-tenure slice) plus a mid-period-2 keeper sub (the honest period split).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from silly_kicks.keeper_identity import (
    KEEPER_APPEARANCE_COLUMNS,
    validate_keeper_appearances,
)
from silly_kicks.providers.statsbomb.appearances import extract_keeper_appearances

RAW = pathlib.Path(__file__).parents[2] / "datasets/statsbomb/raw/events"


def _load(name: str) -> list[dict]:
    with open(RAW / name, encoding="utf-8") as fh:
        return json.load(fh)


def test_full_match_two_keepers_open_intervals() -> None:
    # Real committed match: 2 periods, no GK sub -> each starter is decomposed into a period-1 AND a
    # period-2 row, every one opening at 0.0 and open (end == +inf) to the period end.
    ev = _load("7584.json")
    ap = extract_keeper_appearances(ev, game_id="7584")
    starters = ap[ap["start_time_seconds"] == 0.0]
    assert starters["team_id"].nunique() == 2  # one starting keeper per team
    assert len(starters) == 4  # 2 keepers x 2 periods, each period opening at 0.0
    assert np.isinf(ap["end_time_seconds"]).sum() == 4  # every period row open to the period end


def test_gk_sub_creates_two_intervals() -> None:
    ev = _load("synthetic_gk_sub.json")
    ap = extract_keeper_appearances(ev, game_id="SYN")
    counts = ap.groupby(["team_id", "period_id"]).size()
    assert (counts >= 2).any()  # the subbed team has starter + replacement
    assert (ap["source"] == "sub_events").any()  # the replacement interval is sub-sourced


def test_output_is_a_validated_port() -> None:
    # The extractor returns a validated ``KeeperAppearances`` frame: exact columns, and
    # ``validate_keeper_appearances`` accepts it unchanged (idempotent re-validation).
    ev = _load("synthetic_gk_sub.json")
    ap = extract_keeper_appearances(ev, game_id="SYN")
    assert list(ap.columns) == list(KEEPER_APPEARANCE_COLUMNS)
    validate_keeper_appearances(ap)


def test_starter_closed_at_sub_time_replacement_opens_there() -> None:
    # The subbed keeper's interval closes at the sub minute; the replacement opens there to +inf.
    ev = _load("synthetic_gk_sub.json")
    ap = extract_keeper_appearances(ev, game_id="SYN")
    subbed = ap[ap["source"] == "sub_events"]
    assert len(subbed) == 1
    sub_row = subbed.iloc[0]
    # The replacement interval is open to the period end.
    assert np.isinf(sub_row["end_time_seconds"])
    # The starter for the SAME team-period closes exactly where the replacement opens.
    team_period = ap[(ap["team_id"] == sub_row["team_id"]) & (ap["period_id"] == sub_row["period_id"])]
    starter = team_period[team_period["source"] == "starting_xi"].iloc[0]
    assert starter["start_time_seconds"] == 0.0
    assert starter["end_time_seconds"] == sub_row["start_time_seconds"]
    assert starter["end_time_seconds"] > 0.0


def test_halftime_gk_sub_splits_by_period() -> None:
    # Half-time GK change (team 100 keeper subbed at period-2 t=0) + a mid-period-2 GK change
    # (team 200 keeper subbed at period-2 t=300). The per-period decomposition must split cleanly at
    # the period boundary -- NOT fabricate a period-1 row that "ends" at a period-2 time.
    ev = _load("synthetic_gk_sub_halftime.json")
    ap = extract_keeper_appearances(ev, game_id="SYNHT")

    # Team 100 outgoing starter (9001): a period-1 row open to +inf, and NO period-2 tenure
    # (subbed at the very start of period 2 -> the period-2 slice has start >= end and is dropped).
    out_starter = ap[(ap["team_id"] == 100) & (ap["player_id"] == 9001)]
    assert set(out_starter["period_id"]) == {1}
    assert np.isinf(out_starter.iloc[0]["end_time_seconds"])
    assert out_starter.iloc[0]["source"] == "starting_xi"

    # Team 100 half-time replacement (9098): a single period-2 row, start=0.0, end=+inf, sub-sourced.
    repl = ap[(ap["team_id"] == 100) & (ap["player_id"] == 9098)]
    assert list(repl["period_id"]) == [2]
    r = repl.iloc[0]
    assert r["start_time_seconds"] == 0.0
    assert np.isinf(r["end_time_seconds"])
    assert r["source"] == "sub_events"

    # Team 200 starter (9201) subbed mid-period-2 at t=300: its period-2 row ends at 300 (period_id=2,
    # NOT a fabricated period_id=1 row ending at 300); its period-1 row stays open.
    t200_starter = ap[(ap["team_id"] == 200) & (ap["player_id"] == 9201)]
    p2 = t200_starter[t200_starter["period_id"] == 2].iloc[0]
    assert p2["start_time_seconds"] == 0.0
    assert p2["end_time_seconds"] == 300.0
    p1 = t200_starter[t200_starter["period_id"] == 1].iloc[0]
    assert np.isinf(p1["end_time_seconds"])
    # No period-1 row was fabricated with a finite mid-period-2 end.
    assert not (np.isfinite(t200_starter[t200_starter["period_id"] == 1]["end_time_seconds"])).any()

    # Team 200 mid-period-2 replacement (9298): period-2, start=300.0, sub-sourced.
    repl2 = ap[(ap["team_id"] == 200) & (ap["player_id"] == 9298)]
    assert list(repl2["period_id"]) == [2]
    assert repl2.iloc[0]["start_time_seconds"] == 300.0
    assert repl2.iloc[0]["source"] == "sub_events"


# --- Player Off (no replacement) + Tactical Shift emergency keeper --------------------------------


def _starting_xi(team_id: int, keeper_id: int) -> dict:
    return {
        "type": {"name": "Starting XI"},
        "period": 1,
        "timestamp": "00:00:00.000",
        "team": {"id": team_id},
        "tactics": {"lineup": [{"player": {"id": keeper_id}, "position": {"name": "Goalkeeper"}}]},
    }


def test_keeper_player_off_ends_tenure() -> None:
    # A keeper sent off (Player Off, NO Substitution) ends the keeper's tenure at the event time; with
    # no subsequent Tactical Shift the team simply has no keeper for the remainder (honest -- the
    # emergency keeper is unidentifiable from events alone, never fabricated).
    events = [
        _starting_xi(10, 901),
        _starting_xi(20, 902),
        {
            "type": {"name": "Player Off"},
            "period": 2,
            "timestamp": "00:30:00.000",
            "team": {"id": 10},
            "player": {"id": 901},
        },
    ]
    ap = extract_keeper_appearances(events, game_id="POFF")
    # 901's period-2 tenure ends at the Player Off (1800s), NOT open to the period end.
    p2 = ap[(ap["player_id"] == 901) & (ap["period_id"] == 2)].iloc[0]
    assert p2["end_time_seconds"] == 1800.0
    assert not np.isinf(p2["end_time_seconds"])
    # No emergency keeper was fabricated for team 10 after the Player Off.
    team10_p2 = ap[(ap["team_id"] == 10) & (ap["period_id"] == 2)]
    assert set(team10_p2["player_id"]) == {901}
    # Team 20's keeper is unaffected (open to the period end).
    assert bool(np.isinf(ap[ap["player_id"] == 902]["end_time_seconds"]).all())
    validate_keeper_appearances(ap)


def test_player_off_then_tactical_shift_identifies_emergency_keeper() -> None:
    # After a keeper Player Off, a Tactical Shift naming a new Goalkeeper IS the emergency keeper: the
    # sent-off keeper's tenure ends at the Player Off, the emergency keeper opens at the Tactical Shift.
    events = [
        _starting_xi(10, 901),
        _starting_xi(20, 902),
        {
            "type": {"name": "Player Off"},
            "period": 2,
            "timestamp": "00:30:00.000",
            "team": {"id": 10},
            "player": {"id": 901},
        },
        {
            "type": {"name": "Tactical Shift"},
            "period": 2,
            "timestamp": "00:30:05.000",
            "team": {"id": 10},
            "tactics": {"lineup": [{"player": {"id": 950}, "position": {"name": "Goalkeeper"}}]},
        },
    ]
    ap = extract_keeper_appearances(events, game_id="EMER")
    # 901 ends at the Player Off (1800s).
    assert ap[(ap["player_id"] == 901) & (ap["period_id"] == 2)].iloc[0]["end_time_seconds"] == 1800.0
    # 950 (emergency keeper) opens at the Tactical Shift (1805s), open to the period end, emergency-sourced.
    emer = ap[ap["player_id"] == 950]
    assert list(emer["period_id"]) == [2]
    row = emer.iloc[0]
    assert row["start_time_seconds"] == 1805.0
    assert np.isinf(row["end_time_seconds"])
    assert row["source"] == "emergency_keeper"
    validate_keeper_appearances(ap)


def test_tactical_shift_with_unchanged_keeper_is_a_noop() -> None:
    # A normal Tactical Shift (formation change, keeper unchanged) must NOT create a spurious segment --
    # the output is identical to the same match without the Tactical Shift (byte-identical no-op).
    base = [_starting_xi(10, 901), _starting_xi(20, 902)]
    shift = {
        "type": {"name": "Tactical Shift"},
        "period": 1,
        "timestamp": "00:20:00.000",
        "team": {"id": 10},
        "tactics": {"lineup": [{"player": {"id": 901}, "position": {"name": "Goalkeeper"}}]},
    }
    key = ["team_id", "player_id", "period_id"]
    ap_no_shift = extract_keeper_appearances(base, game_id="NS").sort_values(key).reset_index(drop=True)
    ap_with_shift = extract_keeper_appearances([*base, shift], game_id="NS").sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(ap_no_shift, ap_with_shift)
