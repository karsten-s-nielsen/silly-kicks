"""Gradient Sports keeper-appearance extractor (TF-59 PR1, spec §5.5).

The unit builds a SYNTHETIC GS ``events`` list (raw ``gameEvents`` envelopes) + ``roster`` list
matching the real WC2022 feed shape (verified 2026-09-01): a roster record carries a nested
``{"player": {"id": P}, "team": {"id": T}, "positionGroupType": "GK"|...}``; a substitution is a
``{"gameEvents": {"gameEventType": "SUB", "startGameClock": <period-relative int>, "period": K,
"teamId": <may be null>, "playerOffId": OUT, "playerOnId": IN}}`` envelope.

The ``@e2e`` test pulls ONE real WC2022 GS match's ``events`` + ``roster`` through the owner-tier
pining loader; it skips when the pining token / API is unavailable.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from silly_kicks.keeper_identity import validate_keeper_appearances
from silly_kicks.providers.gradientsports.appearances import extract_keeper_appearances


def _roster() -> list[dict]:  # two starting GKs (901 team10, 902 team20) + a bench GK (999 team20) + outfielders
    return [
        {"player": {"id": 901}, "team": {"id": 10}, "positionGroupType": "GK"},
        {"player": {"id": 902}, "team": {"id": 20}, "positionGroupType": "GK"},
        {"player": {"id": 999}, "team": {"id": 20}, "positionGroupType": "GK"},
        {"player": {"id": 500}, "team": {"id": 10}, "positionGroupType": "DEF"},
        {"player": {"id": 501}, "team": {"id": 10}, "positionGroupType": "MID"},
    ]


def _events_gk_sub() -> list[dict]:  # 2 periods present; team-20 keeper 902 subbed for 999 at period-2 t=130
    return [
        # Ordinary on-ball keeper actions identify the STARTERS (901 team10, 902 team20). Without an
        # action signal a starter is indistinguishable from an unused bench keeper (the real-data
        # defect fixed here), so a realistic stream carries the starters' early actions.
        {"gameEvents": {"gameEventType": "PASS", "startGameClock": 10, "period": 1, "playerId": 901, "teamId": 10}},
        {"gameEvents": {"gameEventType": "PASS", "startGameClock": 12, "period": 1, "playerId": 902, "teamId": 20}},
        {
            "gameEvents": {
                "gameEventType": "SUB",
                "startGameClock": 130,
                "period": 2,
                "playerOffId": 902,
                "playerOnId": 999,
                "teamId": None,
            }
        },
        {
            "gameEvents": {
                "gameEventType": "SUB",
                "startGameClock": 3000,
                "period": 2,
                "playerOffId": 500,
                "playerOnId": 501,
                "teamId": 10,
            }
        },  # non-keeper sub
    ]


def test_gk_sub_distinct_keepers_and_non_keeper_ignored() -> None:
    ap = extract_keeper_appearances(_events_gk_sub(), _roster(), game_id="gs1")
    team10 = {int(x) for x in ap[ap["team_id"] == 10]["player_id"]}
    team20 = {int(x) for x in ap[ap["team_id"] == 20]["player_id"]}
    assert team10 == {901}  # team 10 keeper plays throughout; non-keeper sub ignored
    assert team20 == {902, 999}  # team 20: starter 902 then replacement 999
    assert (ap["source"] == "sub_events").any()  # the replacement segment is sub-sourced


def test_per_period_rows() -> None:
    ap = extract_keeper_appearances(_events_gk_sub(), _roster(), game_id="gs1")
    # team-10 unsubbed keeper 901 has one row per period (both periods present)
    t10 = ap[(ap["team_id"] == 10)]
    assert set(t10["period_id"]) == {1, 2}
    # team-20 starter 902: p1 open + p2 closed at 130; replacement 999: p2 from 130
    t20_902 = ap[(ap["team_id"] == 20) & (ap["player_id"].astype(str) == "902")]
    assert set(t20_902["period_id"]) == {1, 2}


def test_sub_time_splits_team20_starter_and_replacement() -> None:
    ap = extract_keeper_appearances(_events_gk_sub(), _roster(), game_id="gs1")
    # 902's period-2 tenure ends exactly at the null-teamId sub time (team derived from the
    # OUTGOING player's roster team); 999 opens there to the period end (sub_events).
    p902 = ap[(ap["player_id"].astype(str) == "902") & (ap["period_id"] == 2)].iloc[0]
    p999 = ap[(ap["player_id"].astype(str) == "999") & (ap["period_id"] == 2)].iloc[0]
    assert p902["end_time_seconds"] == 130.0
    assert p999["start_time_seconds"] == 130.0
    assert p999["source"] == "sub_events"
    assert bool(np.isinf(p999["end_time_seconds"]))


def test_starters_open_at_period_one() -> None:
    ap = extract_keeper_appearances(_events_gk_sub(), _roster(), game_id="gs1")
    # Both starters open period 1 at 0.0 (starting_xi), open to the period end.
    p1_starters = ap[(ap["period_id"] == 1) & (ap["source"] == "starting_xi")]
    assert {int(x) for x in p1_starters["player_id"]} == {901, 902}
    assert (p1_starters["start_time_seconds"] == 0.0).all()
    assert bool(np.isinf(p1_starters["end_time_seconds"]).all())
    validate_keeper_appearances(ap)


def test_real_world_shape_acting_starter_not_smallest_id() -> None:
    # Real WC2022 shape (match 10502 team 366): THREE roster GKs per team, and the substitutions are
    # OUTFIELDERS -- so NO GK is ever a `playerOnId`. The "GK not seen as a playerOnId" rule can't
    # distinguish the starter from a bench keeper, and the SMALLEST-id GK (8020 / 21000) is an unused
    # bench keeper with 0 on-ball actions. Only the ACTING starter must be picked.
    roster = [
        {"player": {"id": 11241}, "team": {"id": 366}, "positionGroupType": "GK"},  # starter (acts early)
        {"player": {"id": 8020}, "team": {"id": 366}, "positionGroupType": "GK"},  # bench, SMALLEST id, 0 actions
        {"player": {"id": 11099}, "team": {"id": 366}, "positionGroupType": "GK"},  # bench, 0 actions
        {"player": {"id": 22000}, "team": {"id": 900}, "positionGroupType": "GK"},  # opp starter (acts early)
        {"player": {"id": 21000}, "team": {"id": 900}, "positionGroupType": "GK"},  # opp bench, SMALLEST id, 0 actions
        {"player": {"id": 5001}, "team": {"id": 366}, "positionGroupType": "DEF"},
        {"player": {"id": 6001}, "team": {"id": 900}, "positionGroupType": "DEF"},
    ]
    events = [
        {"gameEvents": {"gameEventType": "PASS", "startGameClock": 8, "period": 1, "playerId": 11241, "teamId": 366}},
        {"gameEvents": {"gameEventType": "PASS", "startGameClock": 9, "period": 1, "playerId": 22000, "teamId": 900}},
        # an OUTFIELDER substitution (not a keeper change) -- the only SUB in the stream.
        {
            "gameEvents": {
                "gameEventType": "SUB",
                "startGameClock": 2000,
                "period": 2,
                "playerOffId": 5001,
                "playerOnId": 5002,
                "teamId": 366,
            }
        },
    ]
    ap = extract_keeper_appearances(events, roster, game_id="10502")
    # The ACTING starter is picked, NOT the smallest canonical id (8020 / 21000).
    assert {int(x) for x in ap[ap["team_id"] == 366]["player_id"]} == {11241}
    assert {int(x) for x in ap[ap["team_id"] == 900]["player_id"]} == {22000}
    # No interval is emitted for a never-acting bench keeper.
    all_players = {int(x) for x in ap["player_id"]}
    assert all_players.isdisjoint({8020, 11099, 21000})
    # And it is a valid port (one starting_xi interval per team-period, open to the end).
    assert (ap["source"] == "starting_xi").all()
    validate_keeper_appearances(ap)


def test_flat_roster_and_null_teamid_derivation() -> None:
    # Flat roster shape (``playerId`` / ``teamId``) + case-variant / GOALKEEPER position group.
    flat_roster = [
        {"playerId": 901, "teamId": 10, "positionGroupType": "goalkeeper"},
        {"playerId": 902, "teamId": 20, "positionGroupType": "GOALKEEPER"},
        {"playerId": 999, "teamId": 20, "positionGroupType": "Gk"},
        {"playerId": 500, "teamId": 10, "positionGroupType": "DEF"},
    ]
    ap = extract_keeper_appearances(_events_gk_sub(), flat_roster, game_id="gs1")
    assert {int(x) for x in ap[ap["team_id"] == 20]["player_id"]} == {902, 999}
    assert {int(x) for x in ap[ap["team_id"] == 10]["player_id"]} == {901}


def test_no_subs_one_interval_per_team_period() -> None:
    events = [
        {"gameEvents": {"gameEventType": "PASS", "startGameClock": 5, "period": 1, "playerId": 901, "teamId": 10}},
        {"gameEvents": {"gameEventType": "PASS", "startGameClock": 5, "period": 2, "playerId": 902, "teamId": 20}},
    ]
    ap = extract_keeper_appearances(events, _roster(), game_id="gs1")
    assert (ap["source"] == "starting_xi").all()
    assert bool(np.isinf(ap["end_time_seconds"]).all())
    # Two starters x two periods -> 4 open intervals.
    assert len(ap) == 4


def test_pure_inputs_not_mutated() -> None:
    events = _events_gk_sub()
    roster = _roster()
    events_snapshot = json.loads(json.dumps(events))
    roster_snapshot = json.loads(json.dumps(roster))
    extract_keeper_appearances(events, roster, game_id="gs1")
    assert events == events_snapshot
    assert roster == roster_snapshot


# --- e2e: real WC2022 Gradient Sports match through the owner pining loader ------------------------


@pytest.mark.e2e
def test_gs_real_match_happy_path() -> None:
    import importlib.util

    spec = importlib.util.find_spec("scripts._loader_pining")
    if spec is None:  # scripts/ not importable in this layout
        pytest.skip("scripts._loader_pining not importable")
    from scripts import _loader_pining as lp

    try:
        token = lp._resolve_token(None)
    except Exception as exc:  # no PINING_FOR_THE_DATA_TOKEN -> skip (owner-tier)
        pytest.skip(f"pining token unavailable: {type(exc).__name__}")
    base = lp._base_url()
    try:
        matches = lp._list_matches("gradientsports", token, base)
    except Exception as exc:  # network/API unreachable -> skip
        pytest.skip(f"pining gradientsports list unreachable: {type(exc).__name__}")
    if not matches:
        pytest.skip("no gradientsports matches returned")

    m = matches[0]
    match_id = str(m["id"])
    artifacts = m.get("artifacts", {})
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        try:
            events_path = lp._download_to_temp(
                "gradientsports", match_id, "events", token, base, tmp_dir, filename=artifacts.get("events")
            )
            roster_path = lp._download_to_temp(
                "gradientsports", match_id, "roster", token, base, tmp_dir, filename=artifacts.get("roster")
            )
        except Exception as exc:  # transient S3/redirect blip -> skip, not fail
            pytest.skip(f"pining gradientsports download unreachable: {type(exc).__name__}")

        with open(events_path, encoding="utf-8") as fh:
            events = json.load(fh)
        with open(roster_path, encoding="utf-8") as fh:
            roster = json.load(fh)

    ap = extract_keeper_appearances(events, roster, game_id=match_id)
    validate_keeper_appearances(ap)  # validates on the real feed shape

    # Both starting keepers present: >= 2 keeper intervals total, and each team has a starting_xi
    # interval in the first period.
    assert len(ap) >= 2
    assert (ap["source"] == "starting_xi").any()
    first_period = sorted(ap["period_id"].unique())[0]
    starters_p1 = ap[(ap["period_id"] == first_period) & (ap["source"] == "starting_xi")]
    assert starters_p1["team_id"].nunique() == 2
