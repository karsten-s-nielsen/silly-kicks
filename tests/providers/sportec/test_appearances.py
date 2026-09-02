"""Sportec / DFL keeper-appearance extractor (TF-59 PR1, spec §5.5, Part C).

The unit builds a SYNTHETIC DFL bronze DataFrame (no XML fixture) matching the real
``_IDSSE_EVENTS_BRONZE_COLS`` shape -- ``period`` / ``timestamp_seconds`` (period-relative) +
``sub_player_in`` / ``sub_player_out`` / ``sub_playing_position`` / ``sub_team`` +
``other_action_player`` / ``other_action_team`` / ``other_action_player_becomes_goalkeeper`` -- and a
real :class:`~silly_kicks.providers.sportec.parse.MatchInfo` (all 8 frozen fields). ``PlayingPosition
== "TW"`` in the DFL match-info marks exactly the two STARTING keepers (observed on both committed
fixtures), so ``MatchInfo.gk_player_ids`` = one starter per team.

The ``@e2e`` test parses a real public IDSSE match through the pining loader; it skips when the pining
API is unreachable.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silly_kicks.keeper_identity import validate_keeper_appearances
from silly_kicks.providers.sportec.appearances import extract_keeper_appearances
from silly_kicks.providers.sportec.parse import MatchInfo


def _match_info() -> MatchInfo:
    # Real MatchInfo fields (parse.py:1852-1863), ALL required (frozen dataclass). DFL string ids.
    # player_team_map maps person_id -> "home"/"away" (the REAL semantics of _parse_teams), NOT the CLU
    # id. gk_player_ids = the two STARTING keepers (PlayingPosition="TW" is starters-only in real DFL).
    return MatchInfo(
        home_team_id="DFL-CLU-H",
        away_team_id="DFL-CLU-A",
        player_team_map={"GK-H": "home", "GK-H2": "home", "GK-A": "away", "OUT-A": "away"},
        gk_player_ids=frozenset({"GK-H", "GK-A"}),
        competition_id="DFL-COM-000001",
        season_id="DFL-SEA-0001K6",
        pitch_x=105.0,
        pitch_y=68.0,
    )


def _bronze() -> pd.DataFrame:
    # DFL bronze rows (real column names). period-relative time in `timestamp_seconds`; `period` (NOT
    # `period_id`). Two filler period-1 keeper actions establish periods={1,2} and make the starter
    # earliest-action-unambiguous. One TW substitution (home GK-H -> GK-H2) + one emergency keeper
    # (away OUT-A becomes GK), both in period 2.
    def _row(**kw: object) -> dict[str, object]:
        base: dict[str, object] = {
            "event_type": None,
            "period": None,
            "timestamp_seconds": None,
            "player_id": None,
            "sub_player_in": None,
            "sub_player_out": None,
            "sub_playing_position": None,
            "sub_team": None,
            "other_action_player": None,
            "other_action_team": None,
            "other_action_player_becomes_goalkeeper": None,
        }
        base.update(kw)
        return base

    rows = [
        _row(event_type="Play", period=1, timestamp_seconds=10.0, player_id="GK-H"),
        _row(event_type="Play", period=1, timestamp_seconds=12.0, player_id="GK-A"),
        _row(
            event_type="Substitution",
            period=2,
            timestamp_seconds=600.0,
            sub_player_out="GK-H",
            sub_player_in="GK-H2",
            sub_playing_position="TW",
            sub_team="DFL-CLU-H",
        ),
        _row(
            event_type="OtherPlayerAction",
            period=2,
            timestamp_seconds=500.0,
            other_action_player="OUT-A",
            other_action_team="DFL-CLU-A",
            other_action_player_becomes_goalkeeper="true",
        ),
    ]
    return pd.DataFrame(rows)


def test_gk_sub_and_emergency_keeper() -> None:
    ap = extract_keeper_appearances(_match_info(), _bronze(), game_id="SYN")

    # The TW-subbed team (DFL-CLU-H) has >= 2 keeper intervals (GK-H then GK-H2).
    home = ap[ap["team_id"] == "DFL-CLU-H"]
    assert home["player_id"].nunique() >= 2
    assert ap.groupby(["team_id", "period_id"]).size().max() >= 2

    # The emergency keeper (OUT-A) is emitted with the dedicated source token.
    assert (ap["source"] == "emergency_keeper").any()

    # String ids are preserved un-coerced.
    assert set(ap["player_id"]) >= {"GK-H", "GK-H2", "GK-A", "OUT-A"}

    # A validated port with the right shape.
    validate_keeper_appearances(ap)


def test_starters_open_at_period_one() -> None:
    ap = extract_keeper_appearances(_match_info(), _bronze(), game_id="SYN")
    # Both starters open period 1 at 0.0 (starting_xi), open to the period end.
    p1_starters = ap[(ap["period_id"] == 1) & (ap["source"] == "starting_xi")]
    assert set(p1_starters["player_id"]) == {"GK-H", "GK-A"}
    assert (p1_starters["start_time_seconds"] == 0.0).all()
    assert bool(np.isinf(p1_starters["end_time_seconds"]).all())


def test_home_keeper_change_splits_at_sub_time() -> None:
    ap = extract_keeper_appearances(_match_info(), _bronze(), game_id="SYN")
    # GK-H's period-2 tenure ends exactly at the TW-sub time; GK-H2 opens there to the period end.
    gk_h_p2 = ap[(ap["player_id"] == "GK-H") & (ap["period_id"] == 2)].iloc[0]
    gk_h2_p2 = ap[(ap["player_id"] == "GK-H2") & (ap["period_id"] == 2)].iloc[0]
    assert gk_h_p2["end_time_seconds"] == 600.0
    assert gk_h2_p2["start_time_seconds"] == 600.0
    assert gk_h2_p2["source"] == "sub_events"
    assert bool(np.isinf(gk_h2_p2["end_time_seconds"]))


def _match_info_two_home_tw() -> MatchInfo:
    # The home team has TWO TW keepers: the ACTING starter "GK-H-Z" (lexically LARGER) and a bench
    # keeper "GK-H-A" (lexically SMALLER, never acts). A naive min-id tie-break would wrongly pick
    # GK-H-A; `_resolve_starters` must disambiguate via `_earliest_acting` and pick the one that ACTS.
    # `player_team_map` routes both to "home"; the away side keeps its single starter GK-A.
    return MatchInfo(
        home_team_id="DFL-CLU-H",
        away_team_id="DFL-CLU-A",
        player_team_map={"GK-H-Z": "home", "GK-H-A": "home", "GK-A": "away"},
        gk_player_ids=frozenset({"GK-H-Z", "GK-H-A", "GK-A"}),
        competition_id="DFL-COM-000001",
        season_id="DFL-SEA-0001K6",
        pitch_x=105.0,
        pitch_y=68.0,
    )


def _bronze_row(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "event_type": None,
        "period": None,
        "timestamp_seconds": None,
        "player_id": None,
        "sub_player_in": None,
        "sub_player_out": None,
        "sub_playing_position": None,
        "sub_team": None,
        "other_action_player": None,
        "other_action_team": None,
        "other_action_player_becomes_goalkeeper": None,
    }
    base.update(kw)
    return base


def test_multiple_tw_keepers_pick_earliest_actor_not_min_id() -> None:
    # Only the starter (GK-H-Z) has an on-ball action; the bench keeper (GK-H-A) never acts.
    bronze = pd.DataFrame(
        [
            _bronze_row(event_type="Play", period=1, timestamp_seconds=10.0, player_id="GK-H-Z"),
            _bronze_row(event_type="Play", period=1, timestamp_seconds=12.0, player_id="GK-A"),
        ]
    )
    ap = extract_keeper_appearances(_match_info_two_home_tw(), bronze, game_id="SYN")

    # The extractor seeds the ACTING starter (GK-H-Z), not the lexically-smaller never-acting keeper.
    home = ap[ap["team_id"] == "DFL-CLU-H"]
    assert set(home["player_id"]) == {"GK-H-Z"}
    # The never-acting bench keeper (GK-H-A) gets NO interval at all.
    assert "GK-H-A" not in set(ap["player_id"])
    validate_keeper_appearances(ap)


def test_emergency_keeper_id_carried_in_flag_column() -> None:
    # `_is_truthy_flag_only` fallback: `other_action_player` is null, but the flag column itself
    # carries the emergency keeper's player id (an alternative bronze encoding), so it is used as the
    # incoming keeper for `other_action_team`.
    bronze = pd.DataFrame(
        [
            _bronze_row(event_type="Play", period=1, timestamp_seconds=10.0, player_id="GK-H"),
            _bronze_row(event_type="Play", period=1, timestamp_seconds=12.0, player_id="GK-A"),
            _bronze_row(
                event_type="OtherPlayerAction",
                period=2,
                timestamp_seconds=500.0,
                other_action_player=None,
                other_action_team="DFL-CLU-A",
                other_action_player_becomes_goalkeeper="OUT-A2",
            ),
        ]
    )
    ap = extract_keeper_appearances(_match_info(), bronze, game_id="SYN")

    emergency = ap[ap["source"] == "emergency_keeper"]
    assert set(emergency["player_id"]) == {"OUT-A2"}
    assert (emergency["team_id"] == "DFL-CLU-A").all()
    validate_keeper_appearances(ap)


# --- e2e: real public IDSSE match through the pining loader --------------------------------------


@pytest.mark.e2e
def test_idsse_real_match_happy_path() -> None:
    import importlib.util

    spec = importlib.util.find_spec("scripts._loader_pining")
    if spec is None:  # scripts/ not importable in this layout
        pytest.skip("scripts._loader_pining not importable")
    from scripts import _loader_pining as lp
    from silly_kicks.providers.sportec.parse import parse_dfl_events, parse_dfl_match_info

    token = lp._resolve_token(None)
    base = lp._base_url()
    try:
        matches = lp._list_matches("idsse", token, base)
    except Exception as exc:  # network/API unreachable -> skip (the plan's "without pining token")
        pytest.skip(f"pining IDSSE list unreachable: {type(exc).__name__}")
    if not matches:
        pytest.skip("no public IDSSE matches returned")

    m = matches[0]
    match_id = str(m["id"])
    artifacts = m.get("artifacts", {})
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        try:
            meta_path = lp._download_to_temp(
                "idsse", match_id, "metadata", token, base, tmp_dir, filename=artifacts.get("metadata")
            )
            events_path = lp._download_to_temp(
                "idsse", match_id, "events", token, base, tmp_dir, filename=artifacts.get("events")
            )
        except Exception as exc:  # transient S3/redirect blip -> skip, not fail
            pytest.skip(f"pining IDSSE download unreachable: {type(exc).__name__}")

        bare_id = match_id.removeprefix("DFL-MAT-")
        mi = parse_dfl_match_info(str(meta_path))
        events_bronze = parse_dfl_events(str(events_path), match_info=mi, match_id=bare_id)

    ap = extract_keeper_appearances(mi, events_bronze, game_id=match_id)
    validate_keeper_appearances(ap)  # validates on the real bronze shape

    # The 7 public IDSSE matches have no GK sub -> exactly one starting-keeper interval per team-period,
    # all `starting_xi`, all open to the period end.
    periods = sorted(ap["period_id"].unique())
    assert len(periods) >= 2  # at least two halves
    for p in periods:
        starters = ap[(ap["period_id"] == p) & (ap["source"] == "starting_xi")]
        assert starters["team_id"].nunique() == 2  # two starting keepers present each period
    assert (ap["source"] == "starting_xi").all()  # no substitution in the public corpus
    assert bool(np.isinf(ap["end_time_seconds"]).all())
