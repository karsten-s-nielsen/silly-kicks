"""Tests for ``scripts/_sb_raw.py`` -- the single-sourced StatsBomb raw-JSON flattener.

``flatten_events`` is an EXTRACTION of six copies of the same body
(``build_sb360_coverage.py::_adapt_events``,
``build_worldcup_fixture.py::_adapt_events_to_silly_kicks_input`` and inline adapters in
``tests/spadl/test_add_possessions.py``, ``tests/spadl/test_cross_provider_parity.py``,
``tests/invariants/_loaders.py`` and ``tests/test_xthreat_statsbomb_e2e.py``). The load-bearing
guarantee is that the extracted function reproduces those bodies' output byte-for-byte. Those six
call sites reduce to exactly TWO behaviourally-distinct paths -- the plain flatten and the
``surface_native=("possession",)`` variant -- so pinning both paths covers all six.

``_adapt_events_coverage_legacy`` / ``_adapt_events_worldcup_legacy`` below are the PERMANENT
characterization oracle: verbatim copies of the two pre-extraction bodies (the plain and the
possession-surfacing paths respectively). They are frozen on
purpose -- if ``flatten_events`` ever has to change behaviour, these and the assertions change
together, deliberately, so the drift is a reviewed edit rather than a silent one. (Inlined here
rather than a separate ``_legacy_adapt`` module: ``scripts`` is a namespace package on the test
path, so a sibling module would merge into it confusingly.)
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

import scripts._sb_raw as raw
from silly_kicks.spadl.statsbomb import EXPECTED_INPUT_COLUMNS, convert_to_actions

_EVENTS_FIXTURE = os.path.join(os.path.dirname(__file__), "..", "datasets", "statsbomb", "raw", "events", "7298.json")

# --- Frozen pre-extraction oracle (see module docstring) -------------------------------------

_LEGACY_TOP_LEVEL_KEYS = frozenset({"id", "period", "timestamp", "team", "player", "type", "location"})


def _adapt_events_coverage_legacy(events: list[dict], match_id: int) -> pd.DataFrame:
    """Verbatim ``build_sb360_coverage.py::_adapt_events`` (pre-extraction)."""
    return pd.DataFrame(
        [
            {
                "game_id": match_id,
                "event_id": e.get("id"),
                "period_id": e.get("period"),
                "timestamp": e.get("timestamp"),
                "team_id": (e.get("team") or {}).get("id"),
                "player_id": (e.get("player") or {}).get("id"),
                "type_name": (e.get("type") or {}).get("name"),
                "location": e.get("location"),
                "extra": {k: v for k, v in e.items() if k not in _LEGACY_TOP_LEVEL_KEYS},
            }
            for e in events
        ]
    )


def _adapt_events_worldcup_legacy(events: list[dict], match_id: int) -> pd.DataFrame:
    """Verbatim ``build_worldcup_fixture.py::_adapt_events_to_silly_kicks_input`` (pre-extraction)."""
    return pd.DataFrame(
        [
            {
                "game_id": match_id,
                "event_id": e.get("id"),
                "period_id": e.get("period"),
                "timestamp": e.get("timestamp"),
                "team_id": (e.get("team") or {}).get("id"),
                "player_id": (e.get("player") or {}).get("id"),
                "type_name": (e.get("type") or {}).get("name"),
                "location": e.get("location"),
                "extra": {k: v for k, v in e.items() if k not in _LEGACY_TOP_LEVEL_KEYS},
                "possession": e.get("possession"),
            }
            for e in events
        ]
    )


@pytest.fixture(scope="module")
def raw_events() -> list[dict]:
    with open(_EVENTS_FIXTURE, encoding="utf-8") as f:
        return json.load(f)


# --- Characterization: the de-fork guarantee -------------------------------------------------


def test_flatten_events_equals_coverage_legacy(raw_events):
    got = raw.flatten_events(raw_events, match_id=7298)
    pd.testing.assert_frame_equal(got, _adapt_events_coverage_legacy(raw_events, 7298))


def test_flatten_events_equals_worldcup_legacy(raw_events):
    got = raw.flatten_events(raw_events, match_id=7298, surface_native=("possession",))
    pd.testing.assert_frame_equal(got, _adapt_events_worldcup_legacy(raw_events, 7298))


def test_flatten_events_schema_contract(raw_events):
    cols = set(raw.flatten_events(raw_events, match_id=7298).columns)
    assert EXPECTED_INPUT_COLUMNS <= cols


def test_round_trip_through_convert_to_actions(raw_events):
    flat = raw.flatten_events(raw_events, match_id=7298)
    home = int(flat["team_id"].dropna().iloc[0])
    actions, _report = convert_to_actions(flat, home_team_id=home)
    assert len(actions) > 0
    assert {"type_id", "start_x", "start_y"}.issubset(actions.columns)


# --- The three new parsers (hand-built payloads; no network) ---------------------------------


def test_parse_freeze_frames_passthrough():
    recs = [
        {
            "event_uuid": "abc",
            "freeze_frame": [{"location": [1.0, 2.0], "teammate": True, "actor": False, "keeper": False}],
            "visible_area": [0.0, 0.0, 120.0, 0.0, 120.0, 80.0],
        }
    ]
    out = raw.parse_freeze_frames(recs)
    assert isinstance(out, list) and out[0]["event_uuid"] == "abc"
    assert out[0]["visible_area"] == recs[0]["visible_area"]


def test_parse_metadata_single_match_row():
    md = {"home_team": {"home_team_id": 7}, "xy_fidelity_version": "2", "shot_fidelity_version": "2"}
    out = raw.parse_metadata(md)
    assert out["home_team_id"] == 7
    assert out["xy_fidelity_version"] == 2
    assert out["shot_fidelity_version"] == 2


def test_parse_metadata_defaults_fidelity_to_one():
    md = {"home_team": {"home_team_id": 9}}
    out = raw.parse_metadata(md)
    assert out["home_team_id"] == 9
    assert out["xy_fidelity_version"] == 1
    assert out["shot_fidelity_version"] == 1


def test_parse_roster_lineups_shape():
    lineups = [
        {
            "team_id": 7,
            "lineup": [
                {"player_id": 5, "player_name": "X", "jersey_number": 1, "positions": [{"position": "Goalkeeper"}]}
            ],
        }
    ]
    out = raw.parse_roster(lineups)
    assert out[5]["name"] == "X"
    assert out[5]["team"] == 7
    assert out[5]["jersey"] == 1
    assert out[5]["position"] == "Goalkeeper"
