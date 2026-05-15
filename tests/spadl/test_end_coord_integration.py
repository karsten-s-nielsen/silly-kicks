"""Integration tests: _derive_end_coordinates across converters (Bug #7)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig

_IDSSE_DIR = Path(__file__).parent.parent / "datasets" / "idsse"
_SB_DIR = Path(__file__).parent.parent / "datasets" / "statsbomb"


# -----------------------------------------------------------------------
# Sportec / IDSSE: single-position provider — pass-class gets end != start
# -----------------------------------------------------------------------


class TestSportecEndCoordinates:
    @pytest.fixture()
    def actions(self) -> pd.DataFrame:
        from silly_kicks.spadl.sportec import convert_to_actions

        events = pd.read_parquet(_IDSSE_DIR / "per_period_match.parquet")
        gk_ids: set[str] | None = None
        if "play_goal_keeper_action" in events.columns:
            gk_ids = set(
                events.loc[events["play_goal_keeper_action"].notna(), "player_id"].dropna().astype(str).tolist()
            )
        actions, _report = convert_to_actions(
            events,
            home_team_id="home",
            home_team_start_left=False,  # home attacks LEFT in P1
            goalkeeper_ids=gk_ids,
        )
        return actions

    def test_pass_class_majority_end_neq_start(self, actions: pd.DataFrame):
        pass_type_ids = {
            spadlconfig.actiontype_id[t]
            for t in (
                "pass",
                "cross",
                "throw_in",
                "freekick_crossed",
                "freekick_short",
                "corner_crossed",
                "corner_short",
                "goalkick",
            )
        }
        pass_actions = actions[actions["type_id"].isin(pass_type_ids)]
        has_different_end = (pass_actions["end_x"] != pass_actions["start_x"]) | (
            pass_actions["end_y"] != pass_actions["start_y"]
        )
        # Majority should have end != start; only period-boundary last
        # actions may still have end == start.
        ratio = has_different_end.mean()
        assert ratio > 0.90, f"Only {ratio:.1%} of pass-class actions have end != start"

    def test_shots_keep_end_equals_start(self, actions: pd.DataFrame):
        shot_type_ids = {spadlconfig.actiontype_id[t] for t in ("shot", "shot_penalty", "shot_freekick")}
        shots = actions[actions["type_id"].isin(shot_type_ids)]
        if len(shots) == 0:
            pytest.skip("No shots in fixture")
        same_end = (shots["end_x"] == shots["start_x"]) & (shots["end_y"] == shots["start_y"])
        assert same_end.all(), "Shot end coordinates should equal start"

    def test_dribble_count_decreases(self, actions: pd.DataFrame):
        dribble_id = spadlconfig.actiontype_id["dribble"]
        n_dribbles = (actions["type_id"] == dribble_id).sum()
        # Pre-fix: 708 dribbles (639 spurious). Post-fix: ~69 legitimate.
        # Use generous upper bound to allow for minor variation.
        assert n_dribbles < 200, f"Expected < 200 dribbles after fix, got {n_dribbles}"
        assert n_dribbles > 0, "Should still have some legitimate dribbles"


# -----------------------------------------------------------------------
# StatsBomb: source-provided end coordinates preserved (regression guard)
# -----------------------------------------------------------------------


class TestStatsBombEndCoordinatesPreserved:
    @pytest.fixture()
    def actions(self) -> pd.DataFrame:
        from silly_kicks.spadl.statsbomb import convert_to_actions

        raw_path = _SB_DIR / "raw" / "events" / "7584.json"
        with open(raw_path, encoding="utf-8") as f:
            raw = json.load(f)
        # Adapt raw StatsBomb events to DataFrame format expected by converter.
        _top_level_keys = {
            "id",
            "index",
            "period",
            "timestamp",
            "minute",
            "second",
            "type",
            "possession",
            "possession_team",
            "play_pattern",
            "team",
            "player",
            "position",
            "location",
            "duration",
            "under_pressure",
            "off_camera",
            "out",
            "related_events",
            "tactics",
        }
        events = pd.DataFrame(
            [
                {
                    "event_id": e["id"],
                    "game_id": 7584,
                    "period_id": e["period"],
                    "timestamp": e["timestamp"],
                    "minute": e["minute"],
                    "second": e["second"],
                    "type_id": e["type"]["id"],
                    "type_name": e["type"]["name"],
                    "possession": e.get("possession"),
                    "possession_team_id": e.get("possession_team", {}).get("id"),
                    "possession_team_name": e.get("possession_team", {}).get("name"),
                    "play_pattern_id": e.get("play_pattern", {}).get("id"),
                    "play_pattern_name": e.get("play_pattern", {}).get("name"),
                    "team_id": e.get("team", {}).get("id"),
                    "team_name": e.get("team", {}).get("name"),
                    "player_id": e.get("player", {}).get("id"),
                    "player_name": e.get("player", {}).get("name"),
                    "position_id": e.get("position", {}).get("id"),
                    "position_name": e.get("position", {}).get("name"),
                    "location": e.get("location"),
                    "duration": e.get("duration"),
                    "under_pressure": e.get("under_pressure"),
                    "extra": {k: v for k, v in e.items() if k not in _top_level_keys},
                }
                for e in raw
            ]
        )
        home_team_id = int(events["team_id"].dropna().iloc[0])
        actions, _ = convert_to_actions(
            events,
            home_team_id=home_team_id,
            xy_fidelity_version=1,
            shot_fidelity_version=1,
        )
        return actions

    def test_passes_have_source_end_coordinates(self, actions: pd.DataFrame):
        pass_id = spadlconfig.actiontype_id["pass"]
        passes = actions[actions["type_id"] == pass_id]
        has_different_end = (passes["end_x"] != passes["start_x"]) | (passes["end_y"] != passes["start_y"])
        # StatsBomb provides explicit pass.end_location for virtually all passes.
        ratio = has_different_end.mean()
        assert ratio > 0.95, f"StatsBomb passes should have source end coords; only {ratio:.1%} do"

    def test_clearances_with_source_end_preserved(self, actions: pd.DataFrame):
        clearance_id = spadlconfig.actiontype_id["clearance"]
        clearances = actions[actions["type_id"] == clearance_id]
        if len(clearances) == 0:
            pytest.skip("No clearances in fixture")
        has_different_end = (clearances["end_x"] != clearances["start_x"]) | (
            clearances["end_y"] != clearances["start_y"]
        )
        # StatsBomb provides end coords for ~99.6% of clearances.
        # Guard ensures these are NOT overwritten (unlike old _fix_clearances).
        ratio = has_different_end.mean()
        assert ratio > 0.90, f"StatsBomb clearances should keep source end coords; only {ratio:.1%} do"


# -----------------------------------------------------------------------
# Gradient Sports: shots / tackles / keeper_saves must keep end == start
# -----------------------------------------------------------------------

_GS_DIR = Path(__file__).parent.parent / "datasets" / "gradientsports"


class TestGradientSportsExcludedTypesKeepEnd:
    @pytest.fixture()
    def actions(self) -> pd.DataFrame:
        from silly_kicks.spadl.gradientsports import convert_to_actions
        from tests.spadl.test_gradientsports import _load_synthetic_events

        events = _load_synthetic_events()
        actions, _ = convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        return actions

    def test_shots_tackles_keeper_saves_end_equals_start(
        self,
        actions: pd.DataFrame,
    ):
        """Shots, tackles, keeper_saves should NOT get next-event end coords."""
        excluded_type_ids = {
            spadlconfig.actiontype_id[t] for t in ("shot", "shot_penalty", "shot_freekick", "tackle", "keeper_save")
        }
        excluded = actions[actions["type_id"].isin(excluded_type_ids)]
        if len(excluded) == 0:
            pytest.skip("No shot/tackle/keeper_save in GS fixture")
        same_end = (excluded["end_x"] == excluded["start_x"]) & (excluded["end_y"] == excluded["start_y"])
        assert same_end.all(), (
            f"GS shots/tackles/keeper_saves should keep end==start; "
            f"{(~same_end).sum()}/{len(excluded)} have different end coords"
        )
