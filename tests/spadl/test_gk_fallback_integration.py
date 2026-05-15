"""Integration test: GK fallback via defending_gk_from_frames (Bug #2).

Uses paired IDSSE events (per_period_match.parquet) + tracking
(paired_tracking.parquet) to verify that shots get defending_gk_player_id
populated from tracking frames when events-based lookback finds no
keeper_save.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.sportec import convert_to_actions
from silly_kicks.spadl.utils import add_pre_shot_gk_context

_IDSSE_DIR = Path(__file__).parent.parent / "datasets" / "idsse"
_PAIRED_TRACKING = _IDSSE_DIR / "paired_tracking.parquet"


def _load_tracking_frames() -> pd.DataFrame:
    """Load paired tracking fixture and reshape for defending_gk_from_frames.

    This is a PARTIAL reshape — only the columns accessed by
    defending_gk_from_frames / link_actions_to_frames are mapped.
    The fixture is NOT fully SPORTEC_TRACKING_FRAMES_COLUMNS compliant
    (missing z, ball_state, team_attacking_direction, confidence,
    visibility, is_goalkeeper_source, speed_source).
    """
    raw = pd.read_parquet(_PAIRED_TRACKING)
    # Lakehouse column names -> silly-kicks tracking schema.
    # NB: fixture has both "team" ("home"/"away") and "team_id"
    # (DFL CLU IDs like "DFL-CLU-00000G").  Sportec convert_to_actions
    # produces actions with team_id = "home"/"away", so we must use the
    # "team" column as team_id to match.  Drop the DFL CLU column first
    # to avoid duplicate team_id after rename.
    raw = raw.drop(columns=["team_id"])
    rename = {
        "match_id": "game_id",
        "period": "period_id",
        "frame": "frame_id",
        "timestamp_seconds": "time_seconds",
        "team": "team_id",  # "home"/"away" — matches action team_id
        "x": "x",
        "y": "y",
        "ball_x": "ball_x",
        "ball_y": "ball_y",
        "speed_ms": "speed",
    }
    frames = raw.rename(columns=rename)
    # Add missing columns with defaults.
    if "vx" not in frames.columns:
        frames["vx"] = np.nan
    if "vy" not in frames.columns:
        frames["vy"] = np.nan
    if "ax" not in frames.columns:
        frames["ax"] = np.nan
    if "ay" not in frames.columns:
        frames["ay"] = np.nan
    if "is_ball" not in frames.columns:
        frames["is_ball"] = False
    # Ensure dtypes match schema expectations.
    frames["game_id"] = frames["game_id"].astype(str)
    frames["period_id"] = frames["period_id"].astype("int64")
    frames["frame_id"] = frames["frame_id"].astype("int64")
    frames["time_seconds"] = frames["time_seconds"].astype("float64")
    frames["player_id"] = frames["player_id"].astype(str)
    frames["team_id"] = frames["team_id"].astype(str)
    frames["is_goalkeeper"] = frames["is_goalkeeper"].astype(bool)
    return frames


@pytest.fixture()
def paired_data():
    """Load paired events + tracking for match J03WMX."""
    if not _PAIRED_TRACKING.exists():
        pytest.skip("Paired tracking fixture not available")

    events = pd.read_parquet(_IDSSE_DIR / "per_period_match.parquet")
    gk_ids: set[str] | None = None
    if "play_goal_keeper_action" in events.columns:
        gk_ids = set(events.loc[events["play_goal_keeper_action"].notna(), "player_id"].dropna().astype(str).tolist())
    actions, _report = convert_to_actions(
        events,
        home_team_id="home",
        home_team_start_left=False,  # home attacks LEFT in P1
        goalkeeper_ids=gk_ids,
    )
    frames = _load_tracking_frames()
    return actions, frames


class TestGkFallbackPopulatesDefendingGk:
    """Bug #2: add_pre_shot_gk_context fills NaN GK IDs from tracking."""

    def test_shots_within_tracking_window_have_gk(self, paired_data):
        actions, frames = paired_data
        enriched = add_pre_shot_gk_context(actions, frames=frames)

        shot_type_ids = {spadlconfig.actiontype_id[t] for t in ("shot", "shot_penalty", "shot_freekick")}
        shots = enriched[enriched["type_id"].isin(shot_type_ids)]

        # Tracking covers P1 ts=90-107 and P2 ts=624-640.
        # Find shots within those windows.
        p1_shots = shots[(shots["period_id"] == 1) & (shots["time_seconds"] >= 90.0) & (shots["time_seconds"] <= 107.0)]
        p2_shots = shots[
            (shots["period_id"] == 2) & (shots["time_seconds"] >= 624.0) & (shots["time_seconds"] <= 640.0)
        ]
        covered_shots = pd.concat([p1_shots, p2_shots])
        assert len(covered_shots) >= 2, f"Expected >= 2 shots in tracking windows, got {len(covered_shots)}"

        has_gk = covered_shots["defending_gk_player_id"].notna()
        assert has_gk.all(), (
            f"Shots in tracking window missing defending_gk_player_id: "
            f"{covered_shots[~has_gk][['period_id', 'time_seconds']].to_dict()}"
        )

    def test_resolved_gk_is_opposing_team(self, paired_data):
        actions, frames = paired_data
        enriched = add_pre_shot_gk_context(actions, frames=frames)

        shot_type_ids = {spadlconfig.actiontype_id[t] for t in ("shot", "shot_penalty", "shot_freekick")}
        shots = enriched[enriched["type_id"].isin(shot_type_ids)]

        # Known GK IDs from the tracking fixture.
        home_gk = "DFL-OBJ-0002HE"
        away_gk = "DFL-OBJ-0002DR"

        for _, shot in shots.iterrows():
            gk_id = shot["defending_gk_player_id"]
            if pd.isna(gk_id):
                continue  # Shot outside tracking window
            shooter_team = shot["team_id"]
            # Home shooter -> away GK defends; away shooter -> home GK defends.
            if shooter_team == "home":
                assert str(gk_id) == away_gk, (
                    f"Home shot at t={shot['time_seconds']:.1f}: expected away GK {away_gk}, got {gk_id}"
                )
            else:
                assert str(gk_id) == home_gk, (
                    f"Away shot at t={shot['time_seconds']:.1f}: expected home GK {home_gk}, got {gk_id}"
                )
