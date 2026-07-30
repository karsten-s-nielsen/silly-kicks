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
from silly_kicks.tracking.direction import compute_attacking_direction

_IDSSE_DIR = Path(__file__).parent.parent / "datasets" / "idsse"
_PAIRED_TRACKING = _IDSSE_DIR / "paired_tracking.parquet"

#: Match metadata shared by the events + tracking halves of the paired fixture.
#: ``home_team_start_left=False`` == "home does NOT start on the left", i.e. home
#: DEFENDS the high-x goal in P1 and therefore ATTACKS toward x=0 in P1 (see
#: ``tests/datasets/idsse/README.md``: "Home attacks LEFT in P1").
_HOME_TEAM_ID = "home"
_HOME_TEAM_START_LEFT = False


def _load_tracking_frames(*, game_id: str) -> pd.DataFrame:
    """Load paired tracking fixture and reshape for defending_gk_from_frames.

    This is a PARTIAL reshape — only the columns accessed by
    defending_gk_from_frames / link_actions_to_frames / ADR-028 orientation
    are mapped. The fixture is NOT fully SPORTEC_TRACKING_FRAMES_COLUMNS
    compliant (missing z, ball_state, confidence, visibility,
    is_goalkeeper_source, speed_source).

    ``team_attacking_direction`` (ADR-028) IS populated, because without it
    ``acting_team_attacks_rtl`` silently returns an all-False flip and every
    frame-sampled position this fixture feeds ``add_pre_shot_gk_context``
    lands in the WRONG coordinate convention for half the actions.

    These frames are PER-PERIOD ABSOLUTE (raw DFL orientation carried through
    the lakehouse ``fct_tracking_frames`` mart), NOT the canonical
    home-attacks-right convention ``convert_to_frames`` emits — so the honest
    label is period-dependent, not a blanket ``home="ltr"``. Measured on the
    fixture itself:

    * P1 — away GK median x = 1.75 (lowest of all 22 players), home GK median
      x = 96.28 (highest). Home defends the high-x goal, so home attacks
      RIGHT-TO-LEFT: home ``"rtl"``, away ``"ltr"``.
    * P2 — teams swap: home GK median x = 39.32 (lowest), away GK median
      x = 118.60 (highest). Home ``"ltr"``, away ``"rtl"``.

    Corroborated by event/ball co-location: the P1 home shot at action-LTR
    (90.87, 26.08) sits at tracking-ball (-0.7, 44.3) — i.e. at
    (105 - 90.87, 68 - 26.08) = (14.1, 41.9) modulo the ball crossing the
    goal line — and the P1 home pass origin (93.28, 3.61) sits at
    tracking-ball (13.4, 62.1) vs the reflection (11.7, 64.4). The P1 AWAY
    pass, by contrast, matches its UNreflected action-LTR end (48.69, 52.86)
    at tracking-ball (54.2, 47.7). In P2 every home action matches unreflected
    (e.g. shot at LTR x=79.12 -> ball x=89.9), never the reflection.

    That is exactly ``compute_attacking_direction`` for the UNFLIPPED raw
    input, which is what this helper calls rather than hard-coding a table.
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
    # game_id is taken from the ACTIONS, not from the tracking fixture's own
    # match_id: the tracking half carries "J03WMX" while the events converter
    # emits "idsse_J03WMX". acting_team_attacks_rtl joins on
    # (game_id, period_id, team_id) whenever BOTH sides carry game_id, so the
    # mismatched value would make every direction lookup miss -> an all-False
    # flip that is NOT covered by OrientationUnresolvedWarning (the warning
    # fires on the pre-merge conditions only). link_actions_to_frames never
    # reads game_id, so this is inert for the linkage this fixture exercises.
    frames["game_id"] = str(game_id)
    frames["period_id"] = frames["period_id"].astype("int64")
    frames["frame_id"] = frames["frame_id"].astype("int64")
    frames["time_seconds"] = frames["time_seconds"].astype("float64")
    frames["player_id"] = frames["player_id"].astype(str)
    frames["team_id"] = frames["team_id"].astype(str)
    frames["is_goalkeeper"] = frames["is_goalkeeper"].astype(bool)
    # ADR-028 orientation ground truth. Ball rows get None (the convention
    # convert_to_frames produces); this fixture has no ball rows — the ball is
    # carried as ball_x/ball_y columns on the player rows.
    frames["team_attacking_direction"] = compute_attacking_direction(
        team_id=frames["team_id"],
        period_id=frames["period_id"],
        is_ball=frames["is_ball"],
        home_team_id=_HOME_TEAM_ID,
        home_team_start_left=_HOME_TEAM_START_LEFT,
    )
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
        home_team_id=_HOME_TEAM_ID,
        home_team_start_left=_HOME_TEAM_START_LEFT,  # home attacks LEFT in P1
        goalkeeper_ids=gk_ids,
    )
    game_ids = actions["game_id"].dropna().unique()
    assert len(game_ids) == 1, f"paired fixture must be a single match, got {game_ids!r}"
    frames = _load_tracking_frames(game_id=str(game_ids[0]))
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
