"""Tests for snapshot_to_tracking_frames converter."""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.tracking.schema import TRACKING_CATEGORICAL_DOMAINS, TRACKING_FRAMES_COLUMNS
from tests.tracking._goal_map_helpers import goal_map_for

#: ADR-055 replaced ``home_team_id=1`` at this file's re-keyed call sites. Its frames carry
#: game 1 / period 1 with teams {1, 2} and each keeper at its own end, so this states exactly
#: what ``home_team_id=1`` meant and matches what ``resolve_defended_goals`` derives there.
HOME_GOAL_MAP = goal_map_for({1: 0.0, 2: 105.0})


def test_snapshot_in_source_provider_domain():
    """H1: 'snapshot' must be a valid source_provider value."""
    assert "snapshot" in TRACKING_CATEGORICAL_DOMAINS["source_provider"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def actions_3() -> pd.DataFrame:
    """3-action SPADL fixture.

    action 0 (id=10): has 6 snapshot players (3v3, one GK per side)
    action 1 (id=11): has 0 snapshot players (partial coverage test)
    action 2 (id=12): has 4 snapshot players (2v2, no player_id — synthetic ID test)
    """
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "action_id": [10, 11, 12],
            "period_id": [1, 1, 1],
            "time_seconds": [5.0, 10.0, 15.0],
            "team_id": [100, 200, 100],
            "player_id": [1, 2, 3],
            "start_x": [50.0, 60.0, 70.0],
            "start_y": [34.0, 20.0, 40.0],
            "end_x": [55.0, 65.0, 75.0],
            "end_y": [30.0, 25.0, 35.0],
            "type_id": [0, 0, 0],
            "result_id": [1, 1, 1],
            "bodypart_id": [0, 0, 0],
        }
    )


@pytest.fixture()
def snapshots_3v3() -> pd.DataFrame:
    """6 players for action_id=10 (3v3 with one GK per side)."""
    return pd.DataFrame(
        {
            "action_id": [10, 10, 10, 10, 10, 10],
            "team_id": [100, 100, 100, 200, 200, 200],
            "player_id": [1, 2, 3, 4, 5, 6],
            "is_goalkeeper": [True, False, False, True, False, False],
            "x": [5.0, 40.0, 50.0, 100.0, 60.0, 55.0],
            "y": [34.0, 20.0, 40.0, 34.0, 50.0, 15.0],
        }
    )


@pytest.fixture()
def snapshots_2v2_no_pid() -> pd.DataFrame:
    """4 players for action_id=12 — no player_id column (synthetic ID test)."""
    return pd.DataFrame(
        {
            "action_id": [12, 12, 12, 12],
            "team_id": [100, 100, 200, 200],
            "is_goalkeeper": [True, False, True, False],
            "x": [5.0, 45.0, 100.0, 65.0],
            "y": [34.0, 30.0, 34.0, 40.0],
        }
    )


@pytest.fixture()
def snapshots_combined(snapshots_3v3) -> pd.DataFrame:
    """action 10 has 6 players w/ player_id. Actions 11, 12 have no snapshots."""
    return snapshots_3v3


# ---------------------------------------------------------------------------
# Task 1: Core frames tests
# ---------------------------------------------------------------------------


def test_frames_schema(actions_3, snapshots_combined):
    """Output frames have exactly the 20 TRACKING_FRAMES_COLUMNS."""
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames

    frames, _links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    assert list(frames.columns) == list(TRACKING_FRAMES_COLUMNS.keys())


def test_frames_row_count(actions_3, snapshots_combined):
    """6 player rows + 1 ball row = 7 rows for the single action with data."""
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames

    frames, _links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    assert len(frames) == 7  # 6 players + 1 ball


def test_ball_row(actions_3, snapshots_combined):
    """One ball row per frame, position from action start_x/start_y."""
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames

    frames, _links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    ball = frames[frames["is_ball"]]
    assert len(ball) == 1
    row = ball.iloc[0]
    assert row["x"] == 50.0  # action 10's start_x
    assert row["y"] == 34.0  # action 10's start_y
    assert pd.isna(row["player_id"])
    assert pd.isna(row["team_id"])


def test_frame_metadata(actions_3, snapshots_combined):
    """game_id, period_id, time_seconds derived from actions join."""
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames

    frames, _links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    assert (frames["game_id"] == 1).all()
    assert (frames["period_id"] == 1).all()
    assert (frames["time_seconds"] == 5.0).all()
    assert (frames["frame_id"] == 10).all()  # frame_id = action_id


def test_constant_columns(actions_3, snapshots_combined):
    """Verify NaN/constant columns per spec."""
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames

    frames, _links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    player_rows = frames[~frames["is_ball"]]
    assert player_rows["frame_rate"].isna().all()
    assert player_rows["z"].isna().all()
    assert player_rows["speed"].isna().all()
    # speed_source is NOT NaN (ADR-043): the value is still absent, but its absence is now
    # DECLARED structural rather than left indistinguishable from "not derived yet".
    assert (player_rows["speed_source"] == "unavailable").all()
    assert player_rows["confidence"].isna().all()
    assert player_rows["visibility"].isna().all()
    assert (player_rows["ball_state"] == "alive").all()
    assert (player_rows["team_attacking_direction"] == "ltr").all()
    assert (player_rows["source_provider"] == "snapshot").all()
    assert (player_rows["is_goalkeeper_source"] == "native").all()


# ---------------------------------------------------------------------------
# Task 2: Empty, partial coverage, synthetic player_id, links contract
# ---------------------------------------------------------------------------


def test_empty_snapshots(actions_3):
    """0 snapshots -> empty frames + empty links with correct columns."""
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames

    empty_snap = pd.DataFrame(
        {
            "action_id": pd.Series([], dtype="int64"),
            "team_id": pd.Series([], dtype="int64"),
            "player_id": pd.Series([], dtype="int64"),
            "is_goalkeeper": pd.Series([], dtype="bool"),
            "x": pd.Series([], dtype="float64"),
            "y": pd.Series([], dtype="float64"),
        }
    )
    frames, links = snapshot_to_tracking_frames(empty_snap, actions_3)

    assert len(frames) == 0
    assert list(frames.columns) == list(TRACKING_FRAMES_COLUMNS.keys())
    assert len(links) == 0
    assert list(links.columns) == [
        "action_id",
        "frame_id",
        "time_offset_seconds",
        "n_candidate_frames",
        "link_quality_score",
    ]


def test_partial_coverage(actions_3, snapshots_combined):
    """Actions without snapshot data excluded from both outputs."""
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames

    frames, links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    # Only action_id=10 has snapshot data
    assert set(frames["frame_id"].unique()) == {10}
    assert set(links["action_id"].unique()) == {10}
    # Action 11 and 12 not in outputs
    assert 11 not in links["action_id"].values
    assert 12 not in links["action_id"].values


def test_synthetic_player_id(actions_3, snapshots_2v2_no_pid):
    """When player_id absent, synthetic sequential IDs are generated."""
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames

    frames, _links = snapshot_to_tracking_frames(snapshots_2v2_no_pid, actions_3)
    player_rows = frames[~frames["is_ball"]]
    assert len(player_rows) == 4
    # Synthetic IDs are sequential integers
    pids = player_rows["player_id"].tolist()
    assert pids == [0, 1, 2, 3]


def test_links_contract(actions_3, snapshots_combined):
    """Links have exact-match values per spec."""
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames

    _frames, links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    assert (links["time_offset_seconds"] == 0.0).all()
    assert (links["n_candidate_frames"] == 1).all()
    assert (links["link_quality_score"] == 1.0).all()
    assert (links["frame_id"] == links["action_id"]).all()


# ---------------------------------------------------------------------------
# Task 3: Public import + downstream integration
# ---------------------------------------------------------------------------


def test_public_import():
    """snapshot_to_tracking_frames is importable from silly_kicks.tracking."""
    from silly_kicks.tracking import snapshot_to_tracking_frames  # noqa: F401


def test_downstream_line_break_works(actions_3, snapshots_combined):
    """Downstream works: add_line_break(method='ward') produces valid output."""
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames
    from silly_kicks.tracking.features import add_line_break

    frames, links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    actions_with_data = actions_3[actions_3["action_id"].isin(links["action_id"])]
    result = add_line_break(actions_with_data, frames, links=links, method="ward")
    assert "line_break__ward" in result.columns
    assert "lines_broken__ward" in result.columns
    assert "line_breaking_type__ward" in result.columns
    assert len(result) == len(actions_with_data)
    # Verify Ward actually computed something (not just all-NaN)
    assert result["line_break__ward"].notna().any()


def test_downstream_line_break_REJECTS_home_team_id(actions_3, snapshots_combined):
    """ADR-051 D3 (4.80.0): passing `home_team_id` now raises -- the parameter is GONE.

    This test is the INVERSE of what it asserted before. It used to pin that `home_team_id` was
    REQUIRED (omitting it raised); that guard protected against quietly giving the parameter a
    default, and it is void now the parameter does not exist -- the call it declared impossible
    is the supported one, and the test passed only while the signature it described was live.

    Rather than delete the coverage, it is turned around to pin the Hyrum-visible break itself,
    matching the precedent ADR-055 set for `gk_influence_xfns` / `cover_shadow_xfns`: an existing
    caller passing the old argument gets a loud `TypeError`, never a silently ignored kwarg.
    """
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames
    from silly_kicks.tracking.features import add_line_break

    frames, links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    actions_with_data = actions_3[actions_3["action_id"].isin(links["action_id"])]

    # The supported call: no direction argument at all.
    ok = add_line_break(actions_with_data, frames, links=links, method="ward")
    assert "line_break__ward" in ok.columns

    # DO NOT "FIX" THE ARGUMENT BELOW. Passing the removed `home_team_id` is the POINT: this
    # asserts the break is loud. An automated call-site sweep stripped it once during 4.80.0
    # (leaving a call that cannot raise, so the test passed by asserting nothing) -- a mechanical
    # migration cannot tell a stale call from a deliberate negative one.
    with pytest.raises(TypeError):
        add_line_break(actions_with_data, frames, links=links, method="ward", home_team_id=1)  # type: ignore[call-arg]


def test_downstream_action_context_actor_speed_nan(actions_3, snapshots_combined):
    """actor_speed degrades to NaN; other 3 action_context columns have values.

    add_action_context returns 4 columns total: nearest_defender_distance,
    actor_speed, receiver_zone_density, defenders_in_triangle_to_goal.
    Only actor_speed reads the speed column from the linked frame (NaN on
    snapshots); the other 3 are purely positional.
    """
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames
    from silly_kicks.tracking.features import add_action_context

    frames, links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    actions_with_data = actions_3[actions_3["action_id"].isin(links["action_id"])]
    result = add_action_context(actions_with_data, frames, links=links)
    assert result["actor_speed"].isna().all()
    # Position-only columns should have values (not all NaN)
    assert result["nearest_defender_distance"].notna().any()


def test_downstream_cover_shadows_degrades(actions_3, snapshots_combined):
    """Velocity-dependent add_cover_shadows returns NaN columns, not raises."""
    from unittest.mock import MagicMock

    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames
    from silly_kicks.tracking.features import add_cover_shadows

    frames, links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    actions_with_data = actions_3[actions_3["action_id"].isin(links["action_id"])]
    # xt is a required positional arg (ExpectedThreat). MagicMock suffices
    # because cover_shadows returns None before reaching xt when vx/vy are
    # absent (_cover_shadows.py:792-794).
    mock_xt = MagicMock()
    result = add_cover_shadows(actions_with_data, frames, mock_xt, links=links, goal_map=HOME_GOAL_MAP)
    # Cover shadows requires vx/vy — should degrade to NaN, not raise
    assert "blocking_score" in result.columns
    assert result["blocking_score"].isna().all()


def test_snapshot_frames_mark_velocity_structurally_unavailable(actions_3, snapshots_combined):
    """Every snapshot row declares the marker -- a freeze-frame has no temporal history."""
    from silly_kicks.tracking import SPEED_SOURCE_UNAVAILABLE
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames

    frames, _ = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    assert len(frames) > 0  # non-vacuity: there ARE rows to be marked
    assert (frames["speed_source"] == SPEED_SOURCE_UNAVAILABLE).all()
    # Both facets present -- the marker must not be player-only.
    assert (frames.loc[frames["is_ball"], "speed_source"] == SPEED_SOURCE_UNAVAILABLE).all()
    assert frames["is_ball"].any() and (~frames["is_ball"]).any()
    assert SPEED_SOURCE_UNAVAILABLE in TRACKING_CATEGORICAL_DOMAINS["speed_source"]


def test_downstream_das_degrades(actions_3, snapshots_combined):
    """Velocity-dependent add_das returns NaN columns, not raises.

    ADR-043 non-vacuity: assert the DEGRADE PATH actually executed -- DAS really all-NaN
    AND ``das_source`` really names the structural cause. An all-NaN column alone would
    also be produced by a mapper that found no frames, which is a different bug.
    """
    from silly_kicks.tracking import DAS_SOURCE_UNSCOREABLE_FRAME
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames
    from silly_kicks.tracking.features import add_das

    frames, links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    actions_with_data = actions_3[actions_3["action_id"].isin(links["action_id"])]
    assert len(actions_with_data) > 0  # non-vacuity: there ARE actions to degrade
    # DAS requires vx/vy — the marker makes this an honest degrade, not a raise.
    with pytest.warns(UserWarning, match="unscoreable"):
        result = add_das(actions_with_data, frames, links=links)
    assert "das_team" in result.columns
    assert result["das_team"].isna().all()
    assert result["das_opponent"].isna().all()
    assert result["das_diff"].isna().all()
    assert (result["das_source"] == DAS_SOURCE_UNSCOREABLE_FRAME).all()


def test_unmarked_velocityless_frames_still_raise(actions_3, snapshots_combined):
    """The other direction: strip the marker and the SAME frames must fail LOUD.

    This is what stops the marker from re-absorbing the genuine caller bug ("forgot
    derive_velocities()") that the narrowed catch exists to expose. Same frames, same
    call, only the marker differs -- so a pass here can only come from the marker.
    """
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames
    from silly_kicks.tracking.features import add_das

    frames, links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    unmarked = frames.copy()
    unmarked["speed_source"] = None
    actions_with_data = actions_3[actions_3["action_id"].isin(links["action_id"])]
    with pytest.raises(ValueError, match="velocity columns"):
        add_das(actions_with_data, unmarked, links=links)


def test_partially_marked_frames_still_raise(actions_3, snapshots_combined):
    """A PARTIAL marking is a mixed frame set: the unmarked rows are the caller bug."""
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames
    from silly_kicks.tracking.features import add_das

    frames, links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    partial = frames.copy()
    # `get_loc` returns a slice / boolean mask for a non-unique index, and only a plain
    # positional int is a valid `.iloc` setitem key. Narrowing it here is not ceremony: a
    # duplicated `speed_source` column would otherwise unmark SEVERAL columns and the test
    # would still pass, for the wrong reason.
    speed_source_pos = frames.columns.get_loc("speed_source")
    assert isinstance(speed_source_pos, int), "speed_source is not a single unique column"
    partial.iloc[0, speed_source_pos] = None
    actions_with_data = actions_3[actions_3["action_id"].isin(links["action_id"])]
    with pytest.raises(ValueError, match="velocity columns"):
        add_das(actions_with_data, partial, links=links)


def test_snapshot_actions_are_never_reprojected(actions_3, snapshots_combined):
    """Uniform 'ltr' means acting_team_attacks_rtl resolves BOTH teams to a RESOLVED no-flip.

    Pins the MEANING of the labelling that test_constant_columns pins the VALUE of. A snapshot is
    already in SPADL action-LTR, so the flip mask acting_team_attacks_rtl returns is the input EVERY
    ADR-028 geometry consumer gates its re-projection on -- an all-False (resolved) mask is exactly
    what "never re-projected" means. A future change to per-team directions would flip away-team
    actions; this test fails first, and its mutation leg proves it would.

    Cross-module: this lives in test_snapshot.py for fixture reuse but asserts a property of
    _action_orientation; a future move of acting_team_attacks_rtl touches a test in the snapshot file.

    Accepted limit (spec section 8): this pins the SEAM's contract -- acting_team_attacks_rtl returns
    a resolved no-flip mask for a snapshot -- NOT the guarantee that every consumer keeps routing
    through that seam. The module is the documented SSOT for re-projection, so the seam is the right
    altitude for a doc-hardening cycle.
    """
    from silly_kicks.id_compat import ids_match
    from silly_kicks.tracking._action_orientation import acting_team_attacks_rtl
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames

    away = ids_match(actions_3["team_id"], 200)
    assert away.any()  # premise: the load-bearing away action EXISTS (guards emptiness-vacuity)

    frames, _links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    flip = acting_team_attacks_rtl(actions_3, frames)

    # Property: every action RESOLVES (not <NA>) and never flips -> SB360 is never re-projected.
    assert flip.notna().all()
    assert not flip.any()

    # Non-vacuity: the per-team "fix" this guards against WOULD flip the away action, from a RESOLVED
    # False to a RESOLVED True. Re-assert notna on the MUTATED frame -- nullable-boolean .all() is
    # skipna=True, so without this an <NA> away action would pass .all() silently.
    per_team = frames.copy()
    per_team.loc[ids_match(per_team["team_id"], 200), "team_attacking_direction"] = "rtl"
    flip_mut = acting_team_attacks_rtl(actions_3, per_team)
    assert flip_mut.notna().all()  # still fully resolved post-mutation
    assert flip_mut[away].all()  # away flips to True
    assert not flip_mut[~away].any()  # home unchanged (still False)
