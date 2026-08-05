"""Fixture preconditions. Each is load-bearing for a verdict, so each asserts rather than hopes.

Every expected value here was MEASURED against the real library, not predicted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.sb360 import _fixture as F


def _link_map(links: pd.DataFrame) -> dict[int, int]:
    """action_id -> frame_id, as plain ints.

    A dict rather than repeated ``links.loc[mask, col].iloc[0]``: the DataFrame idiom is both
    O(n) per lookup and untypeable (pandas narrows the result to a Hashable union).
    """
    return {int(a): int(f) for a, f in zip(links["action_id"], links["frame_id"], strict=True)}


def _players_at(frames: pd.DataFrame, frame_id: int) -> pd.DataFrame:
    sel = frames[(frames["frame_id"] == frame_id) & (~frames["is_ball"].astype(bool))]
    return sel.sort_values("player_id").reset_index(drop=True)


def test_leg_a_is_built_by_the_real_producer():
    """Leg A must come from snapshot_to_tracking_frames, so the fixture cannot drift from it
    and so the audit exercises the code path real SB360 data will hit."""
    _, frames, _ = F.build_leg_a()
    assert (frames["source_provider"] == "snapshot").all()
    assert (frames["speed_source"] == "unavailable").all()
    assert "vx" not in frames.columns and "vy" not in frames.columns
    assert int(frames["is_ball"].sum()) == len(F._ACTIONS), "one ball row per synthesised frame"


def test_window_discovery_actually_finds_the_known_window():
    """Non-vacuity for the scan itself: a silent fallback is a hardcoded value dressed as a
    measurement."""
    found = F.discovered_windows()
    assert "add_actor_pre_window.pre_seconds" in found, (
        f"scan missed the known window (features.py:864). Found: {sorted(found)}"
    )
    # The allowlist must stay an allowlist: tau_seconds is an influence-DECAY constant, not a
    # frame window, and a substring heuristic admits it.
    assert not [k for k in found if k.endswith(".tau_seconds")], (
        f"scan admitted a decay constant: {sorted(k for k in found if 'tau' in k)}"
    )
    assert F.required_neighbourhood_seconds() >= 2.0 * max(found.values())


def test_positions_match_exactly_at_linked_frames():
    """The velocity axis holds ROSTER and POSITION fixed; only kinematics vary. Otherwise a
    verdict confounds position with velocity and is unattributable.

    Exact, not approximate: Leg B's times are built as ``t0 + k*step`` over integer k so the
    anchor lands on t0 with no float drift. Measured worst delta 0.0.
    """
    _, frames_a, links_a = F.build_leg_a()
    _, frames_b, links_b = F.build_leg_b()
    map_a, map_b = _link_map(links_a), _link_map(links_b)

    for action_id in map_a:
        a = _players_at(frames_a, map_a[action_id])
        b = _players_at(frames_b, map_b[action_id])
        assert list(a["player_id"]) == list(b["player_id"]), f"roster differs at action {action_id}"
        np.testing.assert_array_equal(a["x"].to_numpy(), b["x"].to_numpy())
        np.testing.assert_array_equal(a["y"].to_numpy(), b["y"].to_numpy())


def test_ball_matches_exactly_at_the_anchor_and_moves_thereafter():
    """The ball is the anchor for every ball-relative feature, and Leg B's ball MOVES.

    Both halves are load-bearing. A static ball measured as breaking two aggregators for lack
    of a kick to detect (`add_elastic_sync` all-NaN on Leg B, `add_shot_goalmouth` with no
    trajectory to fit). But the moment the ball moves, its equality with Leg A at the anchor
    stops being trivially true -- and the players-only position test cannot see it.
    """
    _, frames_a, links_a = F.build_leg_a()
    _, frames_b, links_b = F.build_leg_b()
    map_a, map_b = _link_map(links_a), _link_map(links_b)

    for action_id in map_a:
        ba = frames_a[(frames_a["frame_id"] == map_a[action_id]) & frames_a["is_ball"].astype(bool)]
        bb = frames_b[(frames_b["frame_id"] == map_b[action_id]) & frames_b["is_ball"].astype(bool)]
        assert len(ba) == 1 and len(bb) == 1
        assert float(ba["x"].to_numpy()[0]) == float(bb["x"].to_numpy()[0])
        assert float(ba["y"].to_numpy()[0]) == float(bb["y"].to_numpy()[0])

    balls_b = frames_b[frames_b["is_ball"].astype(bool)]
    assert float(balls_b["x"].max() - balls_b["x"].min()) > 10.0, "Leg B ball never moves"
    assert float(balls_b["z"].max()) > 0.0, "Leg B ball has no height profile"


def test_both_legs_carry_team_in_possession():
    """A caller-side enrichment, applied identically to both legs. Without it `add_das`
    short-circuits on Leg A but raises on Leg B -- an unfair comparison, not a finding."""
    _, frames_a, _ = F.build_leg_a()
    _, frames_b, _ = F.build_leg_b()
    for name, f in (("leg A", frames_a), ("leg B", frames_b)):
        assert "team_in_possession" in f.columns, f"{name} lacks team_in_possession"
        assert f["team_in_possession"].notna().all(), f"{name} has null team_in_possession"


def test_every_action_links_in_leg_b_at_zero_offset():
    _, _, links_b = F.build_leg_b()
    assert int(links_b["frame_id"].isna().sum()) == 0
    assert (links_b["time_offset_seconds"] == 0.0).all(), (
        "an anchor frame that is not exactly at the action time makes the position invariant approximate for no reason"
    )


def test_leg_b_neighbourhood_covers_the_longest_enumerated_window():
    """If Leg B is shorter than a feature's window, that feature is NaN in BOTH legs ->
    no_signal -> not_exercised, and its structurally_impossible annotation becomes
    INADMISSIBLE. The distinction is then silently lost."""
    actions, frames_b, links_b = F.build_leg_b()
    required = F.required_neighbourhood_seconds()
    times = {int(a): float(t) for a, t in zip(actions["action_id"], actions["time_seconds"], strict=True)}

    for action_id in _link_map(links_b):
        t0 = times[action_id]
        # Neighbourhood is per-ACTION: frames for other actions sit at different times, so the
        # window is measured against this action's own contiguous block.
        block = frames_b[(frames_b["time_seconds"] >= t0 - required) & (frames_b["time_seconds"] <= t0 + required)]
        lo = float(block["time_seconds"].min())
        hi = float(block["time_seconds"].max())
        assert lo <= t0 - required + 1e-9, (
            f"action {action_id}: neighbourhood reaches only {t0 - lo}s before t0, need {required}s"
        )
        assert hi >= t0 + required - 1e-9, (
            f"action {action_id}: neighbourhood reaches only {hi - t0}s after t0, need {required}s"
        )


def test_leg_b_motion_is_non_degenerate():
    """Constant velocity makes every acceleration-dependent quantity identically zero in BOTH
    legs, which reads as a false `works`. Measured: speed sigma 0.26, heading sigma 1.74."""
    _, frames_b, _ = F.build_leg_b()
    players = frames_b[~frames_b["is_ball"].astype(bool)]
    first_pid = players["player_id"].to_numpy()[0]
    one = players[players["player_id"] == first_pid].sort_values("time_seconds")
    speeds = one["speed"].to_numpy(dtype="float64")
    headings = np.arctan2(one["vy"].to_numpy(dtype="float64"), one["vx"].to_numpy(dtype="float64"))
    assert float(np.nanstd(speeds)) > 1e-3, f"speed is constant (std={np.nanstd(speeds)}) -- degenerate"
    assert float(np.nanstd(headings)) > 1e-3, f"heading is constant (std={np.nanstd(headings)}) -- degenerate"


def test_velocity_is_consistent_with_position():
    """A positions-only fixture silently produces velocity that contradicts its own positions
    (ADR-045 D1). `_velocity` is the analytic derivative of `_offset`; verify numerically."""
    dt = 1e-6
    for aid in (0, 3):
        for idx in (0, 7):
            t = 300.0
            x0, y0 = F._offset(aid, idx, t - dt)
            x1, y1 = F._offset(aid, idx, t + dt)
            vx, vy = F._velocity(aid, idx, t)
            np.testing.assert_allclose((x1 - x0) / (2 * dt), vx, rtol=1e-6)
            np.testing.assert_allclose((y1 - y0) / (2 * dt), vy, rtol=1e-6)


@pytest.mark.parametrize("id_dtype", ["int64", "Int64", "object"])
def test_id_dtype_parameterization_reaches_the_producer(id_dtype):
    """A hand-built fixture silently picks one id dtype and can mask a real ADR-019 defect.

    The frame dtype is NOT the requested one for `int64`: a frame set carries a NaN-id ball
    row and numpy int64 cannot hold NaN, so the producer's own concat upcasts to float64.
    Measured, not assumed -- and Leg B must land on the same dtype or the legs disagree about
    identity before any value is compared.
    """
    actions_a, frames_a, _ = F.build_leg_a(id_dtype=id_dtype)
    actions_b, frames_b, _ = F.build_leg_b(id_dtype=id_dtype)

    assert str(actions_a["player_id"].dtype) == id_dtype
    assert actions_a["player_id"].dtype == actions_b["player_id"].dtype
    assert str(frames_a["player_id"].dtype) == F.frame_id_dtype(id_dtype)
    assert frames_a["player_id"].dtype == frames_b["player_id"].dtype
    assert frames_a["team_id"].dtype == frames_b["team_id"].dtype


@pytest.mark.parametrize("roster", ["full", "gk_absent", "defender_absent"])
def test_roster_variants_hold_velocity_fixed(roster):
    """The visibility axis varies roster at FIXED velocity -- the mirror of the velocity axis."""
    _, frames_a, _ = F.build_leg_a(roster=roster)
    players = frames_a[~frames_a["is_ball"].astype(bool)]
    keepers = int(players["is_goalkeeper"].sum())
    if roster == "gk_absent":
        assert keepers == 0, "gk_absent variant still contains a keeper"
    else:
        assert keepers == 2 * len(F._ACTIONS), "both keepers, every frame"
        if roster == "defender_absent":
            assert 24 not in {int(p) for p in players["player_id"].dropna()}
    assert (frames_a["speed_source"] == "unavailable").all()


@pytest.mark.parametrize("roster", ["gk_absent", "defender_absent"])
def test_roster_variants_differ_from_the_control(roster):
    """Non-vacuity for the visibility axis: a variant that removes nobody makes every
    visibility verdict vacuous."""
    _, full, _ = F.build_leg_a(roster="full")
    _, variant, _ = F.build_leg_a(roster=roster)
    assert len(variant) < len(full), f"{roster} removed no rows"


def test_visible_area_side_table_is_keyed_by_action_and_never_reaches_the_producer():
    """The polygon rides a side table INSIDE the harness. The snapshots contract is not
    extended and snapshot_to_tracking_frames is not modified."""
    side = F.visible_area_side_table()
    assert set(side.columns) == {"action_id", "polygon"}
    assert sorted(side["action_id"]) == sorted(a[0] for a in F._ACTIONS)
    _, frames, _ = F.build_leg_a()
    assert "polygon" not in frames.columns and "visible_area" not in frames.columns
