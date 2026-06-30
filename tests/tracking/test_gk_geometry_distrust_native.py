"""SkillCorner keeper-origin resolution (broadcast distrust) -- CR 2026-06-30 / 4.37.0.

Distrust scope is GOAL-KICKS ONLY (narrowed after real-data validation: open-play GK passes' native
origin IS the keeper, validated 0.4 m). Covers the provider-trust allowlist, the detection-aware GK
helper WITH ADR-028 home/away re-projection, the goal-kick ladder (home + away every tier), the
open-play-keeps-native contract, the S4 guard, and the regression/destination-invariant gates.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._gk_geometry import (
    _tracking_gk_xy_detected,
    flag_native_goalkick_out_of_region,
    native_origin_is_trusted,
    resolve_gk_geometry,
)

_GOALKICK = 22  # spadlconfig.actiontype_id["goalkick"]
_PASS = 0


# --------------------------------------------------------------------------------------
# Provider-trust allowlist (fail-safe)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "provider,trusted",
    [
        ("gradientsports", True),
        ("idsse", True),
        ("metrica", True),
        ("sportec", True),
        ("statsbomb", True),
        ("wyscout", True),
        ("GradientSports", True),  # case-insensitive
        ("skillcorner", False),  # broadcast -> distrust
        ("SkillCorner", False),
        (None, False),  # fail-safe: unknown -> distrust
        ("some_future_broadcast", False),  # fail-safe default
    ],
)
def test_native_origin_trust_allowlist(provider, trusted):
    assert native_origin_is_trusted(provider) is trusted


# --------------------------------------------------------------------------------------
# Fixtures: a GK (team "1") + ball per frame, with a per-team attacking direction.
# direction="ltr" => home (no ADR-028 flip); "rtl" => away (flip 105-x, 68-y).
# --------------------------------------------------------------------------------------
def _frames_with_gk(detections, direction="ltr"):
    """detections: list of (frame_id, time_seconds, gk_x, gk_y, visible). gk_x/gk_y are in the FRAME
    (home-attacks-right) convention."""
    rows = []
    for fid, t, gx, gy, vis in detections:
        rows.append(
            dict(
                game_id="g",
                period_id=1,
                frame_id=fid,
                time_seconds=t,
                player_id="gk1",
                team_id="1",
                is_goalkeeper=True,
                is_ball=False,
                x=gx,
                y=gy,
                visibility=vis,
                frame_rate=10.0,
                source_provider="skillcorner",
                team_attacking_direction=direction,
            )
        )
        rows.append(
            dict(
                game_id="g",
                period_id=1,
                frame_id=fid,
                time_seconds=t,
                player_id=None,
                team_id=None,
                is_goalkeeper=False,
                is_ball=True,
                x=50.0,
                y=34.0,
                visibility=None,
                frame_rate=10.0,
                source_provider="skillcorner",
                team_attacking_direction=None,
            )
        )
    return pd.DataFrame(rows)


def _action(frame_t, type_id=_PASS, start_x=25.0, start_y=40.0):
    return pd.DataFrame(
        [
            dict(
                game_id="g",
                period_id=1,
                action_id=0,
                team_id="1",
                player_id="gk1",
                type_id=type_id,
                start_x=start_x,
                start_y=start_y,
                end_x=45.0,
                end_y=30.0,
                time_seconds=frame_t,
            )
        ]
    )


# --------------------------------------------------------------------------------------
# Detection-aware helper -- home (no flip) AND away (ADR-028 re-projection)
# --------------------------------------------------------------------------------------
def test_detected_helper_picks_nearest_visible_in_window():
    frames = _frames_with_gk([(98, 9.8, 99.0, 1.0, False), (99, 9.9, 5.0, 33.0, True), (105, 10.5, 8.0, 33.0, True)])
    xy = _tracking_gk_xy_detected(_action(10.0), frames, links=None, window_s=1.0, clamp_goal_area=False)
    assert np.allclose(xy[0], [5.0, 33.0])


def test_detected_helper_ties_break_at_or_before():
    frames = _frames_with_gk([(95, 9.5, 4.0, 30.0, True), (105, 10.5, 9.0, 30.0, True)])
    xy = _tracking_gk_xy_detected(_action(10.0), frames, links=None, window_s=1.0, clamp_goal_area=False)
    assert np.allclose(xy[0], [4.0, 30.0])


def test_detected_helper_away_reprojects_to_action_ltr():
    # away GK detected near its own goal at FRAME x=100 (home-LTR); direction rtl => action-LTR 105-100=5.
    frames = _frames_with_gk([(100, 10.0, 100.0, 34.0, True)], direction="rtl")
    xy = _tracking_gk_xy_detected(_action(10.0), frames, links=None, window_s=1.0, clamp_goal_area=True)
    assert np.allclose(xy[0], [5.0, 34.0])  # re-projected + within the box clamp


def test_detected_helper_home_and_away_same_physical_position_match():
    # The SAME physical keeper position must yield the SAME action-LTR result for home and away.
    home = _tracking_gk_xy_detected(
        _action(10.0),
        _frames_with_gk([(100, 10.0, 5.5, 34.0, True)], "ltr"),
        links=None,
        window_s=1.0,
        clamp_goal_area=True,
    )
    away = _tracking_gk_xy_detected(
        _action(10.0),
        _frames_with_gk([(100, 10.0, 99.5, 34.0, True)], "rtl"),
        links=None,
        window_s=1.0,
        clamp_goal_area=True,
    )
    assert np.allclose(home[0], away[0])  # both -> (5.5, 34) action-LTR


def test_detected_helper_clamp_drops_off_position_keeper_home_and_away():
    home = _tracking_gk_xy_detected(
        _action(10.0),
        _frames_with_gk([(100, 10.0, 20.0, 34.0, True)], "ltr"),
        links=None,
        window_s=1.0,
        clamp_goal_area=True,
    )
    away = _tracking_gk_xy_detected(  # FRAME x=85 -> action-LTR 20 -> > 16.5 -> rejected
        _action(10.0),
        _frames_with_gk([(100, 10.0, 85.0, 34.0, True)], "rtl"),
        links=None,
        window_s=1.0,
        clamp_goal_area=True,
    )
    assert np.isnan(home[0, 0]) and np.isnan(away[0, 0])


def test_detected_helper_no_visible_detection_returns_nan():
    frames = _frames_with_gk([(100, 10.0, 5.0, 34.0, False)])
    xy = _tracking_gk_xy_detected(_action(10.0), frames, links=None, window_s=1.0, clamp_goal_area=False)
    assert np.isnan(xy[0, 0])


# --------------------------------------------------------------------------------------
# Goal-kick distrust ladder -- home AND away, every tier
# --------------------------------------------------------------------------------------
def _acts(type_id, start_x=25.0):
    return _action(10.0, type_id=type_id, start_x=start_x)


@pytest.mark.parametrize("direction,frame_x", [("ltr", 5.5), ("rtl", 99.5)])
def test_distrust_goalkick_detected_uses_tracked_in_box(direction, frame_x):
    frames = _frames_with_gk([(100, 10.0, frame_x, 34.0, True)], direction=direction)
    out = resolve_gk_geometry(_acts(_GOALKICK), frames=frames, distrust_native_origin=True)
    assert out["origin_source"].iloc[0] == "tracking_gk"
    assert np.allclose([out["origin_x"].iloc[0], out["origin_y"].iloc[0]], [5.5, 34.0])  # action-LTR both


@pytest.mark.parametrize("direction", ["ltr", "rtl"])
def test_distrust_goalkick_no_detection_falls_to_rule_point(direction):
    frames = _frames_with_gk([(100, 10.0, 5.5, 34.0, False)], direction=direction)  # not visible
    out = resolve_gk_geometry(_acts(_GOALKICK), frames=frames, distrust_native_origin=True)
    assert out["origin_source"].iloc[0] == "goalkick_prior"
    assert np.allclose([out["origin_x"].iloc[0], out["origin_y"].iloc[0]], [5.5, 34.0])


@pytest.mark.parametrize("direction", ["ltr", "rtl"])
def test_distrust_never_trusts_native_for_goalkick(direction):
    frames = _frames_with_gk([(100, 10.0, 5.5 if direction == "ltr" else 99.5, 34.0, True)], direction=direction)
    out = resolve_gk_geometry(_acts(_GOALKICK), frames=frames, distrust_native_origin=True)
    assert out["origin_x"].iloc[0] != 25.0  # native ball-event origin never used


# --------------------------------------------------------------------------------------
# Open-play GK passes are NOT distrusted (native IS the keeper) -- home AND away
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("direction", ["ltr", "rtl"])
def test_distrust_openplay_gk_pass_keeps_native(direction):
    # even with the keeper detected elsewhere, an open-play GK pass keeps its NATIVE origin.
    frames = _frames_with_gk([(100, 10.0, 45.0, 34.0, True)], direction=direction)
    out = resolve_gk_geometry(_acts(_PASS, start_x=12.0), frames=frames, distrust_native_origin=True)
    assert out["origin_source"].iloc[0] == "native"
    assert out["origin_x"].iloc[0] == 12.0  # native kept (no distrust, no unresolved)


def test_distrust_openplay_gk_pass_nan_native_no_detection_unresolved():
    # the rare residual: a NaN-native open-play pass with no frames -> unresolved (honest).
    a = _acts(_PASS, start_x=12.0)
    a.loc[0, "start_x"] = np.nan
    out = resolve_gk_geometry(a, frames=None, distrust_native_origin=True)
    assert out["origin_source"].iloc[0] == "unresolved"


# --------------------------------------------------------------------------------------
# S4 out-of-region guard
# --------------------------------------------------------------------------------------
def test_s4_flags_and_warns_out_of_region_native_goalkick():
    actions = pd.DataFrame([dict(type_id=_GOALKICK), dict(type_id=_GOALKICK), dict(type_id=_PASS)])
    geom = pd.DataFrame({"origin_x": [40.0, 5.5, 90.0], "origin_source": ["native", "native", "native"]})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        flags = flag_native_goalkick_out_of_region(actions, geom)
    assert list(flags) == [True, False, False]
    assert any("goal-kick" in str(x.message) for x in w)


def test_s4_ignores_imputed_origins():
    actions = pd.DataFrame([dict(type_id=_GOALKICK)])
    geom = pd.DataFrame({"origin_x": [40.0], "origin_source": ["tracking_gk"]})
    flags = flag_native_goalkick_out_of_region(actions, geom)
    assert list(flags) == [False]


# --------------------------------------------------------------------------------------
# Regression discrimination + M1 destination invariant -- home AND away
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("direction", ["ltr", "rtl"])
def test_full_tracking_path_byte_identical_to_no_flag(direction):
    frames = _frames_with_gk([(100, 10.0, 5.5, 34.0, True)], direction=direction)
    frames["source_provider"] = "gradientsports"
    acts = _acts(_GOALKICK)
    acts["start_x"] = 5.5  # native (trusted) origin present
    baseline = resolve_gk_geometry(acts, frames=frames)  # default flag off
    trusted = resolve_gk_geometry(acts, frames=frames, distrust_native_origin=False)
    pd.testing.assert_frame_equal(baseline, trusted)
    assert trusted["origin_source"].iloc[0] == "native"


@pytest.mark.parametrize("direction", ["ltr", "rtl"])
def test_distrust_changes_origin_only_destination_byte_identical(direction):
    # M1: distrust is origin-only -> destination columns byte-identical with the flag on vs off.
    frames = _frames_with_gk([(100, 10.0, 5.5 if direction == "ltr" else 99.5, 34.0, True)], direction=direction)
    acts = _acts(_GOALKICK)
    acts["end_x"], acts["end_y"] = np.nan, np.nan  # force destination onto the next_event ladder
    nxt = _acts(_PASS)
    nxt["action_id"] = 1
    nxt["start_x"], nxt["start_y"] = 60.0, 30.0
    acts = pd.concat([acts, nxt], ignore_index=True)
    off = resolve_gk_geometry(acts, frames=frames, distrust_native_origin=False)
    on = resolve_gk_geometry(acts, frames=frames, distrust_native_origin=True)
    dest_cols = ["dest_x", "dest_y", "dest_source"]
    pd.testing.assert_frame_equal(off[dest_cols], on[dest_cols])  # destination untouched by distrust
    assert not off["origin_source"].equals(on["origin_source"])  # origin DID change (sanity)
