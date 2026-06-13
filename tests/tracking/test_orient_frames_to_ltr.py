"""Unit tests for tracking.orient_frames_to_ltr.

The helper takes UNLABELED absolute frames (team_attacking_direction all-null) and
returns canonical home-attacks-right (LTR) frames: home attacks high x in every
period, away attacks low x. It is the unlabeled-input sibling of play_left_to_right
(which serves already-labeled frames). Decision: ADR-029.

Frame-convention reminder: "home attacks LTR / high x" => home's own goal at x=0,
so the HOME GK sits near x=0 and the AWAY GK near x=105, in EVERY period.
"""

import numpy as np
import pandas as pd
import pytest

# Import from the defining module so Task 1 is self-contained (the package-level
# export is added in Task 2, validated by test_exported_from_tracking_namespace).
from silly_kicks.tracking.utils import orient_frames_to_ltr


def _make_frame(
    period_id: int,
    frame_id: int,
    *,
    home_team_id=100,
    away_team_id=200,
    home_gk_x: float = 5.0,
    away_gk_x: float = 100.0,
    ball_x: float = 52.5,
    ball_y: float = 34.0,
    direction=None,  # UNLABELED by default (absolute frames)
) -> pd.DataFrame:
    """One frame: home GK, away GK, ball. Absolute (unlabeled) by default."""
    base = {
        "game_id": 1,
        "period_id": period_id,
        "frame_id": frame_id,
        "time_seconds": frame_id / 25.0,
        "frame_rate": 25.0,
        "z": float("nan"),
        "speed": 0.0,
        "speed_source": "native",
        "ball_state": "alive",
        "confidence": None,
        "visibility": None,
        "source_provider": "metrica",
    }
    rows = [
        {
            **base,
            "player_id": "HOME-GK",
            "team_id": home_team_id,
            "is_ball": False,
            "is_goalkeeper": True,
            "x": home_gk_x,
            "y": 34.0,
            "team_attacking_direction": direction,
        },
        {
            **base,
            "player_id": "AWAY-GK",
            "team_id": away_team_id,
            "is_ball": False,
            "is_goalkeeper": True,
            "x": away_gk_x,
            "y": 34.0,
            "team_attacking_direction": direction,
        },
        {
            **base,
            "player_id": None,
            "team_id": None,
            "is_ball": True,
            "is_goalkeeper": False,
            "x": ball_x,
            "y": ball_y,
            "team_attacking_direction": None,
        },
    ]
    return pd.DataFrame(rows)


def _two_period_absolute():
    """Home physically attacks right in P1 (home GK at x=5), left in P2 (home GK at x=100).

    home_team_start_left = True (home's own goal on the left in P1).
    """
    p1 = _make_frame(1, 0, home_gk_x=5.0, away_gk_x=100.0)
    p2 = _make_frame(2, 100, home_gk_x=100.0, away_gk_x=5.0)
    return pd.concat([p1, p2], ignore_index=True)


# --- per-period orientation (spec test #2) ---


def test_home_gk_low_x_both_periods():
    out = orient_frames_to_ltr(_two_period_absolute(), home_team_id=100, home_team_start_left=True)
    for pid in (1, 2):
        home_gk = out[(out["period_id"] == pid) & (out["player_id"] == "HOME-GK")].iloc[0]
        away_gk = out[(out["period_id"] == pid) & (out["player_id"] == "AWAY-GK")].iloc[0]
        assert abs(home_gk["x"] - 5.0) < 0.01, f"P{pid} home GK x={home_gk['x']}"
        assert abs(away_gk["x"] - 100.0) < 0.01, f"P{pid} away GK x={away_gk['x']}"


def test_direction_labels_after_orient():
    out = orient_frames_to_ltr(_two_period_absolute(), home_team_id=100, home_team_start_left=True)
    home = out[out["player_id"] == "HOME-GK"]
    away = out[out["player_id"] == "AWAY-GK"]
    assert set(home["team_attacking_direction"].unique()) == {"ltr"}
    assert set(away["team_attacking_direction"].unique()) == {"rtl"}


def test_ball_player_distance_preserved_per_period():
    frames = _two_period_absolute()
    out = orient_frames_to_ltr(frames, home_team_id=100, home_team_start_left=True)
    for pid in (1, 2):
        for pl in ("HOME-GK", "AWAY-GK"):
            raw = frames[(frames["period_id"] == pid)]
            o = out[(out["period_id"] == pid)]
            rp = raw[raw["player_id"] == pl].iloc[0]
            rb = raw[raw["is_ball"]].iloc[0]
            op = o[o["player_id"] == pl].iloc[0]
            ob = o[o["is_ball"]].iloc[0]
            raw_d = np.hypot(rp["x"] - rb["x"], rp["y"] - rb["y"])
            out_d = np.hypot(op["x"] - ob["x"], op["y"] - ob["y"])
            assert abs(raw_d - out_d) < 0.01


def test_extra_time_positive_orientation():  # review concern A
    """P3/P4 orient correctly when home_team_start_left_extratime IS supplied.

    With home_team_start_left=True, home_team_start_left_extratime=False, the
    per-period home-attacks-right flags are P1=True, P2=False, P3=False, P4=True
    (home_attacks_right_per_period). So in ABSOLUTE frames the home GK (own goal
    behind the attack) sits at x=5 in P1/P4 and x=100 in P2/P3. After orient, the
    home GK must land at x≈5 (and away GK x≈100) in ALL FOUR periods. This also
    de-risks the shared ET chain that the sportec/GS adapters use.
    """
    home_gk_abs = {1: 5.0, 2: 100.0, 3: 100.0, 4: 5.0}
    away_gk_abs = {1: 100.0, 2: 5.0, 3: 5.0, 4: 100.0}
    parts = [_make_frame(p, p * 100, home_gk_x=home_gk_abs[p], away_gk_x=away_gk_abs[p]) for p in (1, 2, 3, 4)]
    frames = pd.concat(parts, ignore_index=True)
    out = orient_frames_to_ltr(
        frames, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=False
    )
    for pid in (1, 2, 3, 4):
        home_gk = out[(out["period_id"] == pid) & (out["player_id"] == "HOME-GK")].iloc[0]
        away_gk = out[(out["period_id"] == pid) & (out["player_id"] == "AWAY-GK")].iloc[0]
        assert abs(home_gk["x"] - 5.0) < 0.01, f"P{pid} home GK x={home_gk['x']}"
        assert abs(away_gk["x"] - 100.0) < 0.01, f"P{pid} away GK x={away_gk['x']}"


# --- guards ---


def test_zero_match_raises():  # spec test #4 (C1)
    frames = _make_frame(1, 0)
    with pytest.raises(ValueError, match="ZERO player rows"):
        orient_frames_to_ltr(frames, home_team_id=999, home_team_start_left=True)


def test_missing_column_raises():  # spec test #4b (C5)
    frames = _make_frame(1, 0).drop(columns=["team_id"])
    with pytest.raises(ValueError, match="missing required columns"):
        orient_frames_to_ltr(frames, home_team_id=100, home_team_start_left=True)


def test_et_without_flag_raises():  # spec test #3
    frames = _make_frame(3, 0)
    with pytest.raises(ValueError, match="home_team_start_left_extratime"):
        orient_frames_to_ltr(frames, home_team_id=100, home_team_start_left=True)


def test_already_labeled_raises():  # spec test #6 (C2)
    frames = _make_frame(1, 0, direction="ltr")  # labeled on entry
    with pytest.raises(ValueError, match="play_left_to_right"):
        orient_frames_to_ltr(frames, home_team_id=100, home_team_start_left=True)


def test_double_call_raises():  # spec test #6 (C2): second call is labeled
    out = orient_frames_to_ltr(_two_period_absolute(), home_team_id=100, home_team_start_left=True)
    with pytest.raises(ValueError, match="play_left_to_right"):
        orient_frames_to_ltr(out, home_team_id=100, home_team_start_left=True)


def test_empty_frame_returns_copy():
    empty = _make_frame(1, 0).iloc[0:0]
    out = orient_frames_to_ltr(empty, home_team_id=100, home_team_start_left=True)
    assert len(out) == 0


# --- mirror-invariance (spec test #1, C6) ---


def test_mirror_invariance():
    """orient(F, flag) == orient(mirror(F), not flag)."""
    f = _two_period_absolute()
    mirror = f.copy()
    is_player_or_ball = mirror["x"].notna()
    mirror.loc[is_player_or_ball, "x"] = 105.0 - mirror.loc[is_player_or_ball, "x"]
    mirror.loc[is_player_or_ball, "y"] = 68.0 - mirror.loc[is_player_or_ball, "y"]

    a = orient_frames_to_ltr(f, home_team_id=100, home_team_start_left=True)
    b = orient_frames_to_ltr(mirror, home_team_id=100, home_team_start_left=False)

    a_sorted = a.sort_values(["period_id", "frame_id", "player_id"]).reset_index(drop=True)
    b_sorted = b.sort_values(["period_id", "frame_id", "player_id"]).reset_index(drop=True)
    pd.testing.assert_series_equal(a_sorted["x"], b_sorted["x"], check_names=False)
    pd.testing.assert_series_equal(a_sorted["y"], b_sorted["y"], check_names=False)
    pd.testing.assert_series_equal(
        a_sorted["team_attacking_direction"], b_sorted["team_attacking_direction"], check_names=False
    )


# --- equivalence to the native sportec adapter (spec test #5) ---


def test_equivalence_to_sportec_adapter():
    """sportec.convert_to_frames(ltr) == abs-frames(from same raw) + orient_frames_to_ltr."""
    from silly_kicks.tracking import sportec

    base = {
        "game_id": 1,
        "frame_rate": 25.0,
        "z": float("nan"),
        "speed_native": float("nan"),
        "ball_state": "alive",
    }

    # Raw sportec input (x_centered, y_centered), 2 periods. Home attacks right in P1,
    # left in P2 (physical), encoded in centered coords: P1 home GK at left (x_centered=-47.5),
    # P2 home GK at right (x_centered=+47.5). home_team_start_left=True.
    def raw_row(period, frame, pid, tid, isball, isgk, xc, yc):
        return {
            **base,
            "period_id": period,
            "frame_id": frame,
            "time_seconds": frame / 25.0,
            "player_id": pid,
            "team_id": tid,
            "is_ball": isball,
            "is_goalkeeper": isgk,
            "x_centered": xc,
            "y_centered": yc,
        }

    raw = pd.DataFrame(
        [
            raw_row(1, 0, "HOME-GK", "H", False, True, -47.5, 0.0),
            raw_row(1, 0, "AWAY-GK", "A", False, True, 47.5, 0.0),
            raw_row(1, 0, None, None, True, False, 0.0, 0.0),
            raw_row(2, 100, "HOME-GK", "H", False, True, 47.5, 0.0),
            raw_row(2, 100, "AWAY-GK", "A", False, True, -47.5, 0.0),
            raw_row(2, 100, None, None, True, False, 0.0, 0.0),
        ]
    )
    adapter_out, _ = sportec.convert_to_frames(
        raw,
        home_team_id="H",
        home_team_start_left=True,
        output_convention="ltr",
    )

    # Build the equivalent UNLABELED absolute frames (x = x_centered + 52.5, y = y_centered + 34),
    # matching the sportec adapter's pre-flip absolute coordinates.
    abs_frames = raw.copy()
    abs_frames["x"] = abs_frames["x_centered"] + 52.5
    abs_frames["y"] = abs_frames["y_centered"] + 34.0
    abs_frames["speed"] = abs_frames["speed_native"]
    abs_frames["speed_source"] = None
    abs_frames["team_attacking_direction"] = None
    abs_frames["confidence"] = None
    abs_frames["visibility"] = None
    abs_frames["source_provider"] = "sportec"
    abs_frames = abs_frames.drop(columns=["x_centered", "y_centered", "speed_native"])

    helper_out = orient_frames_to_ltr(abs_frames, home_team_id="H", home_team_start_left=True)

    a = adapter_out.sort_values(["period_id", "frame_id", "is_ball", "player_id"]).reset_index(drop=True)
    h = helper_out.sort_values(["period_id", "frame_id", "is_ball", "player_id"]).reset_index(drop=True)
    pd.testing.assert_series_equal(a["x"], h["x"], check_names=False)
    pd.testing.assert_series_equal(a["y"], h["y"], check_names=False)


def test_exported_from_tracking_namespace():
    import silly_kicks.tracking as t

    assert "orient_frames_to_ltr" in t.__all__
    assert callable(t.orient_frames_to_ltr)
    # compute_attacking_direction stays private (C4)
    assert "compute_attacking_direction" not in t.__all__
