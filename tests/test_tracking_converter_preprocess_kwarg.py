"""End-to-end test: tracking-converter preprocess kwarg round-trip.

PR-S24 / Loop 5 Step 5.4: verify ``convert_to_frames(..., preprocess=...)``
actually invokes the interp -> smooth -> velocity chain on a real converter,
not just resolves the config in isolation.

Uses the PFF native converter as the test bed (simplest signature; integer
ids; no kloppy dependency).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.preprocess import PreprocessConfig


def _toy_pff_raw_frames(n: int = 60, hz: float = 30.0) -> pd.DataFrame:
    """Minimal PFF-shaped raw frames satisfying EXPECTED_INPUT_COLUMNS."""
    rng = np.random.default_rng(42)
    t = np.arange(n) / hz
    rows = []
    for player_id in (1, 2, 99):
        is_gk = player_id == 99
        team_id = 100 if player_id != 99 else 200
        for i, ti in enumerate(t):
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": i,
                    "time_seconds": float(ti),
                    "frame_rate": hz,
                    "player_id": player_id,
                    "team_id": team_id,
                    "is_ball": False,
                    "is_goalkeeper": is_gk,
                    "x_centered": 0.1 * i + rng.normal(0, 0.05),
                    "y_centered": 0.0 + rng.normal(0, 0.05),
                    "z": np.nan,
                    "speed_native": np.nan,
                    "ball_state": "alive",
                }
            )
    # Add a few ball rows
    for i, ti in enumerate(t):
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": i,
                "time_seconds": float(ti),
                "frame_rate": hz,
                "player_id": pd.NA,
                "team_id": pd.NA,
                "is_ball": True,
                "is_goalkeeper": False,
                "x_centered": 0.0,
                "y_centered": 0.0,
                "z": 0.0,
                "speed_native": np.nan,
                "ball_state": "alive",
            }
        )
    df = pd.DataFrame(rows)
    df["player_id"] = df["player_id"].astype("Int64")
    df["team_id"] = df["team_id"].astype("Int64")
    return df


def test_preprocess_none_zero_behaviour_change():
    """Backcompat: default kwarg None -> output has no preprocess columns."""
    from silly_kicks.tracking.pff import convert_to_frames

    raw = _toy_pff_raw_frames()
    frames, _ = convert_to_frames(
        raw,
        home_team_id=100,
        home_team_start_left=True,
        output_convention="ltr",
    )
    assert "vx" not in frames.columns
    assert "x_smoothed" not in frames.columns
    assert "_preprocessed_with" not in frames.columns


def test_preprocess_default_auto_promotes_to_pff():
    """PreprocessConfig.default() in PFF converter auto-promotes to for_provider('pff')."""
    from silly_kicks.tracking.pff import convert_to_frames

    raw = _toy_pff_raw_frames()
    frames, _ = convert_to_frames(
        raw,
        home_team_id=100,
        home_team_start_left=True,
        output_convention="ltr",
        preprocess=PreprocessConfig.default(),
    )
    assert "x_smoothed" in frames.columns
    assert "y_smoothed" in frames.columns
    assert "vx" in frames.columns
    assert "vy" in frames.columns
    assert "speed" in frames.columns
    assert "_preprocessed_with" in frames.columns
    # PFF tag uses sg_window_seconds=0.333 (from for_provider auto-promotion).
    tag = frames["_preprocessed_with"].iloc[0]
    assert "savgol" in tag
    assert "0.333" in tag


def test_preprocess_custom_config_passes_through():
    """Caller-built config (non-default) is NOT auto-promoted."""
    from silly_kicks.tracking.pff import convert_to_frames

    cfg = PreprocessConfig(
        smoothing_method="ema",
        sg_window_seconds=1.0,
        sg_poly_order=3,
        ema_alpha=0.5,
        interpolation_method="linear",
        max_gap_seconds=2.0,
        derive_velocity=True,
    )
    raw = _toy_pff_raw_frames()
    frames, _ = convert_to_frames(
        raw,
        home_team_id=100,
        home_team_start_left=True,
        output_convention="ltr",
        preprocess=cfg,
    )
    tag = frames["_preprocessed_with"].iloc[0]
    assert "ema" in tag


def test_preprocess_raw_x_y_preserved():
    """Hyrum-Law: raw x/y unchanged by smoothing inside the converter."""
    from silly_kicks.tracking.pff import convert_to_frames

    raw = _toy_pff_raw_frames()
    frames_no_pp, _ = convert_to_frames(
        raw,
        home_team_id=100,
        home_team_start_left=True,
        output_convention="ltr",
    )
    frames_pp, _ = convert_to_frames(
        raw,
        home_team_id=100,
        home_team_start_left=True,
        output_convention="ltr",
        preprocess=PreprocessConfig.default(),
    )
    # Raw x/y identical regardless of preprocess.
    pd.testing.assert_series_equal(
        frames_no_pp["x"].reset_index(drop=True),
        frames_pp["x"].reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        frames_no_pp["y"].reset_index(drop=True),
        frames_pp["y"].reset_index(drop=True),
        check_names=False,
    )


def test_preprocess_skip_smoothing_skip_velocity():
    """Caller can disable smoothing AND velocity (config rejects derive_velocity=True+smoothing=None;
    must be derive_velocity=False to be valid)."""
    from silly_kicks.tracking.pff import convert_to_frames

    cfg = PreprocessConfig(
        smoothing_method=None,
        sg_window_seconds=0.4,
        sg_poly_order=3,
        ema_alpha=0.3,
        interpolation_method="linear",
        max_gap_seconds=0.5,
        derive_velocity=False,
    )
    raw = _toy_pff_raw_frames()
    frames, _ = convert_to_frames(
        raw,
        home_team_id=100,
        home_team_start_left=True,
        output_convention="ltr",
        preprocess=cfg,
    )
    # Interp ran (config.interpolation_method != None) but smooth/velocity skipped.
    assert "x_smoothed" not in frames.columns
    assert "vx" not in frames.columns


def test_preprocess_disallowed_combo_rejected_at_construction():
    """C1 fix: derive_velocity=True + smoothing_method=None must raise at construction
    BEFORE the converter sees it."""
    with pytest.raises(ValueError, match="derive_velocity=True requires smoothing_method"):
        PreprocessConfig(
            smoothing_method=None,
            sg_window_seconds=0.4,
            sg_poly_order=3,
            ema_alpha=0.3,
            interpolation_method="linear",
            max_gap_seconds=0.5,
            derive_velocity=True,
        )
