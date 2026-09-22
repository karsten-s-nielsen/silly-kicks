"""TF-56 compute_positioning_gap + PositioningReport + summarize (spec sections 6/7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.positioning import (
    PositioningReport,
    compute_positioning_gap,
    summarize_positioning_gap,
)
from silly_kicks.positioning._report import POSITIONING_GAP_SOURCE_VALUES


def test_conserves_over_the_mixed_fixture(mixed_domain_frames, fitted_xt):
    samples, report = compute_positioning_gap(mixed_domain_frames, xt=fitted_xt)
    assert isinstance(report, PositioningReport)
    assert report.conserves()
    assert report.n_frames_scored + sum(report.drop_reasons.values()) == report.n_frames_in
    assert report.n_frames_in == len(samples)


def test_source_tokens_are_closed_and_per_condition(mixed_domain_frames, fitted_xt):
    samples, _ = compute_positioning_gap(mixed_domain_frames, xt=fitted_xt)
    assert set(samples["positioning_gap_source"]) <= set(POSITIONING_GAP_SOURCE_VALUES)

    by_frame = {
        (int(g), int(p), int(f)): s
        for g, p, f, s in zip(
            samples["game_id"],
            samples["period_id"],
            samples["frame_id"],
            samples["positioning_gap_source"],
            strict=True,
        )
    }
    assert by_frame[(1, 1, 1)] == "scored"
    assert by_frame[(1, 1, 2)] == "excluded_out_of_domain"  # ball far from the defended goal
    assert by_frame[(1, 1, 3)] == "excluded_not_two_teams"  # NOT folded into out_of_domain
    assert by_frame[(1, 1, 4)] == "velocity_unavailable"
    assert by_frame[(2, 1, 1)] == "unresolved_geometry"
    assert by_frame[(1, 1, 6)] == "degenerate_no_movable"


def test_gap_ge_zero_and_honest_nan(mixed_domain_frames, fitted_xt):
    samples, _ = compute_positioning_gap(mixed_domain_frames, xt=fitted_xt)
    scored = samples[samples["positioning_gap_source"] == "scored"]
    dropped = samples[samples["positioning_gap_source"] != "scored"]
    assert (scored["positioning_gap"] >= 0.0).all()
    # Honest-NaN: every non-scored row has NaN gap (never a fabricated 0).
    assert dropped["positioning_gap"].isna().all()
    assert dropped["threat_actual"].isna().all()
    assert dropped["threat_optimum"].isna().all()


def test_scored_frame_has_a_positive_reachable_gap(mixed_domain_frames, fitted_xt):
    samples, _ = compute_positioning_gap(mixed_domain_frames, xt=fitted_xt)
    scored = samples[samples["positioning_gap_source"] == "scored"]
    assert len(scored) == 1
    assert float(scored["positioning_gap"].iloc[0]) > 0.0  # a real reachable reposition exists


def test_undeclared_missing_velocity_raises(one_frame, fitted_xt):
    broken = one_frame.drop(columns=["vx", "vy", "speed_source", "speed"])
    with pytest.raises(ValueError, match=r"vx|vy|velocit"):
        compute_positioning_gap(broken, xt=fitted_xt)


def test_declared_velocity_unavailable_is_counted_not_raised(one_frame, fitted_xt):
    declared = one_frame.copy()
    declared["speed_source"] = "unavailable"
    samples, report = compute_positioning_gap(declared, xt=fitted_xt)
    assert (samples["positioning_gap_source"] == "velocity_unavailable").all()
    assert report.drop_reasons.get("velocity_unavailable") == report.n_frames_in


def test_determinism_byte_identical_gap(mixed_domain_frames, fitted_xt):
    a, _ = compute_positioning_gap(mixed_domain_frames, xt=fitted_xt)
    b, _ = compute_positioning_gap(mixed_domain_frames, xt=fitted_xt)
    pd.testing.assert_series_equal(a["positioning_gap"], b["positioning_gap"])


def test_orientation_uses_goal_map_not_team_identity(mixed_domain_frames, fitted_xt):
    """Away-defending coherence: mirror the whole pitch (teams swap ends) and the scored gap must
    stay >= 0 and finite -- orientation is goal_map-driven, not keyed on team identity."""
    mirrored = mixed_domain_frames.copy()
    nonball = ~mirrored["is_ball"].astype(bool)
    # Point-reflect every position + the ball; velocities negate. Home now defends x=105.
    for col, flip in (("x", 105.0), ("y", 68.0)):
        mirrored[col] = np.where(mirrored[col].notna(), flip - mirrored[col], mirrored[col])
    for col in ("vx", "vy"):
        mirrored[col] = -mirrored[col]
    _ = nonball  # positions flipped for all rows incl. ball
    samples, report = compute_positioning_gap(mirrored, xt=fitted_xt)
    scored = samples[samples["positioning_gap_source"] == "scored"]
    assert report.conserves()
    assert len(scored) >= 1
    assert (scored["positioning_gap"] >= 0.0).all()
    assert np.isfinite(scored["positioning_gap"]).all()


def test_summarize_grain_and_mean_over_scored_only(mixed_domain_frames, fitted_xt):
    samples, _ = compute_positioning_gap(mixed_domain_frames, xt=fitted_xt)
    summary = summarize_positioning_gap(samples)
    assert {"game_id", "team_id", "mean_positioning_gap", "n_scored"} <= set(summary.columns)

    # The one scored frame is game 1, team 1 (home defending).
    row = summary[(summary["game_id"] == 1) & (summary["team_id"] == 1)].iloc[0]
    assert row["n_scored"] == 1
    scored_gap = float(samples.loc[samples["positioning_gap_source"] == "scored", "positioning_gap"].iloc[0])
    assert float(row["mean_positioning_gap"]) == pytest.approx(scored_gap)


def test_empty_frames_returns_empty_conserving(fitted_xt):
    empty = pd.DataFrame(
        columns=[
            "game_id",
            "period_id",
            "frame_id",
            "time_seconds",
            "team_id",
            "is_ball",
            "is_goalkeeper",
            "x",
            "y",
            "vx",
            "vy",
            "speed_source",
        ]
    )
    samples, report = compute_positioning_gap(empty, xt=fitted_xt)
    assert len(samples) == 0
    assert report.n_frames_in == 0 and report.conserves()


def test_positioning_gap_is_pure_no_mutation(mixed_domain_frames, fitted_xt):
    before = mixed_domain_frames.copy(deep=True)
    compute_positioning_gap(mixed_domain_frames, xt=fitted_xt)
    pd.testing.assert_frame_equal(mixed_domain_frames, before)
