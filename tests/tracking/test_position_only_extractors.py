"""Position-only feature-set variants of the xShot / xCross / ghost extractors (Tasks 1-2, D1).

Position-only DROPS the velocity feature(s) so a fitted model can score on a velocity-less SB360
freeze-frame. The `feature_set` literal is EXTENDED to add `"position_only"` while keeping the reserved
`"extended"` slot (which still raises -- Chesterton's Fence). Velocity features are dropped (a shorter
vector), never NaN-filled: the feature contract raises on non-finite (ADR-050).
"""

from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.tracking import _xcross_attempt as xc
from silly_kicks.tracking import _xshot_occurrence as xs
from tests.tracking.test_ghost_gk import _make_ghost_gk_frames
from tests.tracking.test_xcross_attempt import _one_frame as _xcross_one_frame
from tests.tracking.test_xshot_occurrence import _one_frame as _xshot_one_frame

# --------------------------------------------------------------------------------------------------
# xShot (single-column drop: `speed`)
# --------------------------------------------------------------------------------------------------


def test_xshot_position_only_names_drop_only_speed():
    assert "speed" in xs.XSHOT_FEATURE_NAMES_FAITHFUL
    assert "speed" not in xs.XSHOT_FEATURE_NAMES_POSITION_ONLY
    assert xs.XSHOT_FEATURE_NAMES_POSITION_ONLY == [f for f in xs.XSHOT_FEATURE_NAMES_FAITHFUL if f != "speed"]
    assert len(xs.XSHOT_FEATURE_NAMES_POSITION_ONLY) == 26


def test_xshot_position_only_extractor_shape_and_columns():
    row = xs.extract_xshot_features(_xshot_one_frame(), gk_team_id=1, goal_x=0.0, feature_set="position_only")
    assert row.shape == (1, 26)
    assert list(row.columns) == xs.XSHOT_FEATURE_NAMES_POSITION_ONLY
    assert "speed" not in row.columns
    # NOTE: the vector legitimately carries NaN for `_nearest_k` padding (<5 players) and a missing `z`
    # column -- these are XGBoost-missing by design (present in faithful too), NOT a position-only
    # defect. The all-finite "drop, don't NaN-fill" guarantee is exercised on ghost's single-frame path
    # (Task 2), where an absent predecessor could otherwise inject NaN.


def test_xshot_position_only_accepted_at_init_and_extract():
    # RED drivers for the init/extract lift (currently raise != "faithful", :413 / :177).
    m = xs.XShotOccurrenceModel(feature_set="position_only")  # must NOT raise
    assert m.feature_set == "position_only"
    xs.extract_xshot_features(_xshot_one_frame(), gk_team_id=1, goal_x=0.0, feature_set="position_only")


def test_xshot_extended_still_raises_at_all_three_sites():
    # Regression guard (National Park Principle): lift narrows to reject ONLY "extended".
    frame = _xshot_one_frame()
    with pytest.raises(NotImplementedError):
        xs.extract_xshot_features(frame, gk_team_id=1, goal_x=0.0, feature_set="extended")
    with pytest.raises(NotImplementedError):
        xs.XShotOccurrenceModel(feature_set="extended")
    with pytest.raises(NotImplementedError):
        xs.prepare_xshot_training_data(frame, frame.iloc[:0], home_team_id=1, feature_set="extended")


def test_xshot_faithful_unchanged():
    a = xs.extract_xshot_features(_xshot_one_frame(), gk_team_id=1, goal_x=0.0)  # default faithful
    assert list(a.columns) == xs.XSHOT_FEATURE_NAMES_FAITHFUL
    assert a.shape == (1, 27)


# --------------------------------------------------------------------------------------------------
# xCross (single-column drop: `ball_speed`). NOTE: xCross has only TWO guard sites (extract + init);
# `prepare_xcross_training_data` has no feature_set guard (extract guards it downstream).
# The all-finite check is deliberately NOT asserted here: xCross carries caller-supplied confounders
# (score_differential, time flags) that are legitimately NaN on a minimal frame -- the position-only
# guarantee under test is "ball_speed is DROPPED", not "every confounder is finite".
# --------------------------------------------------------------------------------------------------


def test_xcross_position_only_names_drop_only_ball_speed():
    assert "ball_speed" in xc.XCROSS_FEATURE_NAMES_FAITHFUL
    assert "ball_speed" not in xc.XCROSS_FEATURE_NAMES_POSITION_ONLY
    assert xc.XCROSS_FEATURE_NAMES_POSITION_ONLY == [f for f in xc.XCROSS_FEATURE_NAMES_FAITHFUL if f != "ball_speed"]
    assert len(xc.XCROSS_FEATURE_NAMES_POSITION_ONLY) == 15


def test_xcross_position_only_extractor_shape_and_columns():
    row = xc.extract_xcross_features(
        _xcross_one_frame(),
        gk_team_id="B",
        goal_x=105.0,
        carrier_player_id="A1",
        score_differential=0.0,
        feature_set="position_only",
    )
    assert row.shape == (1, 15)
    assert list(row.columns) == xc.XCROSS_FEATURE_NAMES_POSITION_ONLY
    assert "ball_speed" not in row.columns


def test_xcross_position_only_accepted_at_init_and_extract():
    m = xc.XCrossAttemptModel(feature_set="position_only")  # must NOT raise
    assert m.feature_set == "position_only"
    xc.extract_xcross_features(
        _xcross_one_frame(), gk_team_id="B", goal_x=105.0, carrier_player_id="A1", feature_set="position_only"
    )


def test_xcross_extended_still_raises_at_both_sites():
    with pytest.raises(NotImplementedError):
        xc.extract_xcross_features(
            _xcross_one_frame(), gk_team_id="B", goal_x=105.0, carrier_player_id="A1", feature_set="extended"
        )
    with pytest.raises(NotImplementedError):
        xc.XCrossAttemptModel(feature_set="extended")


def test_xcross_faithful_unchanged():
    a = xc.extract_xcross_features(_xcross_one_frame(), gk_team_id="B", goal_x=105.0, carrier_player_id="A1")
    assert list(a.columns) == xc.XCROSS_FEATURE_NAMES_FAITHFUL
    assert a.shape == (1, 16)


# --------------------------------------------------------------------------------------------------
# Ghost (the ASYMMETRIC case: drops 5 velocity features -- 2 are cross-frame derivatives needing
# prev-frame state -- so position_only is a genuinely single-frame-capable extraction path (M1).
# Ghost never had a `feature_set`; this cycle ADDS it (extractor + model + prepare/batch).
# --------------------------------------------------------------------------------------------------
_GHOST_VELOCITY = {"ball_vx", "ball_vy", "ball_speed", "defensive_line_speed", "defending_centroid_vx"}


def test_ghost_position_only_names_drop_the_five_velocity_features():
    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GHOST_GK_FEATURE_NAMES_POSITION_ONLY

    assert set(GHOST_GK_FEATURE_NAMES) - set(GHOST_GK_FEATURE_NAMES_POSITION_ONLY) == _GHOST_VELOCITY
    assert len(GHOST_GK_FEATURE_NAMES_POSITION_ONLY) == 21


def test_ghost_position_only_extracts_finite_on_lone_freeze_frame():
    # The M1 single-frame obligation: a real SB360 freeze frame has NO vx/vy AND no predecessor. The 5
    # velocity features (incl the 2 cross-frame derivatives) are dropped, so the 21-vector is finite
    # even with no vx/vy columns and no prev-frame state -- proving position_only ghost is genuinely
    # single-frame-capable, not a re-labelled velocity path.
    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES_POSITION_ONLY, extract_ghost_gk_features

    frame = _make_ghost_gk_frames(home_team_id=1, away_team_id=2).drop(columns=["vx", "vy"])  # SB360 shape
    row = extract_ghost_gk_features(
        frame, gk_team_id=1, goal_x=0.0, ball_carrier_team_id=2, feature_set="position_only"
    )  # NO prev_defensive_line_x / prev_defending_centroid_x -> no predecessor
    assert row.shape == (1, 21)
    assert list(row.columns) == GHOST_GK_FEATURE_NAMES_POSITION_ONLY
    assert np.isfinite(row.iloc[0].to_numpy(dtype=float)).all()


def test_ghost_position_only_accepted_at_model_and_extract():
    from silly_kicks.tracking._ghost_gk import GhostGkModel, extract_ghost_gk_features

    m = GhostGkModel(feature_set="position_only")  # must NOT raise
    assert m.feature_set == "position_only"
    frame = _make_ghost_gk_frames(home_team_id=1, away_team_id=2)
    extract_ghost_gk_features(frame, gk_team_id=1, goal_x=0.0, feature_set="position_only")


def test_ghost_faithful_unchanged():
    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, extract_ghost_gk_features

    frame = _make_ghost_gk_frames(home_team_id=1, away_team_id=2)
    a = extract_ghost_gk_features(frame, gk_team_id=1, goal_x=0.0)  # default faithful
    assert list(a.columns) == GHOST_GK_FEATURE_NAMES
    assert a.shape == (1, 26)


def test_ghost_position_only_prepare_yields_21_col_matrix():
    # prepare_ghost_gk_training_data(feature_set="position_only") end-to-end, incl. the feature-width
    # check: a hardcoded `!= len(GHOST_GK_FEATURE_NAMES)` (26) would reject the valid 21-col matrix
    # with "Expected 26 features, got 21". Asserts rows are produced (the non-empty path is exercised).
    from silly_kicks.tracking import prepare_ghost_gk_training_data
    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES_POSITION_ONLY
    from tests.tracking.test_ghost_gk import _make_multi_frame_fixture

    frames = _make_multi_frame_fixture(n_frames=5)
    features, labels = prepare_ghost_gk_training_data(
        frames, home_team_id=1, subsample_fps=None, feature_set="position_only"
    )
    assert features.shape[0] > 0  # rows produced -> the non-empty path + width check both exercised
    assert features.shape[0] == labels.shape[0]
    assert list(features.columns) == GHOST_GK_FEATURE_NAMES_POSITION_ONLY
    assert features.shape[1] == 21


def test_ghost_position_only_extract_all_empty_fallback_is_21_cols():
    # The empty-return fallback (no GK feature rows produced) must mirror the non-empty path's
    # feature_set columns. Triggered with NON-EMPTY, GK-less frames: truly-empty frames fail earlier
    # (the home_team_id type-normalization reads frames["team_id"].iloc[0]), and prepare's own
    # len(meta)==0 short-circuit builds its OWN empty columns -- so this consistency fix is observable
    # only on a DIRECT _extract_all call. A hardcoded 26-col empty frame here would be internally
    # inconsistent with the 21-col non-empty concat.
    from silly_kicks.tracking._ghost_gk import (
        GHOST_GK_FEATURE_NAMES_POSITION_ONLY,
        _extract_all_ghost_gk_features,
    )
    from tests.tracking.test_ghost_gk import _make_multi_frame_fixture

    frames = _make_multi_frame_fixture(n_frames=5)
    gk_less = frames[~frames["is_goalkeeper"].astype(bool)].reset_index(drop=True)  # non-empty, no GK
    feats, meta = _extract_all_ghost_gk_features(gk_less, home_team_id=1, feature_set="position_only")
    assert len(meta) == 0  # no GK -> the empty fallback branch is the one exercised
    assert list(feats.columns) == GHOST_GK_FEATURE_NAMES_POSITION_ONLY
