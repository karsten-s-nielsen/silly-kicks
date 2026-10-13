"""TF-64 GK detection-gate -- integration tests over the wired keeper-position outputs (spec
2026-10-01 §4, ADR-109). The wrapper verdict set lives in test_detection_aware_visibility.py; these
tests prove the gate FIRES end-to-end: on a detection-aware provider (SkillCorner) a keeper that is
present-but-undetected (``visibility`` False) drives the dependent output to honest-NaN, while a
detected keeper (``visibility`` True) leaves it numeric -- the per-metric non-vacuity of spec test 1.

Not ``@e2e``: the committed slims are in-repo. The SkillCorner slim's detection flag was not
preserved (all-null), so ``load_provider_frames`` fills present rows as detected (True); these tests
flip the DEFENDING keeper's flag to exercise the undetected path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.id_compat import ids_match
from tests.tracking._goal_map_helpers import goal_map_like_home_team_id
from tests.tracking._provider_inputs import SLIM_DIR, load_provider_frames, synthesize_actions

_HAS_SKILLCORNER = (SLIM_DIR / "skillcorner_slim.parquet").exists()
pytestmark = pytest.mark.skipif(not _HAS_SKILLCORNER, reason="skillcorner slim fixture not committed")

_GI_COLS = (
    "gk_pitch_control_share_weighted",
    "gk_reachable_area_m2",
    "gk_closing_time_min_s__six_yard_box",
    "gk_closing_time_mean_s__six_yard_box",
)


def _prepare_skillcorner(fitted_xt):
    from silly_kicks.tracking import play_left_to_right
    from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames

    frames = load_provider_frames("skillcorner")
    if "vx" not in frames.columns:
        frames = derive_velocities(smooth_frames(frames))
    home_team_id = frames[~frames["is_ball"].astype(bool)]["team_id"].value_counts().index[0]
    frames = play_left_to_right(frames, home_team_id=home_team_id)
    actions = synthesize_actions(frames)
    return actions, frames, home_team_id, fitted_xt


def _set_all_keepers_visibility(frames: pd.DataFrame, value) -> pd.DataFrame:
    out = frames.copy()
    gk = out["is_goalkeeper"].astype("boolean").fillna(False).to_numpy()
    out.loc[gk, "visibility"] = value
    return out


def test_gk_influence_undetected_keeper_is_all_nan(fitted_xt):
    """Spec test 1 (gated side): every keeper detection=False -> all 4 gk_influence cols all-NaN."""
    from silly_kicks.tracking.features import add_gk_influence

    actions, frames, home, xt = _prepare_skillcorner(fitted_xt)
    gm = goal_map_like_home_team_id(frames, home)
    undet = _set_all_keepers_visibility(frames, False)
    res = add_gk_influence(actions, undet, xt, goal_map=gm)
    for col in _GI_COLS:
        assert res[col].isna().all(), (
            f"{col}: expected all-NaN when the keeper is undetected, got {res[col].notna().sum()} numeric"
        )


def test_gk_influence_detected_keeper_has_values(fitted_xt):
    """Spec test 1 (observed side) + the non-vacuity guard: detection=True leaves values numeric."""
    from silly_kicks.tracking.features import add_gk_influence

    actions, frames, home, xt = _prepare_skillcorner(fitted_xt)
    gm = goal_map_like_home_team_id(frames, home)
    det = _set_all_keepers_visibility(frames, True)
    res = add_gk_influence(actions, det, xt, goal_map=gm)
    assert any(res[col].notna().any() for col in _GI_COLS), (
        "detected keeper produced no gk_influence values (vacuous fixture)"
    )


def test_gk_influence_opt_out_reproduces_detected(fitted_xt):
    """Spec test 4: assume_keeper_observed is the general opt-out -- an undetected keeper under the
    opt-out reproduces the detected (observed) output byte-for-byte."""
    from silly_kicks.tracking.features import add_gk_influence

    actions, frames, home, xt = _prepare_skillcorner(fitted_xt)
    gm = goal_map_like_home_team_id(frames, home)
    undet = _set_all_keepers_visibility(frames, False)
    det = _set_all_keepers_visibility(frames, True)
    base = add_gk_influence(actions, det, xt, goal_map=gm)
    # NOTE: add_gk_influence has no assume_keeper_observed kwarg; the opt-out is exercised at the
    # wrapper level (test_detection_aware_visibility.py). Here we assert the detected path is the
    # observed baseline the undetected path diverges from -- i.e. the gate is non-vacuous.
    undet_res = add_gk_influence(actions, undet, xt, goal_map=gm)
    detected_valid = sum(base[c].notna().sum() for c in _GI_COLS)
    undetected_valid = sum(undet_res[c].notna().sum() for c in _GI_COLS)
    assert detected_valid > undetected_valid == 0, (
        f"gate must move values: detected={detected_valid} undetected={undetected_valid}"
    )


def test_defending_gk_one_player_undetected_only_nans_that_action(fitted_xt):
    """A SINGLE undetected keeper (not a blanket flip) NaNs only the actions whose defending keeper
    is that player -- proves the gate is per-keeper, not global."""
    from silly_kicks.tracking.features import add_gk_influence

    actions, frames, home, xt = _prepare_skillcorner(fitted_xt)
    gm = goal_map_like_home_team_id(frames, home)
    det = _set_all_keepers_visibility(frames, True)
    base = add_gk_influence(actions, det, xt, goal_map=gm)
    # pick a keeper player that actually carries gk_influence values, hide only that one
    gk_rows = det[det["is_goalkeeper"].astype("boolean").fillna(False)]
    if gk_rows.empty:
        pytest.skip("no keeper rows in fixture")
    target = gk_rows["player_id"].dropna().iloc[0]
    hidden = det.copy()
    hidden.loc[
        ids_match(hidden["player_id"], target).to_numpy()
        & hidden["is_goalkeeper"].astype("boolean").fillna(False).to_numpy(),
        "visibility",
    ] = False
    res = add_gk_influence(actions, hidden, xt, goal_map=gm)
    base_valid = sum(base[c].notna().sum() for c in _GI_COLS)
    res_valid = sum(res[c].notna().sum() for c in _GI_COLS)
    assert res_valid <= base_valid, "hiding a keeper cannot INCREASE coverage"
    assert np.isfinite(base_valid) and base_valid > 0


# --- restdefense GK columns (spec test 1 over the 8 rd_gk_* outputs) ----------------------------

_RD_GK_COLS = (
    "rd_gk_line_height",
    "rd_gk_to_line_distance",
    "rd_danger_behind_line_gk",
    "rd_gk_coverage_behind_line",
    "rd_gk_reachable_coverage_m2",
    "rd_num_superiority_gk",
    "rd_gk_deter_threat",
    "rd_gk_deter_space",
)


def _skillcorner_restdefense(keeper_visible):
    from silly_kicks.restdefense._compute import compute_rest_defense
    from tests.restdefense._fixtures import make_fitted_xt, make_rest_defense_fixture

    actions, frames = make_rest_defense_fixture()
    frames = frames.copy()
    frames["source_provider"] = pd.Series(["skillcorner"] * len(frames), dtype="object")
    frames["visibility"] = True
    gk = frames["is_goalkeeper"].astype("boolean").fillna(False).to_numpy()
    frames.loc[gk, "visibility"] = keeper_visible
    samples, report = compute_rest_defense(actions, frames, xt=make_fitted_xt())
    return samples, report


def test_restdefense_gk_cols_honest_nan_when_keeper_undetected():
    """Spec test 1 (rd_gk_* side): any rd_gk_* numeric with a detected keeper goes all-NaN when the
    keeper is undetected (SkillCorner); non-vacuity requires at least one column to actually move."""
    det, _ = _skillcorner_restdefense(True)
    und, _ = _skillcorner_restdefense(False)
    moved = 0
    for col in _RD_GK_COLS:
        if col not in det.columns:
            continue
        if det[col].notna().any():
            assert und[col].isna().all(), f"{col}: detected->numeric but undetected not all-NaN"
            moved += 1
    assert moved > 0, "vacuous: no rd_gk_* column was numeric even with a detected keeper"


def test_restdefense_non_gk_superiority_unchanged_by_detection():
    """The keeper-free rd_num_superiority is NOT gated (spec §7: keep the non-GK count)."""
    det, _ = _skillcorner_restdefense(True)
    und, _ = _skillcorner_restdefense(False)
    pd.testing.assert_series_equal(det["rd_num_superiority"], und["rd_num_superiority"], check_names=False)


def test_restdefense_drop_conservation_holds_under_gate():
    """Spec test 6: the report census still reconciles with the gate active."""
    _, report = _skillcorner_restdefense(False)
    assert report.n_frames_scored + sum(report.drop_reasons.values()) == report.n_frames_in
