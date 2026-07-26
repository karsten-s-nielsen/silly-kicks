"""TF-19 sign-off package: the covariate-threshold treatment axis (D5, F7, spec §5.1)."""

from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.causal import layer2_config
from silly_kicks.causal.opportunities import (
    _covariate_depth,
    _label_treatment_covariate,
    _resolve_anchor,
    shot_arm_config,
    xcross_config,
)


def test_action_path_is_untouched_when_the_covariate_axis_is_unset():
    for cfg in (xcross_config({}), shot_arm_config({})):
        assert cfg.treatment_covariate is None
        assert cfg.treatment_threshold_m is None


@pytest.mark.parametrize("gk_r,gk_theta,expected_z", [(20.0, 0.0, 1), (10.0, 0.0, 0)])
def test_treatment_binarises_at_the_penalty_area_line(gk_r, gk_theta, expected_z):
    feats = {"GK_r": gk_r, "GK_theta": gk_theta}
    assert _label_treatment_covariate(feats, "gk_depth_x", 16.5) == expected_z


def test_depth_is_the_x_component_not_the_radius():
    """VACUITY GUARD. A fixture of on-axis keepers passes identically if `GK_r` is thresholded
    directly. This wide case discriminates: r=20 but x=14.1, so a keeper 20 m from goal on a
    diagonal is INSIDE the 16.5 m depth line and must read as CONTROL."""
    feats = {"GK_r": 20.0, "GK_theta": np.pi / 4}
    assert _covariate_depth(feats) == pytest.approx(20.0 * np.cos(np.pi / 4))
    assert _covariate_depth(feats) < 16.5
    assert _label_treatment_covariate(feats, "gk_depth_x", 16.5) == 0
    # ... and the naive form would have called it TREATED, which is the bug this guards
    assert feats["GK_r"] >= 16.5


def test_unknown_covariate_raises():
    with pytest.raises(ValueError, match="unknown treatment_covariate"):
        _label_treatment_covariate({"GK_r": 1.0, "GK_theta": 0.0}, "not_a_covariate", 16.5)


def test_covariate_treated_rows_anchor_at_ENTRY_not_at_none():
    """`_row` computes the anchor from the treatment ACTION. A covariate treatment has none, so
    without `_resolve_anchor` a treated row would take anchor=None and the outcome window would
    explode on a None comparison."""
    assert _resolve_anchor(z=1, t_anchor=None, entry=12.5) == 12.5
    assert _resolve_anchor(z=1, t_anchor=30.0, entry=12.5) == 30.0
    assert _resolve_anchor(z=0, t_anchor=None, entry=12.5) == 12.5


def test_layer2_config_registers_the_landmark_design():
    cfg = layer2_config({})
    assert cfg.treatment_covariate == "gk_depth_x"
    assert cfg.treatment_threshold_m == 16.5  # Law-defined, data-independent
    assert cfg.outcome_type_names == ("shot", "shot_freekick", "shot_penalty")
    assert cfg.outcome_result_ids is None  # an ATTEMPT, not a goal
    assert cfg.outcome_max_distance_m == 16.5
    assert cfg.emit_outcome_partition is True
    assert cfg.domain == "attacking_third"
    assert cfg.extractor == "xs"


def test_every_build_confounder_is_actually_emitted_by_the_xs_extractor():
    """`_row` reads each confounder out of the feature dict with a HARD key lookup, so an
    unproduced name is a build-time KeyError, not a NaN. This is the guard that would have caught
    the first draft, which put five join-time columns into the build-time set."""
    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL

    cfg = layer2_config({})
    for name in tuple(cfg.confounders) + tuple(cfg.gk_block):
        assert name in XSHOT_FEATURE_NAMES_FAITHFUL, f"{name} would KeyError in _row"


def test_the_analysis_matrix_is_a_strict_superset_of_the_build_set():
    from silly_kicks.causal import LAYER2_BUILD_CONFOUNDERS, LAYER2_CONFOUNDERS

    assert set(LAYER2_BUILD_CONFOUNDERS) < set(LAYER2_CONFOUNDERS)
    joined = set(LAYER2_CONFOUNDERS) - set(LAYER2_BUILD_CONFOUNDERS)
    assert joined == {
        "defensive_line_height",
        "defensive_line_compactness",
        "pressure_on_actor__bekkers_pi",
        "score_differential",
        "time_remaining_s",
    }


def test_layer2_config_actually_BUILDS_opportunities():
    """The gate review HIGH 3 demanded: the field-only test above passes even when the config
    cannot be built. This one CALLS the builder, which is where a bad confounder name, a None
    anchor, or a dropped output column surfaces.

    It earned its keep immediately: `build_opportunities` pins its output to an EXPLICIT column
    list, so `_row` built the partition columns and `pd.DataFrame` silently discarded them. No unit
    test on `_partition_from_distances` could have seen that.
    """
    from causal._fixtures import layer2_frames, layer2_shot_actions

    from silly_kicks.causal import build_opportunities, layer2_config

    out = build_opportunities(
        layer2_frames(gk_x=4.0),
        layer2_shot_actions((10.0,)),
        home_team_id=5,
        model_metadata={},
        config=layer2_config({}),
    )
    assert len(out) > 0, "fixture produced no spells -- the test would be vacuous"
    for col in ("Z", "Y_attempt", "Y_close_attempt", "Y_far_attempt", "score_differential"):
        assert col in out.columns, f"{col} was built by _row but dropped from the output frame"
    assert set(out["Z"].unique()) <= {0, 1}


def test_treatment_flips_with_the_keepers_DEPTH_end_to_end():
    """Both sides through the real builder: a keeper on his line is CONTROL, one advanced beyond
    the penalty-area line is TREATED. A one-sided fixture would pass with Z hard-wired."""
    from causal._fixtures import layer2_frames, layer2_shot_actions

    from silly_kicks.causal import build_opportunities, layer2_config

    def _z(gk_x):
        out = build_opportunities(
            layer2_frames(gk_x=gk_x),
            layer2_shot_actions((10.0,)),
            home_team_id=5,
            model_metadata={},
            config=layer2_config({}),
        )
        return int(out.iloc[0]["Z"])

    assert _z(4.0) == 0  # keeper on his line
    assert _z(25.0) == 1  # keeper advanced past 16.5 m


def test_the_partition_reflects_shot_DISTANCE_end_to_end():
    """Close vs far resolved through the real outcome labeller, not the pure helper."""
    from causal._fixtures import layer2_frames, layer2_shot_actions

    from silly_kicks.causal import build_opportunities, layer2_config

    def _row(distance):
        out = build_opportunities(
            layer2_frames(gk_x=4.0),
            layer2_shot_actions((distance,)),
            home_team_id=5,
            model_metadata={},
            config=layer2_config({}),
        )
        return out.iloc[0]

    close = _row(10.0)
    assert (close["Y_attempt"], close["Y_close_attempt"], close["Y_far_attempt"]) == (1, 1, 0)
    far = _row(25.0)
    assert (far["Y_attempt"], far["Y_close_attempt"], far["Y_far_attempt"]) == (1, 0, 1)
