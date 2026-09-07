"""TF-60 Task 10: restdefense instrument-validity probe (mirrors tests/gkdv/test_probe_*).

The two verdict FUNCTIONS are REUSED from gkdv verbatim (public `__all__`); restdefense adds the
rearguard/keeper dose imposers + the rearguard-analog paired-vector controls + a restdefense-local
EXPECTED_DIRECTION for its 4 arm columns. Reported-not-gated (like gkdv A+2).
"""

from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.restdefense._columns import (
    RD_GK_DETER_SPACE,
    RD_GK_DETER_THREAT,
    RD_OUTFIELD_DETER_SPACE,
    RD_OUTFIELD_DETER_THREAT,
)
from silly_kicks.restdefense._probe import (
    EXPECTED_DIRECTION,
    REARGUARD_REALISTIC_MIN_DISP_M,
    expected_direction_for_arm,
    impose_rearguard_dose,
    layer0_instrument_verdict,
    layer1_responsiveness_verdict,
    paired_vector_controls,
)
from tests.restdefense.test_counterfactual import _carrier, _committed_frame
from tests.tracking.test_ghost_gk import _fitted_model
from tests.tracking.test_ghost_outfield_model import _fit_toy

_KEY4 = ["game_id", "period_id", "frame_id", "player_id"]
_REARGUARD = {"h1", "h2", "h3", "h4"}
_ALL_A_OUTFIELD = {"h1", "h2", "h3", "h4", "h5", "h6"}


def _outfield_model():
    return _fit_toy()[0]


def _gk_model():
    return _fitted_model()[0]


def _merge(imposed, frames):
    return imposed.merge(frames, on=_KEY4, suffixes=("_i", ""))


def _moved_rows(m):
    """Rows in a merged (imposed vs frames) table whose x/y changed."""
    dx = m["x_i"].to_numpy(float) != m["x"].to_numpy(float)
    dy = m["y_i"].to_numpy(float) != m["y"].to_numpy(float)
    return m[dx | dy]


# --- EXPECTED_DIRECTION ---------------------------------------------------------------------------


def test_expected_direction_all_four_arms_are_negative():
    arms = {RD_GK_DETER_THREAT, RD_GK_DETER_SPACE, RD_OUTFIELD_DETER_THREAT, RD_OUTFIELD_DETER_SPACE}
    assert set(EXPECTED_DIRECTION) == arms
    for col in arms:
        assert expected_direction_for_arm(col) == "negative"


def test_expected_direction_unmapped_arm_raises():
    with pytest.raises(KeyError):
        expected_direction_for_arm("not_an_arm_column")


# --- dose imposition ------------------------------------------------------------------------------


def test_ladder_dose_moves_only_rearguard_toward_own_goal():
    frames = _committed_frame()
    imposed, targets = impose_rearguard_dose(
        frames,
        which="rearguard",
        home_team_id=1,
        dose="ladder",
        displacement=3.0,
        model=_outfield_model(),
        carrier=_carrier(),
    )
    assert set(targets["player_id"]) == _REARGUARD
    m = _merge(imposed, frames)
    rg = m[m["player_id"].isin(_REARGUARD)]
    # A defends x=0 -> the ladder moves the rearguard 3 m toward x=0; y unchanged.
    np.testing.assert_allclose(rg["x_i"].to_numpy(float), rg["x"].to_numpy(float) - 3.0)
    np.testing.assert_allclose(rg["y_i"].to_numpy(float), rg["y"].to_numpy(float))
    # nothing else moved (keeper, forwards, opponent, ball)
    others = m[~m["player_id"].isin(_REARGUARD)]
    assert (others["x_i"].to_numpy(float) == others["x"].to_numpy(float)).all()
    assert (others["y_i"].to_numpy(float) == others["y"].to_numpy(float)).all()
    # PURE
    assert float(frames[frames["player_id"] == "h1"]["x"].iloc[0]) == 15.0


def test_saturating_goalline_puts_rearguard_on_own_goal_line_keeping_lateral_spread():
    frames = _committed_frame()
    imposed, _targets = impose_rearguard_dose(
        frames,
        which="rearguard",
        home_team_id=1,
        dose="saturating_goalline",
        model=_outfield_model(),
        carrier=_carrier(),
    )
    m = _merge(imposed, frames)
    rg = m[m["player_id"].isin(_REARGUARD)]
    assert (rg["x_i"].to_numpy(float) == 0.0).all()  # own goal line
    np.testing.assert_allclose(rg["y_i"].to_numpy(float), rg["y"].to_numpy(float))  # lateral kept


def test_keeper_dose_moves_only_As_keeper():
    frames = _committed_frame()
    imposed, targets = impose_rearguard_dose(
        frames,
        which="keeper",
        home_team_id=1,
        dose="saturating_goalline",
        model=_gk_model(),
        carrier=_carrier(),
    )
    assert set(targets["player_id"]) == {"h_gk"}
    m = _merge(imposed, frames)
    kg = m[m["player_id"] == "h_gk"]
    assert float(kg["x_i"].iloc[0]) == 0.0  # A's own goal line
    assert float(kg["y_i"].iloc[0]) == float(kg["y"].iloc[0])
    others = m[m["player_id"] != "h_gk"]
    assert (others["x_i"].to_numpy(float) == others["x"].to_numpy(float)).all()
    assert (others["y_i"].to_numpy(float) == others["y"].to_numpy(float)).all()


def test_realistic_dose_uses_the_ghost_position_and_filters_small_moves():
    frames = _committed_frame()
    _imposed, targets = impose_rearguard_dose(
        frames,
        which="rearguard",
        home_team_id=1,
        dose="realistic",
        model=_outfield_model(),
        carrier=_carrier(),
    )
    # every kept frame moved the rearguard by >= the realistic floor; imp == ghost position.
    if len(targets):
        assert targets["displacement_m"].to_numpy(float).max() >= REARGUARD_REALISTIC_MIN_DISP_M
        np.testing.assert_allclose(targets["imp_x"].to_numpy(float), targets["ghost_x"].to_numpy(float))
        np.testing.assert_allclose(targets["imp_y"].to_numpy(float), targets["ghost_y"].to_numpy(float))


def test_no_op_ladder_dose_leaves_frames_unchanged():
    # a planted no-op (displacement 0) moves nothing -> the arm delta would be 0 -> not responsive.
    frames = _committed_frame()
    imposed, _targets = impose_rearguard_dose(
        frames,
        which="rearguard",
        home_team_id=1,
        dose="ladder",
        displacement=0.0,
        model=_outfield_model(),
        carrier=_carrier(),
    )
    m = _merge(imposed, frames)
    assert (m["x_i"].to_numpy(float) == m["x"].to_numpy(float)).all()
    assert (m["y_i"].to_numpy(float) == m["y"].to_numpy(float)).all()
    assert layer1_responsiveness_verdict(gk_med=0.0, nd_med=0.05, placebo_p95=0.05, n_domain=300) == "not_responsive"


# --- paired-vector controls -----------------------------------------------------------------------


def test_paired_controls_move_exactly_one_A_outfielder_each():
    frames = _committed_frame()
    _imposed, targets = impose_rearguard_dose(
        frames,
        which="rearguard",
        home_team_id=1,
        dose="ladder",
        displacement=3.0,
        model=_outfield_model(),
        carrier=_carrier(),
    )
    controls = paired_vector_controls(frames, targets, r=2, rng=np.random.default_rng(0))
    assert set(controls) == {"nearest", "placebo_0", "placebo_1"}
    for name, cf in controls.items():
        moved = _moved_rows(_merge(cf, frames))
        assert len(moved) == 1, f"{name} moved {len(moved)} players (want exactly one)"
        pid = moved["player_id"].iloc[0]
        # the pool is A's NON-rearguard outfielders (rearguard rows are the intervention target)
        assert pid in (_ALL_A_OUTFIELD - _REARGUARD), f"{name} moved a non-pool player {pid!r}"
        assert int(moved["team_id"].iloc[0]) == 1  # A's player
    # PURE
    assert float(frames[frames["player_id"] == "h5"]["x"].iloc[0]) == 55.0


def test_paired_controls_displace_by_the_frame_mean_dose_vector():
    frames = _committed_frame()
    _imposed, targets = impose_rearguard_dose(
        frames,
        which="rearguard",
        home_team_id=1,
        dose="ladder",
        displacement=3.0,
        model=_outfield_model(),
        carrier=_carrier(),
    )
    # every rearguard player moved by (-3, 0) -> the per-frame mean vector is (-3, 0).
    controls = paired_vector_controls(frames, targets, r=1, rng=np.random.default_rng(0))
    moved = _moved_rows(_merge(controls["nearest"], frames))
    assert len(moved) == 1
    assert float(moved["x_i"].iloc[0]) == float(moved["x"].iloc[0]) - 3.0
    assert float(moved["y_i"].iloc[0]) == float(moved["y"].iloc[0])


# --- reused gkdv verdict functions (public, not the private thresholds) ---------------------------


def test_verdict_functions_are_the_gkdv_public_functions():
    import silly_kicks.gkdv as gkdv

    assert layer0_instrument_verdict is gkdv.layer0_instrument_verdict
    assert layer1_responsiveness_verdict is gkdv.layer1_responsiveness_verdict


def test_thin_domain_is_arm_unscoreable_via_the_reused_verdicts():
    # the pooled-not-per-shard thin-domain short-circuit (inherited from gkdv, never re-implemented).
    assert (
        layer0_instrument_verdict(
            realistic_abs=np.full(3, 0.05), saturating_abs=np.full(3, 0.5), placebo_p95=0.02, n_domain=3
        )
        == "arm_unscoreable"
    )
    assert layer1_responsiveness_verdict(gk_med=0.3, nd_med=0.1, placebo_p95=0.1, n_domain=3) == "arm_unscoreable"
