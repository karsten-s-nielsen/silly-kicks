"""TF-54b probe battery: pooled-corpus verdicts + direction registry + dose / paired controls."""

import numpy as np
import pytest

from silly_kicks.territorial_defense._probe import (
    MIN_DOMAIN_FRAMES,
    TD_PROBE_RATIO,
    expected_direction_for_arm,
    impose_defender_dose,
    layer0_instrument_verdict,
    layer1_responsiveness_verdict,
    paired_vector_controls,
)

from ._fixtures import ACTOR_PLAYER_ID, DEFENDING_TEAM_ID, make_rich_frame


def test_td_probe_ratio_is_its_own_registration():
    assert TD_PROBE_RATIO == 2.0  # NOT PHYSICS_ARM_PROBE_RATIO / TF19_PROBE_RATIO / XS_PROBE_RATIO


def test_layer0_arm_unscoreable_on_thin_domain():
    assert (
        layer0_instrument_verdict(
            realistic_abs=[1.0], saturating_abs=[10.0], placebo_p95=0.5, n_domain=MIN_DOMAIN_FRAMES - 1
        )
        == "arm_unscoreable"
    )


def test_layer0_valid_via_multiple_and_via_placebo():
    n = MIN_DOMAIN_FRAMES
    # saturating median >= 5x realistic median -> valid via the multiple leg
    assert (
        layer0_instrument_verdict(realistic_abs=[1.0], saturating_abs=[10.0], placebo_p95=0.0, n_domain=n)
        == "instrument_valid"
    )
    # multiple leg fails (2 < 5) but saturating > placebo p95 -> valid via the placebo leg
    assert (
        layer0_instrument_verdict(realistic_abs=[1.0], saturating_abs=[2.0], placebo_p95=1.5, n_domain=n)
        == "instrument_valid"
    )


def test_layer0_void_when_neither_leg_clears():
    n = MIN_DOMAIN_FRAMES
    assert (
        layer0_instrument_verdict(realistic_abs=[1.0], saturating_abs=[2.0], placebo_p95=3.0, n_domain=n)
        == "instrument_void"
    )


def test_layer0_zero_baseline_does_not_vacuously_validate():
    # real_med=0 -> 5*0 must NOT vacuously pass; placebo leg (0 > 0 is False) backstops -> void.
    n = MIN_DOMAIN_FRAMES
    assert (
        layer0_instrument_verdict(realistic_abs=[0.0], saturating_abs=[0.0], placebo_p95=0.0, n_domain=n)
        == "instrument_void"
    )


def test_layer1_responsive_and_not():
    n = MIN_DOMAIN_FRAMES
    # thresh = 2 * max(nd_med, placebo_p95) = 2*2 = 4
    assert layer1_responsiveness_verdict(defender_med=10.0, nd_med=1.0, placebo_p95=2.0, n_domain=n) == "responsive"
    assert layer1_responsiveness_verdict(defender_med=3.0, nd_med=1.0, placebo_p95=2.0, n_domain=n) == "not_responsive"
    assert (
        layer1_responsiveness_verdict(defender_med=10.0, nd_med=1.0, placebo_p95=2.0, n_domain=n - 1)
        == "arm_unscoreable"
    )


def test_expected_direction_positive_and_raises_on_unmapped():
    assert expected_direction_for_arm("a_threat_suppressed") == "positive"
    assert expected_direction_for_arm("b_threat_suppressed") == "positive"
    with pytest.raises(KeyError):
        expected_direction_for_arm("delta_threat")  # a gkdv arm column -- not ours


def test_impose_defender_dose_displaces_only_that_row_and_is_pure():
    frame = make_rich_frame()
    pos = int(frame.index[frame["player_id"] == ACTOR_PLAYER_ID][0])
    x0 = float(frame.iloc[pos]["x"])
    dosed = impose_defender_dose(frame, defender_pos=pos, dx=5.0, dy=-2.0)
    assert dosed is not frame
    assert float(frame.iloc[pos]["x"]) == x0  # input unmutated
    assert abs(float(dosed.iloc[pos]["x"]) - (x0 + 5.0)) < 1e-9
    # every other row unchanged
    others = [i for i in range(len(frame)) if i != pos]
    assert np.allclose(dosed.iloc[others]["x"].to_numpy(float), frame.iloc[others]["x"].to_numpy(float))


def test_paired_controls_displace_one_other_defender_each():
    frame = make_rich_frame()
    pos = int(frame.index[frame["player_id"] == ACTOR_PLAYER_ID][0])
    ctrls = paired_vector_controls(
        frame, defender_pos=pos, defending_team_id=DEFENDING_TEAM_ID, dx=5.0, dy=0.0, r=2, rng=np.random.default_rng(0)
    )
    assert "nearest" in ctrls and "placebo_0" in ctrls and "placebo_1" in ctrls
    for name, cf in ctrls.items():
        # exactly one row moved, and it is NOT the dosed defender
        moved = ~np.isclose(cf["x"].to_numpy(float), frame["x"].to_numpy(float))
        assert moved.sum() == 1, name
        assert not moved[pos], name
