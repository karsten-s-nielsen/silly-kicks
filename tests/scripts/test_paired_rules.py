"""The registered decision rules (spec 4.1, 4.3). Pure functions; no I/O."""

from _paired import clears_rule, ghost_admission, ghost_admission_report, two_arm_ship


def test_clears_rule_needs_k_minus_1_positive_folds_AND_a_positive_mean():
    assert clears_rule([0.01, 0.01, 0.01, 0.01, -0.001]) is True  # 4/5 positive, mean > 0
    assert clears_rule([0.01, 0.01, 0.01, -0.01, -0.01]) is False  # only 3/5
    assert clears_rule([0.001, 0.001, 0.001, 0.001, -0.02]) is False  # 4/5 but mean < 0


def test_two_arm_rule_ships_sc_extended_only_when_it_clears():
    """ADR-132 (corpus widening spec 5.2): public vs sc_extended, clears_rule unchanged; ties go to less data."""
    assert two_arm_ship([0.02] * 5)[0] == "sc_extended"
    assert two_arm_ship([0.02, 0.02, 0.02, 0.02, -0.001])[0] == "sc_extended"  # 4/5 positive, mean > 0
    assert two_arm_ship([0.02, 0.02, 0.02, -0.01, -0.01])[0] == "public"  # only 3/5
    assert two_arm_ship([0.001, 0.001, 0.001, 0.001, -0.02])[0] == "public"  # 4/5 but mean < 0
    assert two_arm_ship([0.5])[0] == "public"  # K < 2 never clears


def test_ghost_admission_requires_demonstrated_improvement_not_a_wash():
    # delta = MAE_expanded - MAE_baseline; negative = better
    assert ghost_admission([-0.1, -0.1, -0.1, -0.1, 0.01]) is True
    assert ghost_admission([0.0, 0.0, 0.0, 0.0, 0.0]) is False  # a wash keeps the status quo
    assert ghost_admission([-0.1, -0.1, 0.1, 0.1, 0.1]) is False


def test_admission_ignores_nan_folds_rather_than_failing_on_them():
    """M5: a degenerate (single-class) fold must DROP OUT, not flip the verdict to 'don't ship'."""
    assert ghost_admission([-0.1, -0.1, -0.1, float("nan"), -0.1]) is True


def test_the_interpolator_tell_is_a_DIAGNOSTIC_not_a_gate():
    """Spec rev 5: the refusal was RETIRED because it could never change a verdict.

    Admission already requires detected-only improvement, so `improves_all and not
    improves_detected` was reachable only when the fall-through returned False anyway. And under
    rev 3's detected-only TRAINING rule the mechanism it guarded no longer exists -- the model
    never sees an interpolated target.

    What remains is a reason string, so the record can distinguish 'no improvement' from 'improved
    only where the keeper was invented'. It decides NOTHING, and this test says so out loud.
    """
    verdict, reason = ghost_admission_report(
        detected_only_deltas=[0.1] * 5,  # no improvement on SEEN keepers
        all_frames_deltas=[-0.2] * 5,  # 'improves' on interpolated ones
    )
    assert verdict is False
    assert "interpolated" in reason  # the DIAGNOSTIC fires...
    # ...and the verdict is identical without it -- which is precisely why it is not a gate.
    assert ghost_admission([0.1] * 5) is False
