"""The registered decision rules (spec 4.1, 4.3). Pure functions; no I/O."""

from _paired import clears_rule, fixed_sequence_ship, ghost_admission, ghost_admission_report


def test_clears_rule_needs_k_minus_1_positive_folds_AND_a_positive_mean():
    assert clears_rule([0.01, 0.01, 0.01, 0.01, -0.001]) is True  # 4/5 positive, mean > 0
    assert clears_rule([0.01, 0.01, 0.01, -0.01, -0.01]) is False  # only 3/5
    assert clears_rule([0.001, 0.001, 0.001, 0.001, -0.02]) is False  # 4/5 but mean < 0


def test_fixed_sequence_stops_when_sc_extended_fails():
    """Registered cost: `full` cannot ship if sc_extended fails, even if its own deltas clear."""
    ship, why = fixed_sequence_ship(
        sc_extended=[-0.01] * 5,  # fails
        full=[0.02] * 5,  # would have cleared
        full_vs_sc=[0.02] * 5,
    )
    assert ship == "public"
    assert "sc_extended failed" in why


def test_full_displaces_sc_extended_only_by_sign_consistency_not_a_mean():
    """The rev-1 bare-mean tie-break is GONE (spec 4.1). A higher mean is not enough."""
    ship, _ = fixed_sequence_ship(
        sc_extended=[0.02] * 5,
        full=[0.03] * 5,
        full_vs_sc=[0.05, -0.01, -0.01, -0.01, -0.01],  # higher MEAN, but 1/5 folds -> fails
    )
    assert ship == "sc_extended", "ties go to less data, not to noise"


def test_full_ships_when_it_dominates_fold_by_fold():
    ship, _ = fixed_sequence_ship(sc_extended=[0.02] * 5, full=[0.03] * 5, full_vs_sc=[0.01] * 5)
    assert ship == "full"


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
