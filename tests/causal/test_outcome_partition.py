"""TF-19 sign-off package: the spatial outcome filter + the registered partition (D3, spec §5.1)."""

from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.causal.opportunities import (
    _outcome_distance_m,
    _partition_from_distances,
    shot_arm_config,
    xcross_config,
)


def test_legacy_configs_are_byte_identical_with_the_new_fields_defaulted():
    """D3/D5 are additive -- but a default alone is not evidence, so pin both shipped configs."""
    for cfg in (xcross_config({}), shot_arm_config({})):
        assert cfg.outcome_max_distance_m is None
        assert cfg.emit_outcome_partition is False
        assert cfg.treatment_covariate is None
        assert cfg.treatment_threshold_m is None


def test_close_outcome_uses_action_ltr_distance_to_the_attacked_goal():
    # SPADL action-LTR: the attacked goal centre is (105, 34) for BOTH teams
    assert _outcome_distance_m(105.0, 34.0) == pytest.approx(0.0)
    assert _outcome_distance_m(88.5, 34.0) == pytest.approx(16.5)
    assert _outcome_distance_m(105.0, 17.5) == pytest.approx(16.5)


def test_partition_is_exact_by_construction_not_by_two_passes():
    """`Y_far := Y_attempt AND NOT Y_close` is the registered N4 PARTITION.

    A spell containing BOTH a close and a far attempt is classified CLOSE, not both. Under the
    looser "an attempt beyond D" reading that row would score (1, 1, 1), the indicators would
    overlap, and ATT(close) + ATT(far) == ATT(attempt) would fail. If this ever goes red, fix the
    CALLER, never these literals.
    """
    assert _partition_from_distances(np.array([10.0, 30.0]), 16.5) == (1, 1, 0)
    assert _partition_from_distances(np.array([30.0]), 16.5) == (1, 0, 1)
    assert _partition_from_distances(np.array([10.0]), 16.5) == (1, 1, 0)
    assert _partition_from_distances(np.array([]), 16.5) == (0, 0, 0)


@pytest.mark.parametrize(
    "distances",
    [np.array([10.0]), np.array([30.0]), np.array([10.0, 30.0]), np.array([5.0, 8.0]), np.array([])],
)
def test_the_partition_sums_to_the_total_on_every_shape(distances):
    """The arithmetic the coherence check depends on: close + far == attempt, always."""
    y_att, y_close, y_far = _partition_from_distances(distances, 16.5)
    assert y_close + y_far == y_att
