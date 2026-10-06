"""TF-58 Task 10: vector-coding coupling-angle classification."""

from __future__ import annotations

import numpy as np

from silly_kicks.coordination._kernels._vector_coding import (
    PATTERNS,
    classify,
    coupling_angle_deg,
    stationary_mask,
)

# (edge, class just below, class just above) per Moura 2016 Table 1.
_EDGE_CASES = [
    (22.5, "a_phase", "in_phase"),
    (67.5, "in_phase", "b_phase"),
    (112.5, "b_phase", "anti_phase"),
    (157.5, "anti_phase", "a_phase"),
    (202.5, "a_phase", "in_phase"),
    (247.5, "in_phase", "b_phase"),
    (292.5, "b_phase", "anti_phase"),
    (337.5, "anti_phase", "a_phase"),
]


def _name(angle: float) -> str:
    return PATTERNS[int(classify(angle))]


def test_table1_edges_both_sides():
    for edge, below, above in _EDGE_CASES:
        assert _name(edge - 1e-9) == below, (edge, "below")
        assert _name(edge + 1e-9) == above, (edge, "above")
        assert _name(edge) == above, (edge, "on-edge is the upper bin")
    assert _name(0.0) == "a_phase"
    assert _name(360.0) == "a_phase"


def test_four_quadrant_angle():
    assert float(coupling_angle_deg(1.0, 1.0)) == 45.0
    assert float(coupling_angle_deg(-1.0, 1.0)) == 135.0
    assert float(coupling_angle_deg(-1.0, -1.0)) == 225.0
    assert float(coupling_angle_deg(1.0, -1.0)) == 315.0


def test_printed_eq2_is_wrong():
    """C19 regression pin: the printed abs-value Eq. 2 (Moura) gives a wrong ANGLE and, at 135/315, a wrong CLASS."""

    def printed(da: float, db: float) -> float:
        return float(np.degrees(np.arctan(np.abs(db / da))))

    # 225 deg coupling: printed collapses to 45 (wrong angle) but the CLASS is coincidentally in_phase both ways.
    assert printed(-1.0, -1.0) == 45.0
    assert float(coupling_angle_deg(-1.0, -1.0)) == 225.0

    # 135 and 315: the printed form reads anti-phase as in-phase (a real misclassification).
    for da, db in ((-1.0, 1.0), (1.0, -1.0)):
        true = float(coupling_angle_deg(da, db))
        assert _name(printed(da, db)) == "in_phase"
        assert _name(true) == "anti_phase"


def test_stationary_with_epsilon_both_sides():
    eps = 0.5
    assert bool(stationary_mask(eps - 1e-12, 0.1, eps, eps)) is True
    assert bool(stationary_mask(eps + 1e-12, 0.1, eps, eps)) is False
    # exactly one side below its epsilon -> not stationary (the "and")
    assert bool(stationary_mask(0.1, eps + 1e-12, eps, eps)) is False
    assert bool(stationary_mask(eps + 1e-12, 0.1, eps, eps)) is False


def test_negative_zero_angle_maps_to_a_phase():
    # atan2(-0.0, 1.0) is -0.0 -> wrapped to 0.0, not 360 -> A-phase, no crash.
    assert float(coupling_angle_deg(1.0, -0.0)) == 0.0
    assert _name(0.0) == "a_phase"


def test_classify_is_int8_and_vectorised():
    out = classify(np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]))
    assert out.dtype == np.int8
    assert [PATTERNS[i] for i in out] == [
        "a_phase",
        "in_phase",
        "b_phase",
        "anti_phase",
        "a_phase",
        "in_phase",
        "b_phase",
        "anti_phase",
    ]
