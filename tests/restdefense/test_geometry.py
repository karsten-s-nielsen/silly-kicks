"""danger_zone_bounds (TF-60, ADR-080) -- the danger-behind-line strip, own-goal-oriented."""

from silly_kicks.restdefense._geometry import danger_zone_bounds


def test_danger_zone_orientation_own_goal_low_x():
    assert danger_zone_bounds(20.0, 0.0) == (0.0, 20.0)  # own goal x=0, line x=20 -> [0, 20]


def test_danger_zone_orientation_own_goal_high_x():
    assert danger_zone_bounds(85.0, 105.0) == (85.0, 105.0)  # own goal x=105, line x=85 -> [85, 105]


def test_danger_zone_capped_depth():
    assert danger_zone_bounds(20.0, 0.0, zone_depth_m=10.0) == (0.0, 10.0)
    assert danger_zone_bounds(85.0, 105.0, zone_depth_m=10.0) == (95.0, 105.0)


def test_danger_zone_cap_wider_than_strip_is_a_noop():
    # a cap deeper than the strip itself must not widen the strip past the rearguard line
    assert danger_zone_bounds(20.0, 0.0, zone_depth_m=100.0) == (0.0, 20.0)
    assert danger_zone_bounds(85.0, 105.0, zone_depth_m=100.0) == (85.0, 105.0)
