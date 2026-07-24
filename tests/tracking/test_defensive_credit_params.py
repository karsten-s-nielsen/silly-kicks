import pytest

from silly_kicks.tracking.defensive_credit._params import (
    DEFENSIVE_CREDIT_RULES,
    DefensiveCreditParams,
    _is_inside_attacked_box,
)


def test_rules_is_closed_tuple_of_ten():
    assert isinstance(DEFENSIVE_CREDIT_RULES, tuple)
    assert len(DEFENSIVE_CREDIT_RULES) == 10
    assert len(set(DEFENSIVE_CREDIT_RULES)) == 10  # no dupes
    assert "shot_block" in DEFENSIVE_CREDIT_RULES
    assert "pressure_pass_fail" in DEFENSIVE_CREDIT_RULES


def test_defaults_match_spec():
    p = DefensiveCreditParams()
    assert p.proximity_outside_box_m == 4.5
    assert p.proximity_inside_box_m == 3.0
    assert p.resulting_shot_max_actions == 10
    assert p.recovery_max_actions == 3
    assert p.through_ball_delta_xt_min == 0.02
    assert p.beaten_1v1_min_shot_xg == 0.05
    # synchronized boundary derived from the pitch third (105 / 3)
    assert p.synchronized_zone_boundary_x == pytest.approx(35.0)
    assert set(p.rules) == set(DEFENSIVE_CREDIT_RULES)  # all enabled by default


def test_rules_subset_validation():
    with pytest.raises(ValueError, match="unknown rule"):
        DefensiveCreditParams(rules=frozenset({"not_a_rule"}))


def test_negative_proximity_rejected():
    with pytest.raises(ValueError, match="proximity"):
        DefensiveCreditParams(proximity_outside_box_m=-1.0)


def test_box_membership_action_ltr():
    # attacked goal at x=105; box is x >= 105-16.5 = 88.5, |y-34| <= 20.16
    assert _is_inside_attacked_box(100.0, 34.0) is True
    assert _is_inside_attacked_box(100.0, 10.0) is False  # wide of the box
    assert _is_inside_attacked_box(80.0, 34.0) is False  # short of the box
