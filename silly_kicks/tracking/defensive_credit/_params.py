"""Frozen params + the closed rule vocabulary + box geometry for TF-51."""

from __future__ import annotations

from dataclasses import dataclass, field

from silly_kicks.spadl import config as spadlconfig

# --- pitch + box geometry (action-LTR: acting team attacks x=105) ---
_FIELD_LENGTH: float = spadlconfig.field_length  # 105.0
_FIELD_WIDTH: float = spadlconfig.field_width  # 68.0
# Penalty-area geometry. spadlconfig ships no canonical box constant (see CLAUDE.md /
# ADR-019 discussion); the repo duplicates it. We adopt _xcross_attempt.py's values
# (16.5 depth, 20.16 half-width = 40.32/2). NOTE the cross-module discrepancy: _ghost_gk.py
# uses 40.3 (half 20.15). 0.01 m apart; neither cites the other. We pick 40.32 (the FIFA
# Laws figure) and flag it here rather than silently choosing. A canonical spadlconfig
# penalty-area constant is a tracked cross-cutting follow-up (ADR-021 "pitch dims live in
# spadlconfig"); see the TF-51 plan's Task 17.
_BOX_DEPTH_M: float = 16.5
_BOX_HALF_WIDTH_M: float = 20.16
_GOAL_Y_C: float = _FIELD_WIDTH / 2.0  # 34.0

# --- closed rule vocabulary (DAS_SOURCE_VALUES pattern) ---
RULE_PRESSURE_ON_MISSED_SHOT = "pressure_on_missed_shot"
RULE_FAILED_PRESSURE_SHOT_ON_TARGET = "failed_pressure_shot_on_target"
RULE_SHOT_BLOCK = "shot_block"
RULE_PRESSURE_PASS_FAIL = "pressure_pass_fail"  # noqa: S105 -- a rule name, not a password ("pass" substring)
RULE_RECOVERY_DOUBLE_CREDIT = "recovery_double_credit"
RULE_SYNCHRONIZED_FINAL_THIRD_PRESSURE = "synchronized_final_third_pressure"
RULE_FORCED_BAD_TOUCH = "forced_bad_touch"
RULE_FAILED_CROSS_BLOCK = "failed_cross_block"
RULE_FAILED_MARKING_THROUGH_BALL = "failed_marking_through_ball"
RULE_BEATEN_1V1 = "beaten_1v1"

DEFENSIVE_CREDIT_RULES: tuple[str, ...] = (
    RULE_PRESSURE_ON_MISSED_SHOT,
    RULE_FAILED_PRESSURE_SHOT_ON_TARGET,
    RULE_SHOT_BLOCK,
    RULE_PRESSURE_PASS_FAIL,
    RULE_RECOVERY_DOUBLE_CREDIT,
    RULE_SYNCHRONIZED_FINAL_THIRD_PRESSURE,
    RULE_FORCED_BAD_TOUCH,
    RULE_FAILED_CROSS_BLOCK,
    RULE_FAILED_MARKING_THROUGH_BALL,
    RULE_BEATEN_1V1,
)

SIZING_XG = "xg"
SIZING_XT = "xt"


def _is_inside_attacked_box(x: float, y: float) -> bool:
    """True iff (x, y) in action-LTR coords is inside the attacked penalty area (goal at x=105)."""
    return bool((x >= _FIELD_LENGTH - _BOX_DEPTH_M) and (abs(y - _GOAL_Y_C) <= _BOX_HALF_WIDTH_M))


@dataclass(frozen=True)
class DefensiveCreditParams:
    """All fields spec-frozen / intent-set -- never calibrated (see spec section 4.2, 14).

    Examples
    --------
    Construct the defaults, or enable a subset of rules::

        params = DefensiveCreditParams()
        only_shots = DefensiveCreditParams(rules=frozenset({"shot_block", "pressure_on_missed_shot"}))
    """

    proximity_outside_box_m: float = 4.5
    proximity_inside_box_m: float = 3.0
    synchronized_zone_boundary_x: float = field(default_factory=lambda: _FIELD_LENGTH / 3.0)
    resulting_shot_max_actions: int = 10
    recovery_max_actions: int = 3
    through_ball_delta_xt_min: float = 0.02  # provisional
    beaten_1v1_min_shot_xg: float = 0.05  # provisional
    rules: frozenset[str] = field(default_factory=lambda: frozenset(DEFENSIVE_CREDIT_RULES))

    def __post_init__(self) -> None:
        for name, val in (
            ("proximity_outside_box_m", self.proximity_outside_box_m),
            ("proximity_inside_box_m", self.proximity_inside_box_m),
        ):
            if not val > 0:
                raise ValueError(f"{name} must be > 0, got {val}")
        if self.resulting_shot_max_actions < 1 or self.recovery_max_actions < 1:
            raise ValueError("resulting_shot_max_actions and recovery_max_actions must be >= 1")
        unknown = set(self.rules) - set(DEFENSIVE_CREDIT_RULES)
        if unknown:
            raise ValueError(f"unknown rule(s): {sorted(unknown)}; allowed: {DEFENSIVE_CREDIT_RULES}")

    def _proximity_threshold(self, x: float, y: float) -> float:
        """Box-aware marking/pressure radius at an action-LTR anchor location."""
        return self.proximity_inside_box_m if _is_inside_attacked_box(x, y) else self.proximity_outside_box_m
