"""SB360 verdicts -- offball family.

Observations and applicability classes are TRANSCRIBED FROM EXECUTION; only a
human writes an adjudication or a rationale.
"""

from __future__ import annotations

import silly_kicks.tracking as T
from tests.sb360 import _calls as C
from tests.sb360._registry import ADAPTERS, AxisVerdict, _entry

_entry(
    "add_defensive_credit",
    ADAPTERS["add_defensive_credit"](T.add_defensive_credit),
    columns=(
        "audit_xg",
        "defensive_credit_net",
        "defensive_credit_plus",
        "defensive_credit_minus",
        "n_defensive_credits",
    ),
    velocity={
        "audit_xg": AxisVerdict("identical", "works"),
        "defensive_credit_net": AxisVerdict("identical", "works"),
        "defensive_credit_plus": AxisVerdict("identical", "works"),
        "defensive_credit_minus": AxisVerdict("identical", "works"),
        "n_defensive_credits": AxisVerdict("identical", "works"),
    },
    visibility={
        "gk_absent": {
            "audit_xg": AxisVerdict("identical", "works"),
            "defensive_credit_net": AxisVerdict("identical", "works"),
            "defensive_credit_plus": AxisVerdict("identical", "works"),
            "defensive_credit_minus": AxisVerdict("identical", "works"),
            "n_defensive_credits": AxisVerdict("identical", "works"),
        },
        "defender_absent": {
            "audit_xg": AxisVerdict("identical", "works"),
            "defensive_credit_net": AxisVerdict("identical", "works"),
            "defensive_credit_plus": AxisVerdict("identical", "works"),
            "defensive_credit_minus": AxisVerdict("identical", "works"),
            "n_defensive_credits": AxisVerdict("identical", "works"),
        },
        "gk_one_end": {
            "audit_xg": AxisVerdict("identical", "works"),
            "defensive_credit_net": AxisVerdict("identical", "works"),
            "defensive_credit_plus": AxisVerdict("identical", "works"),
            "defensive_credit_minus": AxisVerdict("identical", "works"),
            "n_defensive_credits": AxisVerdict("identical", "works"),
        },
    },
    applicability={
        "audit_xg": "no_support",
        "defensive_credit_net": "region_support",
        "defensive_credit_plus": "no_support",
        "defensive_credit_minus": "region_support",
        "n_defensive_credits": "region_support",
    },
    applicability_deltas={
        "audit_xg": {"extreme": 0.0, "near": 0.0},
        "defensive_credit_net": {"extreme": 0.0, "near": 0.12},
        "defensive_credit_plus": {"extreme": 0.0, "near": 0.0},
        "defensive_credit_minus": {"extreme": 0.0, "near": 0.12},
        "n_defensive_credits": {"extreme": 0.0, "near": 1.0},
    },
)

_entry(
    "add_off_ball_context",
    C.generic(T.add_off_ball_context),
    columns=(
        "n_off_ball_runners_pre_window",
        "max_off_ball_run_displacement_pre_window",
        "mean_off_ball_run_speed_pre_window",
        "n_off_ball_runners_toward_goal_pre_window",
        "line_break",
        "n_attackers_behind_line",
    ),
    velocity={
        "n_off_ball_runners_pre_window": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Cause isolated as frame_count, not velocity: the feature needs a temporal window and a single "
                "freeze-frame legitimately yields a different, single-sample answer. Nothing is fabricated from "
                "absent kinematics. [measured cause=frame_count]"
            ),
        ),
        "max_off_ball_run_displacement_pre_window": AxisVerdict("all_nan", "honest_nan"),
        "mean_off_ball_run_speed_pre_window": AxisVerdict("all_nan", "honest_nan"),
        "n_off_ball_runners_toward_goal_pre_window": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Cause isolated as frame_count, not velocity: the feature needs a temporal window and a single "
                "freeze-frame legitimately yields a different, single-sample answer. Nothing is fabricated from "
                "absent kinematics. [measured cause=frame_count]"
            ),
        ),
        "line_break": AxisVerdict("identical", "works"),
        "n_attackers_behind_line": AxisVerdict("identical", "works"),
    },
    visibility={
        "gk_absent": {
            "n_off_ball_runners_pre_window": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Cause isolated as frame_count, not velocity: the feature needs a temporal window and a "
                    "single freeze-frame legitimately yields a different, single-sample answer. Nothing is "
                    "fabricated from absent kinematics. [measured cause=frame_count]"
                ),
            ),
            "max_off_ball_run_displacement_pre_window": AxisVerdict("all_nan", "honest_nan"),
            "mean_off_ball_run_speed_pre_window": AxisVerdict("all_nan", "honest_nan"),
            "n_off_ball_runners_toward_goal_pre_window": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Cause isolated as frame_count, not velocity: the feature needs a temporal window and a "
                    "single freeze-frame legitimately yields a different, single-sample answer. Nothing is "
                    "fabricated from absent kinematics. [measured cause=frame_count]"
                ),
            ),
            "line_break": AxisVerdict("identical", "works"),
            "n_attackers_behind_line": AxisVerdict("identical", "works"),
        },
        "defender_absent": {
            "n_off_ball_runners_pre_window": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Cause isolated as frame_count, not velocity: the feature needs a temporal window and a "
                    "single freeze-frame legitimately yields a different, single-sample answer. Nothing is "
                    "fabricated from absent kinematics. [measured cause=frame_count]"
                ),
            ),
            "max_off_ball_run_displacement_pre_window": AxisVerdict("all_nan", "honest_nan"),
            "mean_off_ball_run_speed_pre_window": AxisVerdict("all_nan", "honest_nan"),
            "n_off_ball_runners_toward_goal_pre_window": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Cause isolated as frame_count, not velocity: the feature needs a temporal window and a "
                    "single freeze-frame legitimately yields a different, single-sample answer. Nothing is "
                    "fabricated from absent kinematics. [measured cause=frame_count]"
                ),
            ),
            "line_break": AxisVerdict("identical", "works"),
            "n_attackers_behind_line": AxisVerdict("identical", "works"),
        },
        "gk_one_end": {
            "n_off_ball_runners_pre_window": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Cause isolated as frame_count, not velocity: the feature needs a temporal window and a "
                    "single freeze-frame legitimately yields a different, single-sample answer. Nothing is "
                    "fabricated from absent kinematics. [measured cause=frame_count]"
                ),
            ),
            "max_off_ball_run_displacement_pre_window": AxisVerdict("all_nan", "honest_nan"),
            "mean_off_ball_run_speed_pre_window": AxisVerdict("all_nan", "honest_nan"),
            "n_off_ball_runners_toward_goal_pre_window": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Cause isolated as frame_count, not velocity: the feature needs a temporal window and a "
                    "single freeze-frame legitimately yields a different, single-sample answer. Nothing is "
                    "fabricated from absent kinematics. [measured cause=frame_count]"
                ),
            ),
            "line_break": AxisVerdict("identical", "works"),
            "n_attackers_behind_line": AxisVerdict("identical", "works"),
        },
    },
    applicability={
        "n_off_ball_runners_pre_window": "no_support",
        "max_off_ball_run_displacement_pre_window": "no_support",
        "mean_off_ball_run_speed_pre_window": "no_support",
        "n_off_ball_runners_toward_goal_pre_window": "no_support",
        "line_break": "no_support",
        "n_attackers_behind_line": "no_support",
    },
    applicability_deltas={
        "n_off_ball_runners_pre_window": {"extreme": 0.0, "near": 0.0},
        "max_off_ball_run_displacement_pre_window": {"extreme": 0.0, "near": 0.0},
        "mean_off_ball_run_speed_pre_window": {"extreme": 0.0, "near": 0.0},
        "n_off_ball_runners_toward_goal_pre_window": {"extreme": 0.0, "near": 0.0},
        "line_break": {"extreme": 0.0, "near": 0.0},
        "n_attackers_behind_line": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_off_ball_run_values",
    ADAPTERS["add_off_ball_run_values"](T.add_off_ball_run_values),
    columns=(
        "run_value_target",
        "run_value_disruptive_sum",
        "run_value_enabled_pass",
        "n_disruptive_runs",
        "n_valued_disruptive_runs",
    ),
    velocity={
        "run_value_target": AxisVerdict("identical", "works"),
        "run_value_disruptive_sum": AxisVerdict("identical", "works"),
        "run_value_enabled_pass": AxisVerdict("identical", "works"),
        "n_disruptive_runs": AxisVerdict("identical", "works"),
        "n_valued_disruptive_runs": AxisVerdict("identical", "works"),
    },
    visibility={
        "gk_absent": {
            "run_value_target": AxisVerdict("identical", "works"),
            "run_value_disruptive_sum": AxisVerdict("identical", "works"),
            "run_value_enabled_pass": AxisVerdict("identical", "works"),
            "n_disruptive_runs": AxisVerdict("identical", "works"),
            "n_valued_disruptive_runs": AxisVerdict("identical", "works"),
        },
        "defender_absent": {
            "run_value_target": AxisVerdict("identical", "works"),
            "run_value_disruptive_sum": AxisVerdict("identical", "works"),
            "run_value_enabled_pass": AxisVerdict("identical", "works"),
            "n_disruptive_runs": AxisVerdict("identical", "works"),
            "n_valued_disruptive_runs": AxisVerdict("identical", "works"),
        },
        "gk_one_end": {
            "run_value_target": AxisVerdict("identical", "works"),
            "run_value_disruptive_sum": AxisVerdict("identical", "works"),
            "run_value_enabled_pass": AxisVerdict("identical", "works"),
            "n_disruptive_runs": AxisVerdict("identical", "works"),
            "n_valued_disruptive_runs": AxisVerdict("identical", "works"),
        },
    },
    applicability={
        "run_value_target": "no_support",
        "run_value_disruptive_sum": "no_support",
        "run_value_enabled_pass": "no_support",
        "n_disruptive_runs": "no_support",
        "n_valued_disruptive_runs": "no_support",
    },
    applicability_deltas={
        "run_value_target": {"extreme": 0.0, "near": 0.0},
        "run_value_disruptive_sum": {"extreme": 0.0, "near": 0.0},
        "run_value_enabled_pass": {"extreme": 0.0, "near": 0.0},
        "n_disruptive_runs": {"extreme": 0.0, "near": 0.0},
        "n_valued_disruptive_runs": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_off_ball_runs",
    C.generic(T.add_off_ball_runs),
    columns=(
        "n_off_ball_runners_pre_window",
        "max_off_ball_run_displacement_pre_window",
        "mean_off_ball_run_speed_pre_window",
        "n_off_ball_runners_toward_goal_pre_window",
    ),
    velocity={
        "n_off_ball_runners_pre_window": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Cause isolated as frame_count, not velocity: the feature needs a temporal window and a single "
                "freeze-frame legitimately yields a different, single-sample answer. Nothing is fabricated from "
                "absent kinematics. [measured cause=frame_count]"
            ),
        ),
        "max_off_ball_run_displacement_pre_window": AxisVerdict("all_nan", "honest_nan"),
        "mean_off_ball_run_speed_pre_window": AxisVerdict("all_nan", "honest_nan"),
        "n_off_ball_runners_toward_goal_pre_window": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Cause isolated as frame_count, not velocity: the feature needs a temporal window and a single "
                "freeze-frame legitimately yields a different, single-sample answer. Nothing is fabricated from "
                "absent kinematics. [measured cause=frame_count]"
            ),
        ),
    },
    visibility={
        "gk_absent": {
            "n_off_ball_runners_pre_window": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Cause isolated as frame_count, not velocity: the feature needs a temporal window and a "
                    "single freeze-frame legitimately yields a different, single-sample answer. Nothing is "
                    "fabricated from absent kinematics. [measured cause=frame_count]"
                ),
            ),
            "max_off_ball_run_displacement_pre_window": AxisVerdict("all_nan", "honest_nan"),
            "mean_off_ball_run_speed_pre_window": AxisVerdict("all_nan", "honest_nan"),
            "n_off_ball_runners_toward_goal_pre_window": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Cause isolated as frame_count, not velocity: the feature needs a temporal window and a "
                    "single freeze-frame legitimately yields a different, single-sample answer. Nothing is "
                    "fabricated from absent kinematics. [measured cause=frame_count]"
                ),
            ),
        },
        "defender_absent": {
            "n_off_ball_runners_pre_window": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Cause isolated as frame_count, not velocity: the feature needs a temporal window and a "
                    "single freeze-frame legitimately yields a different, single-sample answer. Nothing is "
                    "fabricated from absent kinematics. [measured cause=frame_count]"
                ),
            ),
            "max_off_ball_run_displacement_pre_window": AxisVerdict("all_nan", "honest_nan"),
            "mean_off_ball_run_speed_pre_window": AxisVerdict("all_nan", "honest_nan"),
            "n_off_ball_runners_toward_goal_pre_window": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Cause isolated as frame_count, not velocity: the feature needs a temporal window and a "
                    "single freeze-frame legitimately yields a different, single-sample answer. Nothing is "
                    "fabricated from absent kinematics. [measured cause=frame_count]"
                ),
            ),
        },
        "gk_one_end": {
            "n_off_ball_runners_pre_window": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Cause isolated as frame_count, not velocity: the feature needs a temporal window and a "
                    "single freeze-frame legitimately yields a different, single-sample answer. Nothing is "
                    "fabricated from absent kinematics. [measured cause=frame_count]"
                ),
            ),
            "max_off_ball_run_displacement_pre_window": AxisVerdict("all_nan", "honest_nan"),
            "mean_off_ball_run_speed_pre_window": AxisVerdict("all_nan", "honest_nan"),
            "n_off_ball_runners_toward_goal_pre_window": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Cause isolated as frame_count, not velocity: the feature needs a temporal window and a "
                    "single freeze-frame legitimately yields a different, single-sample answer. Nothing is "
                    "fabricated from absent kinematics. [measured cause=frame_count]"
                ),
            ),
        },
    },
    applicability={
        "n_off_ball_runners_pre_window": "no_support",
        "max_off_ball_run_displacement_pre_window": "no_support",
        "mean_off_ball_run_speed_pre_window": "no_support",
        "n_off_ball_runners_toward_goal_pre_window": "no_support",
    },
    applicability_deltas={
        "n_off_ball_runners_pre_window": {"extreme": 0.0, "near": 0.0},
        "max_off_ball_run_displacement_pre_window": {"extreme": 0.0, "near": 0.0},
        "mean_off_ball_run_speed_pre_window": {"extreme": 0.0, "near": 0.0},
        "n_off_ball_runners_toward_goal_pre_window": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_press_commitment",
    C.generic(T.add_press_commitment),
    columns=(
        "press_commitment",
        "press_commitment_closing_speed",
        "press_commitment_source",
    ),
    velocity={
        "press_commitment": AxisVerdict(
            "no_signal",
            "not_exercised",
            rationale=(
                "The fixture does not produce this column's domain on either leg (no pressing sequence, shot- "
                "occurrence context, or blocking defender to score). A fixture inadequacy, not a library property "
                "-- widening the fixture would move it. [measured cause=velocity+frame_count]"
            ),
        ),
        "press_commitment_closing_speed": AxisVerdict(
            "no_signal",
            "not_exercised",
            rationale=(
                "The fixture does not produce this column's domain on either leg (no pressing sequence, shot- "
                "occurrence context, or blocking defender to score). A fixture inadequacy, not a library property "
                "-- widening the fixture would move it. [measured cause=velocity+frame_count]"
            ),
        ),
        "press_commitment_source": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "A provenance column: its job is to report WHICH path produced the value, so reporting a "
                "different path on a freeze-frame leg than on a tracking leg is correct behaviour. ADR-043 "
                "designed das_source to do exactly this. [measured cause=velocity]"
            ),
        ),
    },
    visibility={
        "gk_absent": {
            "press_commitment": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "The fixture does not produce this column's domain on either leg (no pressing sequence, shot- "
                    "occurrence context, or blocking defender to score). A fixture inadequacy, not a library "
                    "property -- widening the fixture would move it. [measured cause=velocity+frame_count]"
                ),
            ),
            "press_commitment_closing_speed": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "The fixture does not produce this column's domain on either leg (no pressing sequence, shot- "
                    "occurrence context, or blocking defender to score). A fixture inadequacy, not a library "
                    "property -- widening the fixture would move it. [measured cause=velocity+frame_count]"
                ),
            ),
            "press_commitment_source": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "A provenance column: its job is to report WHICH path produced the value, so reporting a "
                    "different path on a freeze-frame leg than on a tracking leg is correct behaviour. ADR-043 "
                    "designed das_source to do exactly this. [measured cause=velocity]"
                ),
            ),
        },
        "defender_absent": {
            "press_commitment": AxisVerdict("all_nan", "honest_nan"),
            "press_commitment_closing_speed": AxisVerdict("all_nan", "honest_nan"),
            "press_commitment_source": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "A provenance column: its job is to report WHICH path produced the value, so reporting a "
                    "different path on a freeze-frame leg than on a tracking leg is correct behaviour. ADR-043 "
                    "designed das_source to do exactly this. [measured cause=velocity]"
                ),
            ),
        },
        "gk_one_end": {
            "press_commitment": AxisVerdict("all_nan", "honest_nan"),
            "press_commitment_closing_speed": AxisVerdict("all_nan", "honest_nan"),
            "press_commitment_source": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "A provenance column: its job is to report WHICH path produced the value, so reporting a "
                    "different path on a freeze-frame leg than on a tracking leg is correct behaviour. ADR-043 "
                    "designed das_source to do exactly this. [measured cause=velocity]"
                ),
            ),
        },
    },
    applicability={
        "press_commitment": "no_support",
        "press_commitment_closing_speed": "no_support",
        "press_commitment_source": "no_support",
    },
    applicability_deltas={
        "press_commitment": {"extreme": 0.0, "near": 0.0},
        "press_commitment_closing_speed": {"extreme": 0.0, "near": 0.0},
        "press_commitment_source": {"extreme": 0.0, "near": 0.0},
    },
)
