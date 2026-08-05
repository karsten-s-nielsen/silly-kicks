"""SB360 verdicts -- space family.

Observations and applicability classes are TRANSCRIBED FROM EXECUTION; only a
human writes an adjudication or a rationale.
"""

from __future__ import annotations

import silly_kicks.tracking as T
from tests.sb360 import _calls as C
from tests.sb360._registry import ADAPTERS, AxisVerdict, _entry

_entry(
    "add_cover_shadows",
    ADAPTERS["add_cover_shadows"](T.add_cover_shadows),
    columns=(
        "n_blocked_receivers",
        "n_potential_receivers",
        "blocking_score",
        "blocked_threat_fraction",
        "max_single_defender_blocking_score",
        "max_single_defender_player_id",
    ),
    velocity={
        "n_blocked_receivers": AxisVerdict("all_nan", "honest_nan"),
        "n_potential_receivers": AxisVerdict("all_nan", "honest_nan"),
        "blocking_score": AxisVerdict("all_nan", "honest_nan"),
        "blocked_threat_fraction": AxisVerdict("all_nan", "honest_nan"),
        "max_single_defender_blocking_score": AxisVerdict("all_nan", "honest_nan"),
        "max_single_defender_player_id": AxisVerdict(
            "no_signal",
            "not_exercised",
            rationale=(
                "The fixture does not produce this column's domain on either leg (no pressing sequence, shot- "
                "occurrence context, or blocking defender to score). A fixture inadequacy, not a library property "
                "-- widening the fixture would move it. [measured cause=velocity+frame_count]"
            ),
        ),
    },
    visibility={
        "gk_absent": {
            "n_blocked_receivers": AxisVerdict("all_nan", "honest_nan"),
            "n_potential_receivers": AxisVerdict("all_nan", "honest_nan"),
            "blocking_score": AxisVerdict("all_nan", "honest_nan"),
            "blocked_threat_fraction": AxisVerdict("all_nan", "honest_nan"),
            "max_single_defender_blocking_score": AxisVerdict("all_nan", "honest_nan"),
            "max_single_defender_player_id": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "The fixture does not produce this column's domain on either leg (no pressing sequence, shot- "
                    "occurrence context, or blocking defender to score). A fixture inadequacy, not a library "
                    "property -- widening the fixture would move it. [measured cause=velocity+frame_count]"
                ),
            ),
        },
        "defender_absent": {
            "n_blocked_receivers": AxisVerdict("all_nan", "honest_nan"),
            "n_potential_receivers": AxisVerdict("all_nan", "honest_nan"),
            "blocking_score": AxisVerdict("all_nan", "honest_nan"),
            "blocked_threat_fraction": AxisVerdict("all_nan", "honest_nan"),
            "max_single_defender_blocking_score": AxisVerdict("all_nan", "honest_nan"),
            "max_single_defender_player_id": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "The fixture does not produce this column's domain on either leg (no pressing sequence, shot- "
                    "occurrence context, or blocking defender to score). A fixture inadequacy, not a library "
                    "property -- widening the fixture would move it. [measured cause=velocity+frame_count]"
                ),
            ),
        },
    },
    applicability={
        "n_blocked_receivers": "no_support",
        "n_potential_receivers": "no_support",
        "blocking_score": "no_support",
        "blocked_threat_fraction": "no_support",
        "max_single_defender_blocking_score": "no_support",
        "max_single_defender_player_id": "no_support",
    },
    applicability_deltas={
        "n_blocked_receivers": {"extreme": 0.0, "near": 0.0},
        "n_potential_receivers": {"extreme": 0.0, "near": 0.0},
        "blocking_score": {"extreme": 0.0, "near": 0.0},
        "blocked_threat_fraction": {"extreme": 0.0, "near": 0.0},
        "max_single_defender_blocking_score": {"extreme": 0.0, "near": 0.0},
        "max_single_defender_player_id": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_das",
    C.generic(T.add_das),
    columns=(
        "das_team",
        "das_opponent",
        "das_diff",
        "das_source",
    ),
    velocity={
        "das_team": AxisVerdict("all_nan", "honest_nan"),
        "das_opponent": AxisVerdict("all_nan", "honest_nan"),
        "das_diff": AxisVerdict("all_nan", "honest_nan"),
        "das_source": AxisVerdict(
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
            "das_team": AxisVerdict("all_nan", "honest_nan"),
            "das_opponent": AxisVerdict("all_nan", "honest_nan"),
            "das_diff": AxisVerdict("all_nan", "honest_nan"),
            "das_source": AxisVerdict(
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
            "das_team": AxisVerdict("all_nan", "honest_nan"),
            "das_opponent": AxisVerdict("all_nan", "honest_nan"),
            "das_diff": AxisVerdict("all_nan", "honest_nan"),
            "das_source": AxisVerdict(
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
        "das_team": "no_support",
        "das_opponent": "no_support",
        "das_diff": "no_support",
        "das_source": "no_support",
    },
    applicability_deltas={
        "das_team": {"extreme": 0.0, "near": 0.0},
        "das_opponent": {"extreme": 0.0, "near": 0.0},
        "das_diff": {"extreme": 0.0, "near": 0.0},
        "das_source": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_obso",
    ADAPTERS["add_obso"](T.add_obso),
    columns=(
        "obso_actual",
        "obso_peak",
        "obso_optimal",
        "obso_epv_source",
    ),
    velocity={
        "obso_actual": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than the "
                "velocity-informed one, but a coherent quantity rather than an invented one. A consumer should "
                "know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
            ),
        ),
        "obso_peak": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than the "
                "velocity-informed one, but a coherent quantity rather than an invented one. A consumer should "
                "know the value is positional-only; it is not a fabrication. [measured "
                "cause=velocity+frame_count]"
            ),
        ),
        "obso_optimal": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than the "
                "velocity-informed one, but a coherent quantity rather than an invented one. A consumer should "
                "know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
            ),
        ),
        "obso_epv_source": AxisVerdict("identical", "works"),
    },
    visibility={
        "gk_absent": {
            "obso_actual": AxisVerdict("identical", "works"),
            "obso_peak": AxisVerdict("identical", "works"),
            "obso_optimal": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
                ),
            ),
            "obso_epv_source": AxisVerdict("identical", "works"),
        },
        "defender_absent": {
            "obso_actual": AxisVerdict("identical", "works"),
            "obso_peak": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured "
                    "cause=velocity+frame_count]"
                ),
            ),
            "obso_optimal": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
                ),
            ),
            "obso_epv_source": AxisVerdict("identical", "works"),
        },
    },
    applicability={
        "obso_actual": "no_support",
        "obso_peak": "no_support",
        "obso_optimal": "no_support",
        "obso_epv_source": "no_support",
    },
    applicability_deltas={
        "obso_actual": {"extreme": 0.0, "near": 0.0},
        "obso_peak": {"extreme": 0.0, "near": 0.0},
        "obso_optimal": {"extreme": 0.0, "near": 0.0},
        "obso_epv_source": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_pausa",
    ADAPTERS["add_pausa"](T.add_pausa),
    columns=(
        "obso_actual",
        "obso_peak",
        "obso_optimal",
        "obso_epv_source",
        "pausa_temporal",
        "pausa_spatial",
        "pausa_composite",
    ),
    velocity={
        "obso_actual": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than the "
                "velocity-informed one, but a coherent quantity rather than an invented one. A consumer should "
                "know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
            ),
        ),
        "obso_peak": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than the "
                "velocity-informed one, but a coherent quantity rather than an invented one. A consumer should "
                "know the value is positional-only; it is not a fabrication. [measured "
                "cause=velocity+frame_count]"
            ),
        ),
        "obso_optimal": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than the "
                "velocity-informed one, but a coherent quantity rather than an invented one. A consumer should "
                "know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
            ),
        ),
        "obso_epv_source": AxisVerdict("identical", "works"),
        "pausa_temporal": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Cause isolated as frame_count, not velocity: the feature needs a temporal window and a single "
                "freeze-frame legitimately yields a different, single-sample answer. Nothing is fabricated from "
                "absent kinematics. [measured cause=frame_count]"
            ),
        ),
        "pausa_spatial": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than the "
                "velocity-informed one, but a coherent quantity rather than an invented one. A consumer should "
                "know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
            ),
        ),
        "pausa_composite": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than the "
                "velocity-informed one, but a coherent quantity rather than an invented one. A consumer should "
                "know the value is positional-only; it is not a fabrication. [measured "
                "cause=velocity+frame_count]"
            ),
        ),
    },
    visibility={
        "gk_absent": {
            "obso_actual": AxisVerdict("identical", "works"),
            "obso_peak": AxisVerdict("identical", "works"),
            "obso_optimal": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
                ),
            ),
            "obso_epv_source": AxisVerdict("identical", "works"),
            "pausa_temporal": AxisVerdict("identical", "works"),
            "pausa_spatial": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
                ),
            ),
            "pausa_composite": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured "
                    "cause=velocity+frame_count]"
                ),
            ),
        },
        "defender_absent": {
            "obso_actual": AxisVerdict("identical", "works"),
            "obso_peak": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured "
                    "cause=velocity+frame_count]"
                ),
            ),
            "obso_optimal": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
                ),
            ),
            "obso_epv_source": AxisVerdict("identical", "works"),
            "pausa_temporal": AxisVerdict("identical", "works"),
            "pausa_spatial": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
                ),
            ),
            "pausa_composite": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured "
                    "cause=velocity+frame_count]"
                ),
            ),
        },
    },
    applicability={
        "obso_actual": "no_support",
        "obso_peak": "no_support",
        "obso_optimal": "no_support",
        "obso_epv_source": "no_support",
        "pausa_temporal": "no_support",
        "pausa_spatial": "no_support",
        "pausa_composite": "no_support",
    },
    applicability_deltas={
        "obso_actual": {"extreme": 0.0, "near": 0.0},
        "obso_peak": {"extreme": 0.0, "near": 0.0},
        "obso_optimal": {"extreme": 0.0, "near": 0.0},
        "obso_epv_source": {"extreme": 0.0, "near": 0.0},
        "pausa_temporal": {"extreme": 0.0, "near": 0.0},
        "pausa_spatial": {"extreme": 0.0, "near": 0.0},
        "pausa_composite": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_pitch_control",
    C.generic(T.add_pitch_control),
    columns=("pitch_control_at_target__spearman",),
    velocity={
        "pitch_control_at_target__spearman": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than the "
                "velocity-informed one, but a coherent quantity rather than an invented one. A consumer should "
                "know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
            ),
        ),
    },
    visibility={
        "gk_absent": {
            "pitch_control_at_target__spearman": AxisVerdict("identical", "works"),
        },
        "defender_absent": {
            "pitch_control_at_target__spearman": AxisVerdict("identical", "works"),
        },
    },
    applicability={
        "pitch_control_at_target__spearman": "no_support",
    },
    applicability_deltas={
        "pitch_control_at_target__spearman": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_player_influence",
    ADAPTERS["add_player_influence"](T.add_player_influence),
    columns=(
        "actor_reachable_area_m2",
        "off_ball_xt_team",
        "off_ball_xt_opponent",
        "off_ball_xt_diff",
        "reachable_area_team",
        "reachable_area_opponent",
        "reachable_area_diff",
    ),
    velocity={
        "actor_reachable_area_m2": AxisVerdict("all_nan", "honest_nan"),
        "off_ball_xt_team": AxisVerdict("all_nan", "honest_nan"),
        "off_ball_xt_opponent": AxisVerdict("all_nan", "honest_nan"),
        "off_ball_xt_diff": AxisVerdict("all_nan", "honest_nan"),
        "reachable_area_team": AxisVerdict("all_nan", "honest_nan"),
        "reachable_area_opponent": AxisVerdict("all_nan", "honest_nan"),
        "reachable_area_diff": AxisVerdict("all_nan", "honest_nan"),
    },
    visibility={
        "gk_absent": {
            "actor_reachable_area_m2": AxisVerdict("all_nan", "honest_nan"),
            "off_ball_xt_team": AxisVerdict("all_nan", "honest_nan"),
            "off_ball_xt_opponent": AxisVerdict("all_nan", "honest_nan"),
            "off_ball_xt_diff": AxisVerdict("all_nan", "honest_nan"),
            "reachable_area_team": AxisVerdict("all_nan", "honest_nan"),
            "reachable_area_opponent": AxisVerdict("all_nan", "honest_nan"),
            "reachable_area_diff": AxisVerdict("all_nan", "honest_nan"),
        },
        "defender_absent": {
            "actor_reachable_area_m2": AxisVerdict("all_nan", "honest_nan"),
            "off_ball_xt_team": AxisVerdict("all_nan", "honest_nan"),
            "off_ball_xt_opponent": AxisVerdict("all_nan", "honest_nan"),
            "off_ball_xt_diff": AxisVerdict("all_nan", "honest_nan"),
            "reachable_area_team": AxisVerdict("all_nan", "honest_nan"),
            "reachable_area_opponent": AxisVerdict("all_nan", "honest_nan"),
            "reachable_area_diff": AxisVerdict("all_nan", "honest_nan"),
        },
    },
    applicability={
        "actor_reachable_area_m2": "no_support",
        "off_ball_xt_team": "no_support",
        "off_ball_xt_opponent": "no_support",
        "off_ball_xt_diff": "no_support",
        "reachable_area_team": "no_support",
        "reachable_area_opponent": "no_support",
        "reachable_area_diff": "no_support",
    },
    applicability_deltas={
        "actor_reachable_area_m2": {"extreme": 0.0, "near": 0.0},
        "off_ball_xt_team": {"extreme": 0.0, "near": 0.0},
        "off_ball_xt_opponent": {"extreme": 0.0, "near": 0.0},
        "off_ball_xt_diff": {"extreme": 0.0, "near": 0.0},
        "reachable_area_team": {"extreme": 0.0, "near": 0.0},
        "reachable_area_opponent": {"extreme": 0.0, "near": 0.0},
        "reachable_area_diff": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_space_creation",
    ADAPTERS["add_space_creation"](T.add_space_creation),
    columns=(
        "space_created_m2",
        "space_denied_m2_opponent",
        "obso_epv_source",
    ),
    velocity={
        "space_created_m2": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than the "
                "velocity-informed one, but a coherent quantity rather than an invented one. A consumer should "
                "know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
            ),
        ),
        "space_denied_m2_opponent": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than the "
                "velocity-informed one, but a coherent quantity rather than an invented one. A consumer should "
                "know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
            ),
        ),
        "obso_epv_source": AxisVerdict("identical", "works"),
    },
    visibility={
        "gk_absent": {
            "space_created_m2": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
                ),
            ),
            "space_denied_m2_opponent": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
                ),
            ),
            "obso_epv_source": AxisVerdict("identical", "works"),
        },
        "defender_absent": {
            "space_created_m2": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
                ),
            ),
            "space_denied_m2_opponent": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
                ),
            ),
            "obso_epv_source": AxisVerdict("identical", "works"),
        },
    },
    applicability={
        "space_created_m2": "support_data_defined",
        "space_denied_m2_opponent": "support_data_defined",
        "obso_epv_source": "no_support",
    },
    applicability_deltas={
        "space_created_m2": {"extreme": 0.0003106691724212851, "near": 0.0007322852434015203},
        "space_denied_m2_opponent": {"extreme": 0.0002771696771546317, "near": 0.003391238345130887},
        "obso_epv_source": {"extreme": 0.0, "near": 0.0},
    },
)
