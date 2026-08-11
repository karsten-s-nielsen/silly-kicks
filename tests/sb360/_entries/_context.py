"""SB360 verdicts -- context family.

Observations and applicability classes are TRANSCRIBED FROM EXECUTION; only a
human writes an adjudication or a rationale.
"""

from __future__ import annotations

import silly_kicks.tracking as T
from tests.sb360 import _calls as C
from tests.sb360._registry import ADAPTERS, AxisVerdict, _entry

_entry(
    "add_action_context",
    C.generic(T.add_action_context),
    columns=(
        "nearest_defender_distance",
        "actor_speed",
        "receiver_zone_density",
        "defenders_in_triangle_to_goal",
    ),
    velocity={
        "nearest_defender_distance": AxisVerdict("identical", "works"),
        "actor_speed": AxisVerdict("all_nan", "honest_nan"),
        "receiver_zone_density": AxisVerdict("identical", "works"),
        "defenders_in_triangle_to_goal": AxisVerdict("identical", "works"),
    },
    visibility={
        "gk_absent": {
            "nearest_defender_distance": AxisVerdict("identical", "works"),
            "actor_speed": AxisVerdict("all_nan", "honest_nan"),
            "receiver_zone_density": AxisVerdict("identical", "works"),
            "defenders_in_triangle_to_goal": AxisVerdict("identical", "works"),
        },
        "defender_absent": {
            "nearest_defender_distance": AxisVerdict("identical", "works"),
            "actor_speed": AxisVerdict("all_nan", "honest_nan"),
            "receiver_zone_density": AxisVerdict("identical", "works"),
            "defenders_in_triangle_to_goal": AxisVerdict("identical", "works"),
        },
        "gk_one_end": {
            "nearest_defender_distance": AxisVerdict("identical", "works"),
            "actor_speed": AxisVerdict("all_nan", "honest_nan"),
            "receiver_zone_density": AxisVerdict("identical", "works"),
            "defenders_in_triangle_to_goal": AxisVerdict("identical", "works"),
        },
    },
    applicability={
        "nearest_defender_distance": "region_support",
        "actor_speed": "no_support",
        "receiver_zone_density": "no_support",
        "defenders_in_triangle_to_goal": "no_support",
    },
    applicability_deltas={
        "nearest_defender_distance": {"extreme": 0.0, "near": 28.098694739347096},
        "actor_speed": {"extreme": 0.0, "near": 0.0},
        "receiver_zone_density": {"extreme": 0.0, "near": 0.0},
        "defenders_in_triangle_to_goal": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_actor_pre_window",
    C.generic(T.add_actor_pre_window),
    columns=(
        "actor_arc_length_pre_window",
        "actor_displacement_pre_window",
    ),
    velocity={
        "actor_arc_length_pre_window": AxisVerdict(
            "partial_nan",
            "differs_by_design",
            rationale=(
                "Cause isolated as frame_count. On a freeze-frame the pre-window contains a single sample, so the "
                "metric is defined for some actions and not others; the NaNs are honest absences rather than "
                "fabricated values. [measured cause=frame_count]"
            ),
        ),
        "actor_displacement_pre_window": AxisVerdict(
            "partial_nan",
            "differs_by_design",
            rationale=(
                "Cause isolated as frame_count. On a freeze-frame the pre-window contains a single sample, so the "
                "metric is defined for some actions and not others; the NaNs are honest absences rather than "
                "fabricated values. [measured cause=frame_count]"
            ),
        ),
    },
    visibility={
        "gk_absent": {
            "actor_arc_length_pre_window": AxisVerdict("all_nan", "honest_nan"),
            "actor_displacement_pre_window": AxisVerdict("all_nan", "honest_nan"),
        },
        "defender_absent": {
            "actor_arc_length_pre_window": AxisVerdict(
                "partial_nan",
                "differs_by_design",
                rationale=(
                    "Cause isolated as frame_count. On a freeze-frame the pre-window contains a single sample, so "
                    "the metric is defined for some actions and not others; the NaNs are honest absences rather "
                    "than fabricated values. [measured cause=frame_count]"
                ),
            ),
            "actor_displacement_pre_window": AxisVerdict(
                "partial_nan",
                "differs_by_design",
                rationale=(
                    "Cause isolated as frame_count. On a freeze-frame the pre-window contains a single sample, so "
                    "the metric is defined for some actions and not others; the NaNs are honest absences rather "
                    "than fabricated values. [measured cause=frame_count]"
                ),
            ),
        },
        "gk_one_end": {
            "actor_arc_length_pre_window": AxisVerdict("all_nan", "honest_nan"),
            "actor_displacement_pre_window": AxisVerdict("all_nan", "honest_nan"),
        },
    },
    applicability={
        "actor_arc_length_pre_window": "support_data_defined",
        "actor_displacement_pre_window": "support_data_defined",
    },
    applicability_deltas={
        "actor_arc_length_pre_window": {"extreme": 3.844281922815732, "near": 0.0},
        "actor_displacement_pre_window": {"extreme": 3.844281922815725, "near": 0.0},
    },
)

_entry(
    "add_elastic_sync",
    C.generic(T.add_elastic_sync),
    columns=(
        "elastic_frame_id",
        "elastic_confidence",
        "elastic_error_seconds",
    ),
    velocity={
        "elastic_frame_id": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "A provenance column: its job is to report WHICH path produced the value, so reporting a "
                "different path on a freeze-frame leg than on a tracking leg is correct behaviour. ADR-043 "
                "designed das_source to do exactly this. [measured cause=velocity+frame_count]"
            ),
        ),
        "elastic_confidence": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Both legs compute this from inputs they actually hold -- nothing is imputed -- but they differ "
                "in BOTH velocity availability and temporal support, so the isolation probe could not attribute "
                "the change to one of them. The value is honest on each leg; what a consumer must not do is "
                "compare a freeze-frame number against a tracking number as though they were the same "
                "measurement. [measured cause=velocity+frame_count]"
            ),
        ),
        "elastic_error_seconds": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "A provenance column: its job is to report WHICH path produced the value, so reporting a "
                "different path on a freeze-frame leg than on a tracking leg is correct behaviour. ADR-043 "
                "designed das_source to do exactly this. [measured cause=velocity+frame_count]"
            ),
        ),
    },
    visibility={
        "gk_absent": {
            "elastic_frame_id": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "A provenance column: its job is to report WHICH path produced the value, so reporting a "
                    "different path on a freeze-frame leg than on a tracking leg is correct behaviour. ADR-043 "
                    "designed das_source to do exactly this. [measured cause=velocity+frame_count]"
                ),
            ),
            "elastic_confidence": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Both legs compute this from inputs they actually hold -- nothing is imputed -- but they "
                    "differ in BOTH velocity availability and temporal support, so the isolation probe could not "
                    "attribute the change to one of them. The value is honest on each leg; what a consumer must "
                    "not do is compare a freeze-frame number against a tracking number as though they were the "
                    "same measurement. [measured cause=velocity+frame_count]"
                ),
            ),
            "elastic_error_seconds": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "A provenance column: its job is to report WHICH path produced the value, so reporting a "
                    "different path on a freeze-frame leg than on a tracking leg is correct behaviour. ADR-043 "
                    "designed das_source to do exactly this. [measured cause=velocity+frame_count]"
                ),
            ),
        },
        "defender_absent": {
            "elastic_frame_id": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "A provenance column: its job is to report WHICH path produced the value, so reporting a "
                    "different path on a freeze-frame leg than on a tracking leg is correct behaviour. ADR-043 "
                    "designed das_source to do exactly this. [measured cause=velocity+frame_count]"
                ),
            ),
            "elastic_confidence": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Both legs compute this from inputs they actually hold -- nothing is imputed -- but they "
                    "differ in BOTH velocity availability and temporal support, so the isolation probe could not "
                    "attribute the change to one of them. The value is honest on each leg; what a consumer must "
                    "not do is compare a freeze-frame number against a tracking number as though they were the "
                    "same measurement. [measured cause=velocity+frame_count]"
                ),
            ),
            "elastic_error_seconds": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "A provenance column: its job is to report WHICH path produced the value, so reporting a "
                    "different path on a freeze-frame leg than on a tracking leg is correct behaviour. ADR-043 "
                    "designed das_source to do exactly this. [measured cause=velocity+frame_count]"
                ),
            ),
        },
        "gk_one_end": {
            "elastic_frame_id": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "A provenance column: its job is to report WHICH path produced the value, so reporting a "
                    "different path on a freeze-frame leg than on a tracking leg is correct behaviour. ADR-043 "
                    "designed das_source to do exactly this. [measured cause=velocity+frame_count]"
                ),
            ),
            "elastic_confidence": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Both legs compute this from inputs they actually hold -- nothing is imputed -- but they "
                    "differ in BOTH velocity availability and temporal support, so the isolation probe could not "
                    "attribute the change to one of them. The value is honest on each leg; what a consumer must "
                    "not do is compare a freeze-frame number against a tracking number as though they were the "
                    "same measurement. [measured cause=velocity+frame_count]"
                ),
            ),
            "elastic_error_seconds": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "A provenance column: its job is to report WHICH path produced the value, so reporting a "
                    "different path on a freeze-frame leg than on a tracking leg is correct behaviour. ADR-043 "
                    "designed das_source to do exactly this. [measured cause=velocity+frame_count]"
                ),
            ),
        },
    },
    applicability={
        "elastic_frame_id": "no_support",
        "elastic_confidence": "support_data_defined",
        "elastic_error_seconds": "no_support",
    },
    applicability_deltas={
        "elastic_frame_id": {"extreme": 0.0, "near": 0.0},
        "elastic_confidence": {"extreme": 0.00039999999999995595, "near": 0.1839000000000004},
        "elastic_error_seconds": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_gradientsports_player_ids",
    ADAPTERS["add_gradientsports_player_ids"](T.add_gradientsports_player_ids),
    columns=("gs_jersey_resolution_rate",),
    velocity={
        "gs_jersey_resolution_rate": AxisVerdict("identical", "works"),
    },
    visibility={
        "gk_absent": {
            "gs_jersey_resolution_rate": AxisVerdict("identical", "works"),
        },
        "defender_absent": {
            "gs_jersey_resolution_rate": AxisVerdict("identical", "works"),
        },
        "gk_one_end": {
            "gs_jersey_resolution_rate": AxisVerdict("identical", "works"),
        },
    },
    applicability={
        "gs_jersey_resolution_rate": "no_support",
    },
    applicability_deltas={
        "gs_jersey_resolution_rate": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_pressure_on_actor",
    C.generic(T.add_pressure_on_actor),
    columns=("pressure_on_actor__andrienko_oval",),
    velocity={
        "pressure_on_actor__andrienko_oval": AxisVerdict("identical", "works"),
    },
    visibility={
        "gk_absent": {
            "pressure_on_actor__andrienko_oval": AxisVerdict("identical", "works"),
        },
        "defender_absent": {
            "pressure_on_actor__andrienko_oval": AxisVerdict("identical", "works"),
        },
        "gk_one_end": {
            "pressure_on_actor__andrienko_oval": AxisVerdict("identical", "works"),
        },
    },
    applicability={
        "pressure_on_actor__andrienko_oval": "region_support",
    },
    applicability_deltas={
        "pressure_on_actor__andrienko_oval": {"extreme": 0.0, "near": 44.15914354723026},
    },
)

_entry(
    "add_sync_score",
    ADAPTERS["add_sync_score"](T.add_sync_score),
    columns=(
        "sync_score_min",
        "sync_score_mean",
        "sync_score_high_quality_frac",
    ),
    velocity={
        "sync_score_min": AxisVerdict("identical", "works"),
        "sync_score_mean": AxisVerdict("identical", "works"),
        "sync_score_high_quality_frac": AxisVerdict("identical", "works"),
    },
    visibility={
        "gk_absent": {
            "sync_score_min": AxisVerdict("identical", "works"),
            "sync_score_mean": AxisVerdict("identical", "works"),
            "sync_score_high_quality_frac": AxisVerdict("identical", "works"),
        },
        "defender_absent": {
            "sync_score_min": AxisVerdict("identical", "works"),
            "sync_score_mean": AxisVerdict("identical", "works"),
            "sync_score_high_quality_frac": AxisVerdict("identical", "works"),
        },
        "gk_one_end": {
            "sync_score_min": AxisVerdict("identical", "works"),
            "sync_score_mean": AxisVerdict("identical", "works"),
            "sync_score_high_quality_frac": AxisVerdict("identical", "works"),
        },
    },
    applicability={
        "sync_score_min": "no_support",
        "sync_score_mean": "no_support",
        "sync_score_high_quality_frac": "no_support",
    },
    applicability_deltas={
        "sync_score_min": {"extreme": 0.0, "near": 0.0},
        "sync_score_mean": {"extreme": 0.0, "near": 0.0},
        "sync_score_high_quality_frac": {"extreme": 0.0, "near": 0.0},
    },
)
