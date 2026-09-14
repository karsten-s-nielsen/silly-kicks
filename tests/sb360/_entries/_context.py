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

_ELASTIC_SB360_START_COLS = (
    "elastic_frame_id",
    "elastic_confidence",
    "elastic_error_seconds",
)
_ELASTIC_SB360_RECV_COLS = (
    "elastic_receive_frame_id",
    "elastic_receive_confidence",
    "elastic_receive_error_seconds",
)
_ELASTIC_SB360_COLS = _ELASTIC_SB360_START_COLS + _ELASTIC_SB360_RECV_COLS
_ELASTIC_HONEST_NAN_RATIONALE = (
    "ELASTIC-NW is continuous-only: candidate detection needs a dense ball trajectory. The freeze-frame "
    "Leg A (POSITIONAL_ONLY, no trajectory) returns an honest empty alignment (all-NaN) -- "
    "align_events_to_frames declines freeze-frame input rather than fabricate a sync from disconnected "
    "snapshots. The synthetic full-tracking Leg B aligns this event; the leg difference is a data-regime "
    "boundary, not a library defect."
)


def _el_hn() -> AxisVerdict:
    return AxisVerdict("all_nan", "honest_nan", rationale=_ELASTIC_HONEST_NAN_RATIONALE)


_entry(
    "add_elastic_sync",
    C.generic(T.add_elastic_sync),
    columns=_ELASTIC_SB360_COLS,
    velocity={c: _el_hn() for c in _ELASTIC_SB360_COLS},
    visibility={
        # Every elastic column is all-NaN on the freeze-frame Leg A (POSITIONAL_ONLY -> empty
        # alignment). The synthetic full-tracking Leg B aligns both the start AND the reception on
        # every roster, so every column is all_nan/honest_nan (the leg difference is a data-regime
        # boundary, not a defect). Reception included: with the central-difference accel the
        # candidate-detection now surfaces a reception touch on Leg B even on the reduced rosters.
        "gk_absent": {c: _el_hn() for c in _ELASTIC_SB360_COLS},
        "defender_absent": {c: _el_hn() for c in _ELASTIC_SB360_COLS},
        "gk_one_end": {c: _el_hn() for c in _ELASTIC_SB360_COLS},
    },
    applicability={c: "no_support" for c in _ELASTIC_SB360_COLS},
    applicability_deltas={c: {"extreme": 0.0, "near": 0.0} for c in _ELASTIC_SB360_COLS},
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
    ADAPTERS["add_pressure_on_actor"](T.add_pressure_on_actor),
    columns=(
        "pressure_on_actor__andrienko_oval",
        "pressure_on_actor__bekkers_pi",
    ),
    velocity={
        "pressure_on_actor__andrienko_oval": AxisVerdict("identical", "works"),
        "pressure_on_actor__bekkers_pi": AxisVerdict(
            "all_nan",
            "honest_nan",
            rationale=(
                "bekkers_pi is velocity-derived (probabilistic TTI + a velocity-GATED active-pressing "
                "speed_threshold filter). Its zero-velocity form is artifact-dependent, not a smooth "
                "limit -- so it SUPPRESSES to honest-NaN on the velocity-less leg (Tier-3, ADR-063 "
                "amendment / spec Part 4), while the velocity leg scores. This is the audit-observable "
                "counterpart to xShot/space_creation (which the two-leg fixture cannot exercise). "
                "[measured cause=velocity]"
            ),
        ),
    },
    visibility={
        "gk_absent": {
            "pressure_on_actor__andrienko_oval": AxisVerdict("identical", "works"),
            "pressure_on_actor__bekkers_pi": AxisVerdict("all_nan", "honest_nan"),
        },
        "defender_absent": {
            "pressure_on_actor__andrienko_oval": AxisVerdict("identical", "works"),
            "pressure_on_actor__bekkers_pi": AxisVerdict("all_nan", "honest_nan"),
        },
        "gk_one_end": {
            "pressure_on_actor__andrienko_oval": AxisVerdict("identical", "works"),
            "pressure_on_actor__bekkers_pi": AxisVerdict("all_nan", "honest_nan"),
        },
    },
    applicability={
        "pressure_on_actor__andrienko_oval": "region_support",
        "pressure_on_actor__bekkers_pi": "no_support",
    },
    applicability_deltas={
        "pressure_on_actor__andrienko_oval": {"extreme": 0.0, "near": 44.15914354723026},
        "pressure_on_actor__bekkers_pi": {"extreme": 0.0, "near": 0.0},
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
