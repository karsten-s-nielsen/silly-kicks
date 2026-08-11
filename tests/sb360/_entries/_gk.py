"""SB360 verdicts -- gk family.

Observations and applicability classes are TRANSCRIBED FROM EXECUTION; only a
human writes an adjudication or a rationale.
"""

from __future__ import annotations

import silly_kicks.tracking as T
from tests.sb360 import _calls as C
from tests.sb360._registry import ADAPTERS, AxisVerdict, _entry

_entry(
    "add_ghost_gk",
    C.generic(T.add_ghost_gk),
    columns=(
        "ghost_gk_x",
        "ghost_gk_y",
        "ghost_gk_source",
    ),
    velocity={
        "ghost_gk_x": AxisVerdict("all_nan", "honest_nan"),
        "ghost_gk_y": AxisVerdict("all_nan", "honest_nan"),
        "ghost_gk_source": AxisVerdict(
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
            "ghost_gk_x": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "By construction: the gk_absent roster removes the keeper, so this GK feature has nothing to "
                    "measure in EITHER leg. Recorded as unexercised because the vocabulary admits nothing else "
                    "from no_signal -- but the collapse IS the visibility finding: the column hard-depends on the "
                    "keeper being in the freeze-frame, which for SB360 means in the broadcast camera's view. "
                    "[measured cause=n/a]"
                ),
            ),
            "ghost_gk_y": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "By construction: the gk_absent roster removes the keeper, so this GK feature has nothing to "
                    "measure in EITHER leg. Recorded as unexercised because the vocabulary admits nothing else "
                    "from no_signal -- but the collapse IS the visibility finding: the column hard-depends on the "
                    "keeper being in the freeze-frame, which for SB360 means in the broadcast camera's view. "
                    "[measured cause=n/a]"
                ),
            ),
            "ghost_gk_source": AxisVerdict(
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
            "ghost_gk_x": AxisVerdict("all_nan", "honest_nan"),
            "ghost_gk_y": AxisVerdict("all_nan", "honest_nan"),
            "ghost_gk_source": AxisVerdict(
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
            "ghost_gk_x": AxisVerdict("all_nan", "honest_nan"),
            "ghost_gk_y": AxisVerdict("all_nan", "honest_nan"),
            "ghost_gk_source": AxisVerdict(
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
        "ghost_gk_x": "no_support",
        "ghost_gk_y": "no_support",
        "ghost_gk_source": "no_support",
    },
    applicability_deltas={
        "ghost_gk_x": {"extreme": 0.0, "near": 0.0},
        "ghost_gk_y": {"extreme": 0.0, "near": 0.0},
        "ghost_gk_source": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_gk_completion",
    C.generic(T.add_gk_completion),
    columns=("gk_completion",),
    velocity={
        "gk_completion": AxisVerdict("identical", "works"),
    },
    visibility={
        "gk_absent": {
            "gk_completion": AxisVerdict("identical", "works"),
        },
        "defender_absent": {
            "gk_completion": AxisVerdict("identical", "works"),
        },
        "gk_one_end": {
            "gk_completion": AxisVerdict("identical", "works"),
        },
    },
    applicability={
        "gk_completion": "no_support",
    },
    applicability_deltas={
        "gk_completion": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_gk_influence",
    ADAPTERS["add_gk_influence"](T.add_gk_influence),
    columns=(
        "gk_pitch_control_share_weighted",
        "gk_reachable_area_m2",
        "gk_closing_time_min_s__six_yard_box",
        "gk_closing_time_mean_s__six_yard_box",
    ),
    velocity={
        "gk_pitch_control_share_weighted": AxisVerdict("all_nan", "honest_nan"),
        "gk_reachable_area_m2": AxisVerdict("all_nan", "honest_nan"),
        "gk_closing_time_min_s__six_yard_box": AxisVerdict("all_nan", "honest_nan"),
        "gk_closing_time_mean_s__six_yard_box": AxisVerdict("all_nan", "honest_nan"),
    },
    visibility={
        "gk_absent": {
            "gk_pitch_control_share_weighted": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "By construction: the gk_absent roster removes the keeper, so this GK feature has nothing to "
                    "measure in EITHER leg. Recorded as unexercised because the vocabulary admits nothing else "
                    "from no_signal -- but the collapse IS the visibility finding: the column hard-depends on the "
                    "keeper being in the freeze-frame, which for SB360 means in the broadcast camera's view. "
                    "[measured cause=n/a]"
                ),
            ),
            "gk_reachable_area_m2": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "By construction: the gk_absent roster removes the keeper, so this GK feature has nothing to "
                    "measure in EITHER leg. Recorded as unexercised because the vocabulary admits nothing else "
                    "from no_signal -- but the collapse IS the visibility finding: the column hard-depends on the "
                    "keeper being in the freeze-frame, which for SB360 means in the broadcast camera's view. "
                    "[measured cause=n/a]"
                ),
            ),
            "gk_closing_time_min_s__six_yard_box": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "By construction: the gk_absent roster removes the keeper, so this GK feature has nothing to "
                    "measure in EITHER leg. Recorded as unexercised because the vocabulary admits nothing else "
                    "from no_signal -- but the collapse IS the visibility finding: the column hard-depends on the "
                    "keeper being in the freeze-frame, which for SB360 means in the broadcast camera's view. "
                    "[measured cause=n/a]"
                ),
            ),
            "gk_closing_time_mean_s__six_yard_box": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "By construction: the gk_absent roster removes the keeper, so this GK feature has nothing to "
                    "measure in EITHER leg. Recorded as unexercised because the vocabulary admits nothing else "
                    "from no_signal -- but the collapse IS the visibility finding: the column hard-depends on the "
                    "keeper being in the freeze-frame, which for SB360 means in the broadcast camera's view. "
                    "[measured cause=n/a]"
                ),
            ),
        },
        "defender_absent": {
            "gk_pitch_control_share_weighted": AxisVerdict("all_nan", "honest_nan"),
            "gk_reachable_area_m2": AxisVerdict("all_nan", "honest_nan"),
            "gk_closing_time_min_s__six_yard_box": AxisVerdict("all_nan", "honest_nan"),
            "gk_closing_time_mean_s__six_yard_box": AxisVerdict("all_nan", "honest_nan"),
        },
        "gk_one_end": {
            "gk_pitch_control_share_weighted": AxisVerdict("all_nan", "honest_nan"),
            "gk_reachable_area_m2": AxisVerdict("all_nan", "honest_nan"),
            "gk_closing_time_min_s__six_yard_box": AxisVerdict("all_nan", "honest_nan"),
            "gk_closing_time_mean_s__six_yard_box": AxisVerdict("all_nan", "honest_nan"),
        },
    },
    applicability={
        "gk_pitch_control_share_weighted": "no_support",
        "gk_reachable_area_m2": "no_support",
        "gk_closing_time_min_s__six_yard_box": "no_support",
        "gk_closing_time_mean_s__six_yard_box": "no_support",
    },
    applicability_deltas={
        "gk_pitch_control_share_weighted": {"extreme": 0.0, "near": 0.0},
        "gk_reachable_area_m2": {"extreme": 0.0, "near": 0.0},
        "gk_closing_time_min_s__six_yard_box": {"extreme": 0.0, "near": 0.0},
        "gk_closing_time_mean_s__six_yard_box": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_pre_shot_gk_angle",
    ADAPTERS["add_pre_shot_gk_angle"](T.add_pre_shot_gk_angle),
    columns=(
        "defending_gk_player_id",
        "pre_shot_gk_angle_to_shot_trajectory",
        "pre_shot_gk_angle_off_goal_line",
    ),
    velocity={
        "defending_gk_player_id": AxisVerdict("identical", "works"),
        "pre_shot_gk_angle_to_shot_trajectory": AxisVerdict("identical", "works"),
        "pre_shot_gk_angle_off_goal_line": AxisVerdict("identical", "works"),
    },
    visibility={
        "gk_absent": {
            "defending_gk_player_id": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "By construction: the gk_absent roster removes the keeper, so this GK feature has nothing to "
                    "measure in EITHER leg. Recorded as unexercised because the vocabulary admits nothing else "
                    "from no_signal -- but the collapse IS the visibility finding: the column hard-depends on the "
                    "keeper being in the freeze-frame, which for SB360 means in the broadcast camera's view. "
                    "[measured cause=n/a]"
                ),
            ),
            "pre_shot_gk_angle_to_shot_trajectory": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "By construction: the gk_absent roster removes the keeper, so this GK feature has nothing to "
                    "measure in EITHER leg. Recorded as unexercised because the vocabulary admits nothing else "
                    "from no_signal -- but the collapse IS the visibility finding: the column hard-depends on the "
                    "keeper being in the freeze-frame, which for SB360 means in the broadcast camera's view. "
                    "[measured cause=n/a]"
                ),
            ),
            "pre_shot_gk_angle_off_goal_line": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "By construction: the gk_absent roster removes the keeper, so this GK feature has nothing to "
                    "measure in EITHER leg. Recorded as unexercised because the vocabulary admits nothing else "
                    "from no_signal -- but the collapse IS the visibility finding: the column hard-depends on the "
                    "keeper being in the freeze-frame, which for SB360 means in the broadcast camera's view. "
                    "[measured cause=n/a]"
                ),
            ),
        },
        "defender_absent": {
            "defending_gk_player_id": AxisVerdict("identical", "works"),
            "pre_shot_gk_angle_to_shot_trajectory": AxisVerdict("identical", "works"),
            "pre_shot_gk_angle_off_goal_line": AxisVerdict("identical", "works"),
        },
        "gk_one_end": {
            "defending_gk_player_id": AxisVerdict("identical", "works"),
            "pre_shot_gk_angle_to_shot_trajectory": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "The fixture does not produce this column's domain on either leg (no pressing sequence, shot- "
                    "occurrence context, or blocking defender to score). A fixture inadequacy, not a library "
                    "property -- widening the fixture would move it. [measured cause=n/a]"
                ),
            ),
            "pre_shot_gk_angle_off_goal_line": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "The fixture does not produce this column's domain on either leg (no pressing sequence, shot- "
                    "occurrence context, or blocking defender to score). A fixture inadequacy, not a library "
                    "property -- widening the fixture would move it. [measured cause=n/a]"
                ),
            ),
        },
    },
    applicability={
        "defending_gk_player_id": "no_support",
        "pre_shot_gk_angle_to_shot_trajectory": "no_support",
        "pre_shot_gk_angle_off_goal_line": "no_support",
    },
    applicability_deltas={
        "defending_gk_player_id": {"extreme": 0.0, "near": 0.0},
        "pre_shot_gk_angle_to_shot_trajectory": {"extreme": 0.0, "near": 0.0},
        "pre_shot_gk_angle_off_goal_line": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_pre_shot_gk_position",
    ADAPTERS["add_pre_shot_gk_position"](T.add_pre_shot_gk_position),
    columns=(
        "defending_gk_player_id",
        "pre_shot_gk_x",
        "pre_shot_gk_y",
        "pre_shot_gk_distance_to_goal",
        "pre_shot_gk_distance_to_shot",
    ),
    velocity={
        "defending_gk_player_id": AxisVerdict("identical", "works"),
        "pre_shot_gk_x": AxisVerdict("identical", "works"),
        "pre_shot_gk_y": AxisVerdict("identical", "works"),
        "pre_shot_gk_distance_to_goal": AxisVerdict("identical", "works"),
        "pre_shot_gk_distance_to_shot": AxisVerdict("identical", "works"),
    },
    visibility={
        "gk_absent": {
            "defending_gk_player_id": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "By construction: the gk_absent roster removes the keeper, so this GK feature has nothing to "
                    "measure in EITHER leg. Recorded as unexercised because the vocabulary admits nothing else "
                    "from no_signal -- but the collapse IS the visibility finding: the column hard-depends on the "
                    "keeper being in the freeze-frame, which for SB360 means in the broadcast camera's view. "
                    "[measured cause=n/a]"
                ),
            ),
            "pre_shot_gk_x": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "By construction: the gk_absent roster removes the keeper, so this GK feature has nothing to "
                    "measure in EITHER leg. Recorded as unexercised because the vocabulary admits nothing else "
                    "from no_signal -- but the collapse IS the visibility finding: the column hard-depends on the "
                    "keeper being in the freeze-frame, which for SB360 means in the broadcast camera's view. "
                    "[measured cause=n/a]"
                ),
            ),
            "pre_shot_gk_y": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "By construction: the gk_absent roster removes the keeper, so this GK feature has nothing to "
                    "measure in EITHER leg. Recorded as unexercised because the vocabulary admits nothing else "
                    "from no_signal -- but the collapse IS the visibility finding: the column hard-depends on the "
                    "keeper being in the freeze-frame, which for SB360 means in the broadcast camera's view. "
                    "[measured cause=n/a]"
                ),
            ),
            "pre_shot_gk_distance_to_goal": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "By construction: the gk_absent roster removes the keeper, so this GK feature has nothing to "
                    "measure in EITHER leg. Recorded as unexercised because the vocabulary admits nothing else "
                    "from no_signal -- but the collapse IS the visibility finding: the column hard-depends on the "
                    "keeper being in the freeze-frame, which for SB360 means in the broadcast camera's view. "
                    "[measured cause=n/a]"
                ),
            ),
            "pre_shot_gk_distance_to_shot": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "By construction: the gk_absent roster removes the keeper, so this GK feature has nothing to "
                    "measure in EITHER leg. Recorded as unexercised because the vocabulary admits nothing else "
                    "from no_signal -- but the collapse IS the visibility finding: the column hard-depends on the "
                    "keeper being in the freeze-frame, which for SB360 means in the broadcast camera's view. "
                    "[measured cause=n/a]"
                ),
            ),
        },
        "defender_absent": {
            "defending_gk_player_id": AxisVerdict("identical", "works"),
            "pre_shot_gk_x": AxisVerdict("identical", "works"),
            "pre_shot_gk_y": AxisVerdict("identical", "works"),
            "pre_shot_gk_distance_to_goal": AxisVerdict("identical", "works"),
            "pre_shot_gk_distance_to_shot": AxisVerdict("identical", "works"),
        },
        "gk_one_end": {
            "defending_gk_player_id": AxisVerdict("identical", "works"),
            "pre_shot_gk_x": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "The fixture does not produce this column's domain on either leg (no pressing sequence, shot- "
                    "occurrence context, or blocking defender to score). A fixture inadequacy, not a library "
                    "property -- widening the fixture would move it. [measured cause=n/a]"
                ),
            ),
            "pre_shot_gk_y": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "The fixture does not produce this column's domain on either leg (no pressing sequence, shot- "
                    "occurrence context, or blocking defender to score). A fixture inadequacy, not a library "
                    "property -- widening the fixture would move it. [measured cause=n/a]"
                ),
            ),
            "pre_shot_gk_distance_to_goal": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "The fixture does not produce this column's domain on either leg (no pressing sequence, shot- "
                    "occurrence context, or blocking defender to score). A fixture inadequacy, not a library "
                    "property -- widening the fixture would move it. [measured cause=n/a]"
                ),
            ),
            "pre_shot_gk_distance_to_shot": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "The fixture does not produce this column's domain on either leg (no pressing sequence, shot- "
                    "occurrence context, or blocking defender to score). A fixture inadequacy, not a library "
                    "property -- widening the fixture would move it. [measured cause=n/a]"
                ),
            ),
        },
    },
    applicability={
        "defending_gk_player_id": "no_support",
        "pre_shot_gk_x": "no_support",
        "pre_shot_gk_y": "no_support",
        "pre_shot_gk_distance_to_goal": "no_support",
        "pre_shot_gk_distance_to_shot": "no_support",
    },
    applicability_deltas={
        "defending_gk_player_id": {"extreme": 0.0, "near": 0.0},
        "pre_shot_gk_x": {"extreme": 0.0, "near": 0.0},
        "pre_shot_gk_y": {"extreme": 0.0, "near": 0.0},
        "pre_shot_gk_distance_to_goal": {"extreme": 0.0, "near": 0.0},
        "pre_shot_gk_distance_to_shot": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_shot_goalmouth",
    C.generic(T.add_shot_goalmouth),
    columns=(
        "shot_crossing_y",
        "shot_crossing_z",
        "shot_speed",
        "shot_time_to_goal_line",
        "shot_on_target_derived",
        "shot_crossing_source",
        "shot_crossing_confidence",
        "shot_fit_n_frames",
        "shot_fit_rmse",
        "shot_fit_end_reason",
        "shot_z_profile",
    ),
    velocity={
        "shot_crossing_y": AxisVerdict("all_nan", "honest_nan"),
        "shot_crossing_z": AxisVerdict("all_nan", "honest_nan"),
        "shot_speed": AxisVerdict("all_nan", "honest_nan"),
        "shot_time_to_goal_line": AxisVerdict("all_nan", "honest_nan"),
        "shot_on_target_derived": AxisVerdict("all_nan", "honest_nan"),
        "shot_crossing_source": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "A provenance column: its job is to report WHICH path produced the value, so reporting a "
                "different path on a freeze-frame leg than on a tracking leg is correct behaviour. ADR-043 "
                "designed das_source to do exactly this. [measured cause=frame_count]"
            ),
        ),
        "shot_crossing_confidence": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "A provenance column: its job is to report WHICH path produced the value, so reporting a "
                "different path on a freeze-frame leg than on a tracking leg is correct behaviour. ADR-043 "
                "designed das_source to do exactly this. [measured cause=frame_count]"
            ),
        ),
        "shot_fit_n_frames": AxisVerdict("all_nan", "honest_nan"),
        "shot_fit_rmse": AxisVerdict("all_nan", "honest_nan"),
        "shot_fit_end_reason": AxisVerdict("all_nan", "honest_nan"),
        "shot_z_profile": AxisVerdict("all_nan", "honest_nan"),
    },
    visibility={
        "gk_absent": {
            "shot_crossing_y": AxisVerdict("all_nan", "honest_nan"),
            "shot_crossing_z": AxisVerdict("all_nan", "honest_nan"),
            "shot_speed": AxisVerdict("all_nan", "honest_nan"),
            "shot_time_to_goal_line": AxisVerdict("all_nan", "honest_nan"),
            "shot_on_target_derived": AxisVerdict("all_nan", "honest_nan"),
            "shot_crossing_source": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "A provenance column: its job is to report WHICH path produced the value, so reporting a "
                    "different path on a freeze-frame leg than on a tracking leg is correct behaviour. ADR-043 "
                    "designed das_source to do exactly this. [measured cause=frame_count]"
                ),
            ),
            "shot_crossing_confidence": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "A provenance column: its job is to report WHICH path produced the value, so reporting a "
                    "different path on a freeze-frame leg than on a tracking leg is correct behaviour. ADR-043 "
                    "designed das_source to do exactly this. [measured cause=frame_count]"
                ),
            ),
            "shot_fit_n_frames": AxisVerdict("all_nan", "honest_nan"),
            "shot_fit_rmse": AxisVerdict("all_nan", "honest_nan"),
            "shot_fit_end_reason": AxisVerdict("all_nan", "honest_nan"),
            "shot_z_profile": AxisVerdict("all_nan", "honest_nan"),
        },
        "defender_absent": {
            "shot_crossing_y": AxisVerdict("all_nan", "honest_nan"),
            "shot_crossing_z": AxisVerdict("all_nan", "honest_nan"),
            "shot_speed": AxisVerdict("all_nan", "honest_nan"),
            "shot_time_to_goal_line": AxisVerdict("all_nan", "honest_nan"),
            "shot_on_target_derived": AxisVerdict("all_nan", "honest_nan"),
            "shot_crossing_source": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "A provenance column: its job is to report WHICH path produced the value, so reporting a "
                    "different path on a freeze-frame leg than on a tracking leg is correct behaviour. ADR-043 "
                    "designed das_source to do exactly this. [measured cause=frame_count]"
                ),
            ),
            "shot_crossing_confidence": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "A provenance column: its job is to report WHICH path produced the value, so reporting a "
                    "different path on a freeze-frame leg than on a tracking leg is correct behaviour. ADR-043 "
                    "designed das_source to do exactly this. [measured cause=frame_count]"
                ),
            ),
            "shot_fit_n_frames": AxisVerdict("all_nan", "honest_nan"),
            "shot_fit_rmse": AxisVerdict("all_nan", "honest_nan"),
            "shot_fit_end_reason": AxisVerdict("all_nan", "honest_nan"),
            "shot_z_profile": AxisVerdict("all_nan", "honest_nan"),
        },
        "gk_one_end": {
            "shot_crossing_y": AxisVerdict("all_nan", "honest_nan"),
            "shot_crossing_z": AxisVerdict("all_nan", "honest_nan"),
            "shot_speed": AxisVerdict("all_nan", "honest_nan"),
            "shot_time_to_goal_line": AxisVerdict("all_nan", "honest_nan"),
            "shot_on_target_derived": AxisVerdict("all_nan", "honest_nan"),
            "shot_crossing_source": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "A provenance column: its job is to report WHICH path produced the value, so reporting a "
                    "different path on a freeze-frame leg than on a tracking leg is correct behaviour. ADR-043 "
                    "designed das_source to do exactly this. [measured cause=frame_count]"
                ),
            ),
            "shot_crossing_confidence": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "A provenance column: its job is to report WHICH path produced the value, so reporting a "
                    "different path on a freeze-frame leg than on a tracking leg is correct behaviour. ADR-043 "
                    "designed das_source to do exactly this. [measured cause=frame_count]"
                ),
            ),
            "shot_fit_n_frames": AxisVerdict("all_nan", "honest_nan"),
            "shot_fit_rmse": AxisVerdict("all_nan", "honest_nan"),
            "shot_fit_end_reason": AxisVerdict("all_nan", "honest_nan"),
            "shot_z_profile": AxisVerdict("all_nan", "honest_nan"),
        },
    },
    applicability={
        "shot_crossing_y": "no_support",
        "shot_crossing_z": "no_support",
        "shot_speed": "no_support",
        "shot_time_to_goal_line": "no_support",
        "shot_on_target_derived": "no_support",
        "shot_crossing_source": "no_support",
        "shot_crossing_confidence": "no_support",
        "shot_fit_n_frames": "no_support",
        "shot_fit_rmse": "no_support",
        "shot_fit_end_reason": "no_support",
        "shot_z_profile": "no_support",
    },
    applicability_deltas={
        "shot_crossing_y": {"extreme": 0.0, "near": 0.0},
        "shot_crossing_z": {"extreme": 0.0, "near": 0.0},
        "shot_speed": {"extreme": 0.0, "near": 0.0},
        "shot_time_to_goal_line": {"extreme": 0.0, "near": 0.0},
        "shot_on_target_derived": {"extreme": 0.0, "near": 0.0},
        "shot_crossing_source": {"extreme": 0.0, "near": 0.0},
        "shot_crossing_confidence": {"extreme": 0.0, "near": 0.0},
        "shot_fit_n_frames": {"extreme": 0.0, "near": 0.0},
        "shot_fit_rmse": {"extreme": 0.0, "near": 0.0},
        "shot_fit_end_reason": {"extreme": 0.0, "near": 0.0},
        "shot_z_profile": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_xcross_attempt",
    C.generic(T.add_xcross_attempt),
    columns=("xcross_attempt",),
    velocity={
        "xcross_attempt": AxisVerdict(
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
            "xcross_attempt": AxisVerdict(
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
            "xcross_attempt": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "The fixture does not produce this column's domain on either leg (no pressing sequence, shot- "
                    "occurrence context, or blocking defender to score). A fixture inadequacy, not a library "
                    "property -- widening the fixture would move it. [measured cause=velocity+frame_count]"
                ),
            ),
        },
        "gk_one_end": {
            "xcross_attempt": AxisVerdict(
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
        "xcross_attempt": "no_support",
    },
    applicability_deltas={
        "xcross_attempt": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_xshot_occurrence",
    C.generic(T.add_xshot_occurrence),
    columns=("xshot_occurrence",),
    velocity={
        "xshot_occurrence": AxisVerdict(
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
            "xshot_occurrence": AxisVerdict(
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
            "xshot_occurrence": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "The fixture does not produce this column's domain on either leg (no pressing sequence, shot- "
                    "occurrence context, or blocking defender to score). A fixture inadequacy, not a library "
                    "property -- widening the fixture would move it. [measured cause=velocity+frame_count]"
                ),
            ),
        },
        "gk_one_end": {
            "xshot_occurrence": AxisVerdict(
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
        "xshot_occurrence": "no_support",
    },
    applicability_deltas={
        "xshot_occurrence": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_xt_gk",
    ADAPTERS["add_xt_gk"](T.add_xt_gk),
    columns=(
        "xt_gk_base",
        "xt_gk_pev",
        "xt_gk_rav",
        "xt_gk_dzv",
        "xt_gk_pressure",
        "xt_gk",
        "xt_gk_origin_x",
        "xt_gk_origin_y",
        "xt_gk_dest_x",
        "xt_gk_dest_y",
        "xt_gk_origin_source",
        "xt_gk_dest_source",
        "xt_gk_origin_confidence",
        "xt_gk_completion_variant",
        "xt_gk_completion_source",
        "xt_gk_native_goalkick_out_of_region",
    ),
    velocity={
        "xt_gk_base": AxisVerdict("identical", "works"),
        "xt_gk_pev": AxisVerdict("identical", "works"),
        "xt_gk_rav": AxisVerdict("identical", "works"),
        "xt_gk_dzv": AxisVerdict("identical", "works"),
        "xt_gk_pressure": AxisVerdict("identical", "works"),
        "xt_gk": AxisVerdict("identical", "works"),
        "xt_gk_origin_x": AxisVerdict("identical", "works"),
        "xt_gk_origin_y": AxisVerdict("identical", "works"),
        "xt_gk_dest_x": AxisVerdict("identical", "works"),
        "xt_gk_dest_y": AxisVerdict("identical", "works"),
        "xt_gk_origin_source": AxisVerdict("identical", "works"),
        "xt_gk_dest_source": AxisVerdict("identical", "works"),
        "xt_gk_origin_confidence": AxisVerdict("identical", "works"),
        "xt_gk_completion_variant": AxisVerdict("identical", "works"),
        "xt_gk_completion_source": AxisVerdict("identical", "works"),
        "xt_gk_native_goalkick_out_of_region": AxisVerdict("identical", "works"),
    },
    visibility={
        "gk_absent": {
            "xt_gk_base": AxisVerdict("identical", "works"),
            "xt_gk_pev": AxisVerdict("identical", "works"),
            "xt_gk_rav": AxisVerdict("identical", "works"),
            "xt_gk_dzv": AxisVerdict("identical", "works"),
            "xt_gk_pressure": AxisVerdict("identical", "works"),
            "xt_gk": AxisVerdict("identical", "works"),
            "xt_gk_origin_x": AxisVerdict("identical", "works"),
            "xt_gk_origin_y": AxisVerdict("identical", "works"),
            "xt_gk_dest_x": AxisVerdict("identical", "works"),
            "xt_gk_dest_y": AxisVerdict("identical", "works"),
            "xt_gk_origin_source": AxisVerdict("identical", "works"),
            "xt_gk_dest_source": AxisVerdict("identical", "works"),
            "xt_gk_origin_confidence": AxisVerdict("identical", "works"),
            "xt_gk_completion_variant": AxisVerdict("identical", "works"),
            "xt_gk_completion_source": AxisVerdict("identical", "works"),
            "xt_gk_native_goalkick_out_of_region": AxisVerdict("identical", "works"),
        },
        "defender_absent": {
            "xt_gk_base": AxisVerdict("identical", "works"),
            "xt_gk_pev": AxisVerdict("identical", "works"),
            "xt_gk_rav": AxisVerdict("identical", "works"),
            "xt_gk_dzv": AxisVerdict("identical", "works"),
            "xt_gk_pressure": AxisVerdict("identical", "works"),
            "xt_gk": AxisVerdict("identical", "works"),
            "xt_gk_origin_x": AxisVerdict("identical", "works"),
            "xt_gk_origin_y": AxisVerdict("identical", "works"),
            "xt_gk_dest_x": AxisVerdict("identical", "works"),
            "xt_gk_dest_y": AxisVerdict("identical", "works"),
            "xt_gk_origin_source": AxisVerdict("identical", "works"),
            "xt_gk_dest_source": AxisVerdict("identical", "works"),
            "xt_gk_origin_confidence": AxisVerdict("identical", "works"),
            "xt_gk_completion_variant": AxisVerdict("identical", "works"),
            "xt_gk_completion_source": AxisVerdict("identical", "works"),
            "xt_gk_native_goalkick_out_of_region": AxisVerdict("identical", "works"),
        },
        "gk_one_end": {
            "xt_gk_base": AxisVerdict("identical", "works"),
            "xt_gk_pev": AxisVerdict("identical", "works"),
            "xt_gk_rav": AxisVerdict("identical", "works"),
            "xt_gk_dzv": AxisVerdict("identical", "works"),
            "xt_gk_pressure": AxisVerdict("identical", "works"),
            "xt_gk": AxisVerdict("identical", "works"),
            "xt_gk_origin_x": AxisVerdict("identical", "works"),
            "xt_gk_origin_y": AxisVerdict("identical", "works"),
            "xt_gk_dest_x": AxisVerdict("identical", "works"),
            "xt_gk_dest_y": AxisVerdict("identical", "works"),
            "xt_gk_origin_source": AxisVerdict("identical", "works"),
            "xt_gk_dest_source": AxisVerdict("identical", "works"),
            "xt_gk_origin_confidence": AxisVerdict("identical", "works"),
            "xt_gk_completion_variant": AxisVerdict("identical", "works"),
            "xt_gk_completion_source": AxisVerdict("identical", "works"),
            "xt_gk_native_goalkick_out_of_region": AxisVerdict("identical", "works"),
        },
    },
    applicability={
        "xt_gk_base": "no_support",
        "xt_gk_pev": "region_support",
        "xt_gk_rav": "no_support",
        "xt_gk_dzv": "no_support",
        "xt_gk_pressure": "region_support",
        "xt_gk": "region_support",
        "xt_gk_origin_x": "no_support",
        "xt_gk_origin_y": "no_support",
        "xt_gk_dest_x": "no_support",
        "xt_gk_dest_y": "no_support",
        "xt_gk_origin_source": "no_support",
        "xt_gk_dest_source": "no_support",
        "xt_gk_origin_confidence": "no_support",
        "xt_gk_completion_variant": "no_support",
        "xt_gk_completion_source": "no_support",
        "xt_gk_native_goalkick_out_of_region": "no_support",
    },
    applicability_deltas={
        "xt_gk_base": {"extreme": 0.0, "near": 0.0},
        "xt_gk_pev": {"extreme": 0.0, "near": 0.21913932166756828},
        "xt_gk_rav": {"extreme": 0.0, "near": 0.0},
        "xt_gk_dzv": {"extreme": 0.0, "near": 0.0},
        "xt_gk_pressure": {"extreme": 0.0, "near": 0.5865351900097704},
        "xt_gk": {"extreme": 0.0, "near": 0.05478483041689208},
        "xt_gk_origin_x": {"extreme": 0.0, "near": 0.0},
        "xt_gk_origin_y": {"extreme": 0.0, "near": 0.0},
        "xt_gk_dest_x": {"extreme": 0.0, "near": 0.0},
        "xt_gk_dest_y": {"extreme": 0.0, "near": 0.0},
        "xt_gk_origin_source": {"extreme": 0.0, "near": 0.0},
        "xt_gk_dest_source": {"extreme": 0.0, "near": 0.0},
        "xt_gk_origin_confidence": {"extreme": 0.0, "near": 0.0},
        "xt_gk_completion_variant": {"extreme": 0.0, "near": 0.0},
        "xt_gk_completion_source": {"extreme": 0.0, "near": 0.0},
        "xt_gk_native_goalkick_out_of_region": {"extreme": 0.0, "near": 0.0},
    },
)
