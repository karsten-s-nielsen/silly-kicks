"""Layer-1 output column names + sample keys (single source, TF-60 / ADR-080)."""

from __future__ import annotations

RD_SAMPLE_KEYS = ["game_id", "period_id", "team_id", "action_id"]

#: Frame grouping key (single source for every per-frame O(1) lookup; ADR-068 group_rows).
RD_FRAME_KEYS = ["game_id", "period_id", "frame_id"]

# Layer-1 metric column bases (one per spec §7.1 row).
RD_NUM_SUPERIORITY = "rd_num_superiority"
RD_NUM_SUPERIORITY_GK = "rd_num_superiority_gk"
RD_ZONE_OCCUPANCY = "rd_zone_occupancy"
RD_LINE_HEIGHT = "rd_line_height"
RD_LINE_HEIGHT_RELATIVE = "rd_line_height_relative"
RD_COMPACTNESS_X = "rd_compactness_x"
RD_WIDTH = "rd_width"
RD_DEPTH = "rd_depth"
RD_SHAPE_STAGGER = "rd_shape_2_3_vs_3_2"
RD_GK_LINE_HEIGHT = "rd_gk_line_height"
RD_GK_TO_LINE_DISTANCE = "rd_gk_to_line_distance"

RD_LAYER1_COLUMNS = [
    RD_NUM_SUPERIORITY,
    RD_NUM_SUPERIORITY_GK,
    RD_ZONE_OCCUPANCY,
    RD_LINE_HEIGHT,
    RD_LINE_HEIGHT_RELATIVE,
    RD_COMPACTNESS_X,
    RD_WIDTH,
    RD_DEPTH,
    RD_SHAPE_STAGGER,
    RD_GK_LINE_HEIGHT,
    RD_GK_TO_LINE_DISTANCE,
]

#: Provenance for the goal-end resolution behind every geometry-dependent metric.
RD_GEOMETRY_SOURCE = "rd_geometry_source"  # {"resolved", "guessed", "unresolved"}

#: Closed vocabulary for RD_GEOMETRY_SOURCE. "resolved" = a GoalMap end from clear GK evidence;
#: "guessed" = a GoalMap allow_guess fallback (its metrics are computed but the end is an inference,
#: which matters on FOV-cropped SB360, IMPL-02); "unresolved" = no end at all -> honest-NaN metrics.
RD_GEOMETRY_SOURCE_VALUES = ("resolved", "guessed", "unresolved")
