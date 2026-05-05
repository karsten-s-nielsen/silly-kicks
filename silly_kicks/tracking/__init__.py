"""silly_kicks.tracking --- tracking-data namespace.

PR-S19 (silly-kicks 2.7.0) shipped the primitive layer: schema, per-provider
adapters, and the link_actions_to_frames primitive. PR-S20 (silly-kicks 2.8.0,
ADR-005) shipped the first tracking-aware feature set (action_context: 4
features + aggregator + lift_to_states extension utility) on top of those
primitives. PR-S21 (silly-kicks 2.9.0) shipped pre_shot_gk_position_*. PR-S24
(silly-kicks 3.1.0) ships TF-6 sync_score, TF-8 smoothing/velocity, TF-9
interpolation, TF-12 pre_shot_gk_angle_* + the shared `preprocess` module.
PR-S27 ships TF-13 defending_gk_from_frames + TF-14 compute_defensive_line +
6 per-Series action-coupled features + add_defensive_line + defensive_line_xfns.
"""

__all__ = [
    "KLOPPY_TRACKING_FRAMES_COLUMNS",
    "PFF_TRACKING_FRAMES_COLUMNS",
    "SPORTEC_TRACKING_FRAMES_COLUMNS",
    "TRACKING_CATEGORICAL_DOMAINS",
    "TRACKING_CONSTRAINTS",
    "TRACKING_FRAMES_COLUMNS",
    "ActionFrameContext",
    "LinkReport",
    "PreprocessConfig",
    "TrackingConversionReport",
    "actor_speed",
    "add_action_context",
    "add_defensive_line",
    "add_line_break",
    "add_off_ball_context",
    "add_off_ball_runs",
    "add_pre_shot_gk_angle",
    "add_pre_shot_gk_position",
    "add_sync_score",
    "back_line_high_x",
    "back_n_count",
    "ball_carrier_at_action",
    "compactness_x",
    "compute_defensive_line",
    "defenders_in_triangle_to_goal",
    "defending_gk_from_frames",
    "defensive_line_x",
    "defensive_line_xfns",
    "derive_velocities",
    "feature_framework",
    "features",
    "get_provider_defaults",
    "infer_ball_carrier",
    "interpolate_frames",
    "kloppy",
    "lateral_width",
    "lift_to_states",
    "link_actions_to_frames",
    "max_lateral_gap",
    "nearest_defender_distance",
    "off_ball_context_xfns",
    "pff",
    "play_left_to_right",
    "pre_shot_gk_angle_default_xfns",
    "pre_shot_gk_angle_off_goal_line",
    "pre_shot_gk_angle_to_shot_trajectory",
    "pre_shot_gk_default_xfns",
    "pre_shot_gk_full_default_xfns",
    "preprocess",
    "receiver_zone_density",
    "schema",
    "slice_around_event",
    "smooth_frames",
    "sportec",
    "sync_score",
    "tracking_default_xfns",
    "utils",
]

from . import feature_framework, features, pff, preprocess, schema, sportec, utils
from ._ball_carrier import infer_ball_carrier
from ._defensive_line import compute_defensive_line
from .feature_framework import ActionFrameContext, lift_to_states
from .features import (
    actor_speed,
    add_action_context,
    add_defensive_line,
    add_line_break,
    add_off_ball_context,
    add_off_ball_runs,
    add_pre_shot_gk_angle,
    add_pre_shot_gk_position,
    back_line_high_x,
    back_n_count,
    ball_carrier_at_action,
    compactness_x,
    defenders_in_triangle_to_goal,
    defending_gk_from_frames,
    defensive_line_x,
    defensive_line_xfns,
    lateral_width,
    max_lateral_gap,
    nearest_defender_distance,
    off_ball_context_xfns,
    pre_shot_gk_angle_default_xfns,
    pre_shot_gk_angle_off_goal_line,
    pre_shot_gk_angle_to_shot_trajectory,
    pre_shot_gk_default_xfns,
    pre_shot_gk_full_default_xfns,
    receiver_zone_density,
    tracking_default_xfns,
)
from .preprocess import (
    PreprocessConfig,
    derive_velocities,
    get_provider_defaults,
    interpolate_frames,
    smooth_frames,
)
from .schema import (
    KLOPPY_TRACKING_FRAMES_COLUMNS,
    PFF_TRACKING_FRAMES_COLUMNS,
    SPORTEC_TRACKING_FRAMES_COLUMNS,
    TRACKING_CATEGORICAL_DOMAINS,
    TRACKING_CONSTRAINTS,
    TRACKING_FRAMES_COLUMNS,
    LinkReport,
    TrackingConversionReport,
)
from .utils import add_sync_score, link_actions_to_frames, play_left_to_right, slice_around_event, sync_score

try:
    from . import kloppy
except ImportError:
    pass
