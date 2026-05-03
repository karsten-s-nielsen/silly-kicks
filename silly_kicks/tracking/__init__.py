"""silly_kicks.tracking --- tracking-data namespace.

PR-S19 (silly-kicks 2.7.0) shipped the primitive layer: schema, per-provider
adapters, and the link_actions_to_frames primitive. PR-S20 (silly-kicks 2.8.0,
ADR-005) shipped the first tracking-aware feature set (action_context: 4
features + aggregator + lift_to_states extension utility) on top of those
primitives. PR-S21 (silly-kicks 2.9.0) shipped pre_shot_gk_position_*. PR-S24
(silly-kicks 3.1.0) ships TF-6 sync_score, TF-8 smoothing/velocity, TF-9
interpolation, TF-12 pre_shot_gk_angle_* + the shared `preprocess` module.
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
    "add_pre_shot_gk_angle",
    "add_pre_shot_gk_position",
    "add_sync_score",
    "defenders_in_triangle_to_goal",
    "derive_velocities",
    "feature_framework",
    "features",
    "get_provider_defaults",
    "interpolate_frames",
    "kloppy",
    "lift_to_states",
    "link_actions_to_frames",
    "nearest_defender_distance",
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
from .feature_framework import ActionFrameContext, lift_to_states
from .features import (
    actor_speed,
    add_action_context,
    add_pre_shot_gk_angle,
    add_pre_shot_gk_position,
    defenders_in_triangle_to_goal,
    nearest_defender_distance,
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
