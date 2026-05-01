"""silly_kicks.tracking --- tracking-data namespace.

PR-S19 (silly-kicks 2.7.0) shipped the primitive layer: schema, per-provider
adapters, and the link_actions_to_frames primitive. PR-S20 (silly-kicks 2.8.0,
ADR-005) shipped the first tracking-aware feature set (action_context: 4
features + aggregator + lift_to_states extension utility) on top of those
primitives.
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
    "TrackingConversionReport",
    "actor_speed",
    "add_action_context",
    "defenders_in_triangle_to_goal",
    "feature_framework",
    "features",
    "kloppy",
    "lift_to_states",
    "link_actions_to_frames",
    "nearest_defender_distance",
    "pff",
    "play_left_to_right",
    "receiver_zone_density",
    "schema",
    "slice_around_event",
    "sportec",
    "tracking_default_xfns",
    "utils",
]

from . import feature_framework, features, pff, schema, sportec, utils
from .feature_framework import ActionFrameContext, lift_to_states
from .features import (
    actor_speed,
    add_action_context,
    defenders_in_triangle_to_goal,
    nearest_defender_distance,
    receiver_zone_density,
    tracking_default_xfns,
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
from .utils import link_actions_to_frames, play_left_to_right, slice_around_event

try:
    from . import kloppy
except ImportError:
    pass
