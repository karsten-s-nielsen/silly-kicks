"""silly_kicks.tracking --- tracking-data namespace (PR-S19, primitive layer).

Schema, per-provider adapters, and the link_actions_to_frames primitive.
Tracking-aware features (action_context, pressure_on_carrier, pitch control,
etc.) ship in PR-S20+ scoping cycles. See ADR-004.
"""

__all__ = [
    "KLOPPY_TRACKING_FRAMES_COLUMNS",
    "PFF_TRACKING_FRAMES_COLUMNS",
    "SPORTEC_TRACKING_FRAMES_COLUMNS",
    "TRACKING_CATEGORICAL_DOMAINS",
    "TRACKING_CONSTRAINTS",
    "TRACKING_FRAMES_COLUMNS",
    "LinkReport",
    "TrackingConversionReport",
    "kloppy",
    "link_actions_to_frames",
    "pff",
    "play_left_to_right",
    "schema",
    "slice_around_event",
    "sportec",
    "utils",
]

from . import pff, schema, sportec, utils
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
