"""Implementation of the SPADL language."""

__all__ = [
    "ABSOLUTE_FRAME_HOME_RIGHT",
    "PER_PERIOD_ABSOLUTE",
    "PFF_SPADL_COLUMNS",
    "POSSESSION_PERSPECTIVE",
    "SPADL_COLUMNS",
    "SPORTEC_SPADL_COLUMNS",
    "BoundaryMetrics",
    "ConversionReport",
    "CoverageMetrics",
    "DetectionResult",
    "InputConvention",
    "actiontypes_df",
    "add_gk_distribution_metrics",
    "add_gk_role",
    "add_names",
    "add_possessions",
    "add_pre_shot_gk_context",
    "bodyparts_df",
    "boundary_metrics",
    "config",
    "coverage_metrics",
    "detect_input_convention",
    "kloppy",
    "opta",
    "pff",
    "play_left_to_right",
    "results_df",
    "statsbomb",
    "to_spadl_ltr",
    "use_tackle_winner_as_actor",
    "validate_input_convention",
    "validate_spadl",
    "wyscout",
]

from . import config, opta, pff, statsbomb, wyscout
from .config import actiontypes_df, bodyparts_df, results_df
from .orientation import (
    ABSOLUTE_FRAME_HOME_RIGHT,
    PER_PERIOD_ABSOLUTE,
    POSSESSION_PERSPECTIVE,
    DetectionResult,
    InputConvention,
    detect_input_convention,
    to_spadl_ltr,
    validate_input_convention,
)
from .schema import PFF_SPADL_COLUMNS, SPADL_COLUMNS, SPORTEC_SPADL_COLUMNS, ConversionReport
from .sportec import use_tackle_winner_as_actor
from .utils import (
    BoundaryMetrics,
    CoverageMetrics,
    add_gk_distribution_metrics,
    add_gk_role,
    add_names,
    add_possessions,
    add_pre_shot_gk_context,
    boundary_metrics,
    coverage_metrics,
    play_left_to_right,
    validate_spadl,
)

try:
    from . import kloppy
except ImportError:
    pass
