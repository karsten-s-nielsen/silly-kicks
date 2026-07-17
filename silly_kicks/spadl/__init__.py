"""Implementation of the SPADL language."""

__all__ = [
    "ABSOLUTE_FRAME_HOME_RIGHT",
    "GRADIENTSPORTS_SPADL_COLUMNS",
    "PER_PERIOD_ABSOLUTE",
    "POSSESSION_PERSPECTIVE",
    "SKILLCORNER_SPADL_COLUMNS",
    "SPADL_COLUMNS",
    "SPORTEC_SPADL_COLUMNS",
    "BoundaryMetrics",
    "ConversionReport",
    "CoverageMetrics",
    "DetectionResult",
    "InputConvention",
    "actiontypes_df",
    "add_game_state",
    "add_gk_distribution_metrics",
    "add_gk_role",
    "add_names",
    "add_possessions",
    "add_pre_shot_gk_context",
    "add_restart_coordinates",
    "bodyparts_df",
    "boundary_metrics",
    "config",
    "coverage_metrics",
    "detect_input_convention",
    "gradientsports",
    "kloppy",
    "opta",
    "play_left_to_right",
    "require_et_direction",
    "resolve_next_touch_receiver",
    "results_df",
    "skillcorner",
    "statsbomb",
    "to_spadl_ltr",
    "use_tackle_winner_as_actor",
    "validate_input_convention",
    "validate_spadl",
    "wyscout",
]

from ..tracking.direction import require_et_direction
from . import config, gradientsports, opta, statsbomb, wyscout
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
from .schema import (
    GRADIENTSPORTS_SPADL_COLUMNS,
    SKILLCORNER_SPADL_COLUMNS,
    SPADL_COLUMNS,
    SPORTEC_SPADL_COLUMNS,
    ConversionReport,
)
from .sportec import use_tackle_winner_as_actor
from .utils import (
    BoundaryMetrics,
    CoverageMetrics,
    add_game_state,
    add_gk_distribution_metrics,
    add_gk_role,
    add_names,
    add_possessions,
    add_pre_shot_gk_context,
    add_restart_coordinates,
    boundary_metrics,
    coverage_metrics,
    play_left_to_right,
    resolve_next_touch_receiver,
    validate_spadl,
)

try:
    from . import kloppy
except ImportError:
    pass

try:
    from . import skillcorner
except ImportError:
    pass
