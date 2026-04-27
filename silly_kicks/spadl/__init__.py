"""Implementation of the SPADL language."""

__all__ = [
    "SPADL_COLUMNS",
    "ConversionReport",
    "actiontypes_df",
    "add_gk_distribution_metrics",
    "add_gk_role",
    "add_names",
    "add_possessions",
    "add_pre_shot_gk_context",
    "bodyparts_df",
    "config",
    "kloppy",
    "opta",
    "play_left_to_right",
    "results_df",
    "statsbomb",
    "validate_spadl",
    "wyscout",
]

from . import config, opta, statsbomb, wyscout
from .config import actiontypes_df, bodyparts_df, results_df
from .schema import SPADL_COLUMNS, ConversionReport
from .utils import (
    add_gk_distribution_metrics,
    add_gk_role,
    add_names,
    add_possessions,
    add_pre_shot_gk_context,
    play_left_to_right,
    validate_spadl,
)

try:
    from . import kloppy
except ImportError:
    pass
