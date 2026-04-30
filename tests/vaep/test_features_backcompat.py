"""Backwards-compat: every currently-public symbol of silly_kicks.vaep.features
remains importable via the package path through and after the 2.3.0 decomposition.
"""

from __future__ import annotations

import importlib

import pytest

_PUBLIC_SYMBOLS: tuple[str, ...] = (
    "Actions",
    "Features",
    "FeatureTransfomer",
    "GameStates",
    "actiontype",
    "actiontype_onehot",
    "actiontype_result_onehot",
    "actiontype_result_onehot_prev_only",
    "assist_type",
    "bodypart",
    "bodypart_detailed",
    "bodypart_detailed_onehot",
    "bodypart_onehot",
    "cross_zone",
    "endlocation",
    "endpolar",
    "feature_column_names",
    "gamestates",
    "goalscore",
    "movement",
    "play_left_to_right",
    "player_possession_time",
    "result",
    "result_onehot",
    "result_onehot_prev_only",
    "simple",
    "space_delta",
    "speed",
    "startlocation",
    "startpolar",
    "team",
    "time",
    "time_delta",
)


@pytest.mark.parametrize("symbol_name", _PUBLIC_SYMBOLS)
def test_symbol_importable_from_package_path(symbol_name: str) -> None:
    """Every currently-public symbol stays importable from the package path."""
    mod = importlib.import_module("silly_kicks.vaep.features")
    assert hasattr(mod, symbol_name), (
        f"{symbol_name} no longer importable from silly_kicks.vaep.features. "
        f"Decomposition must preserve every public symbol via __init__.py re-exports."
    )
