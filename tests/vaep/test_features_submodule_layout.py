"""Submodule layout lock: each public symbol of silly_kicks.vaep.features
is defined in its expected submodule. Locks the 2.3.0 decomposition contract.
"""

from __future__ import annotations

import importlib

import pytest

_LAYOUT: tuple[tuple[str, str], ...] = (
    # core
    ("gamestates", "core"),
    ("simple", "core"),
    ("play_left_to_right", "core"),
    ("feature_column_names", "core"),
    ("Actions", "core"),
    ("GameStates", "core"),
    ("Features", "core"),
    ("FeatureTransfomer", "core"),
    # actiontype
    ("actiontype", "actiontype"),
    ("actiontype_onehot", "actiontype"),
    # result
    ("result", "result"),
    ("result_onehot", "result"),
    ("actiontype_result_onehot", "result"),
    ("result_onehot_prev_only", "result"),
    ("actiontype_result_onehot_prev_only", "result"),
    # bodypart
    ("bodypart", "bodypart"),
    ("bodypart_detailed", "bodypart"),
    ("bodypart_onehot", "bodypart"),
    ("bodypart_detailed_onehot", "bodypart"),
    # spatial
    ("startlocation", "spatial"),
    ("endlocation", "spatial"),
    ("startpolar", "spatial"),
    ("endpolar", "spatial"),
    ("movement", "spatial"),
    ("space_delta", "spatial"),
    # temporal
    ("time", "temporal"),
    ("time_delta", "temporal"),
    ("speed", "temporal"),
    # context
    ("team", "context"),
    ("player_possession_time", "context"),
    ("goalscore", "context"),
    # specialty
    ("cross_zone", "specialty"),
    ("assist_type", "specialty"),
)


@pytest.mark.parametrize("symbol_name, expected_submodule", _LAYOUT)
def test_symbol_lives_in_expected_submodule(symbol_name: str, expected_submodule: str) -> None:
    """Each symbol is defined in its expected submodule.

    For function/class symbols defined within our package, ``__module__`` is
    the canonical source. For type aliases (e.g. ``Actions = pd.DataFrame``),
    ``__module__`` points to the underlying class's defining module
    (``pandas.core.frame``), not where the alias was bound — we instead verify
    the symbol is reachable via the expected submodule path with object identity.
    """
    mod = importlib.import_module("silly_kicks.vaep.features")
    symbol = getattr(mod, symbol_name)
    expected_full = f"silly_kicks.vaep.features.{expected_submodule}"
    actual = getattr(symbol, "__module__", None)

    if actual is not None and actual.startswith("silly_kicks.vaep.features."):
        # Function or class defined within our package — __module__ is canonical.
        assert actual == expected_full, (
            f"{symbol_name} should be defined in {expected_full} (actually defined in {actual})"
        )
    else:
        # Type alias to external class (or no __module__) — fall back to
        # source-presence check via object identity.
        sub = importlib.import_module(expected_full)
        assert hasattr(sub, symbol_name), (
            f"Type alias {symbol_name} should be present in {expected_full} (not found via module attribute)"
        )
        assert getattr(sub, symbol_name) is symbol, (
            f"{symbol_name} accessible via package and submodule but they resolve to different objects"
        )
