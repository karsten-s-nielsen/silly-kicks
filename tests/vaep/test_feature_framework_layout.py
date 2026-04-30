"""Framework module layout lock: every framework primitive's canonical
home is silly_kicks.vaep.feature_framework. Closes A9 (silly-kicks 2.4.0).

For function/class symbols, ``__module__`` is the canonical check. For
type aliases (``Actions = pd.DataFrame``), ``__module__`` points to the
alias target's defining module (``pandas.core.frame``); attribute
presence on the framework module is the binding check.
"""

from __future__ import annotations

import importlib

import pytest

_FRAMEWORK_SYMBOLS: tuple[str, ...] = (
    "Actions",
    "FeatureTransfomer",
    "Features",
    "GameStates",
    "actiontype_categorical",
    "gamestates",
    "simple",
)

_FRAMEWORK_MODULE = "silly_kicks.vaep.feature_framework"


@pytest.mark.parametrize("symbol_name", _FRAMEWORK_SYMBOLS)
def test_symbol_lives_in_framework_module(symbol_name: str) -> None:
    """Each framework primitive is canonically defined in feature_framework."""
    mod = importlib.import_module(_FRAMEWORK_MODULE)
    assert hasattr(mod, symbol_name), f"{symbol_name} not exposed by {_FRAMEWORK_MODULE}"
    symbol = getattr(mod, symbol_name)
    actual = getattr(symbol, "__module__", None)
    if actual is not None and actual.startswith("silly_kicks."):
        # Function/class defined within our package — __module__ is canonical.
        assert actual == _FRAMEWORK_MODULE, (
            f"{symbol_name} should be defined in {_FRAMEWORK_MODULE} (actually defined in {actual})"
        )
    # Else: type alias. Attribute-presence check above is sufficient.
