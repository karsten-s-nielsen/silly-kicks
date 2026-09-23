"""`scripts/_item_outcome.ItemExcluded` — the leaf module the loaders raise and `_driver` recognises.

tests/scripts/ has NO __init__.py on purpose (it mirrors the sibling script tests); conftest puts
scripts/ on sys.path, so the bare import works.
"""

from __future__ import annotations

import ast
import inspect

import _item_outcome
import pytest
from _item_outcome import ItemExcluded


def test_reason_and_details_round_trip():
    exc = ItemExcluded("S1 geometry gate", details={"player_off_pitch_rate": 0.34, "ball_off_pitch_rate": 0.001})
    assert exc.reason == "S1 geometry gate"
    assert exc.details == {"player_off_pitch_rate": 0.34, "ball_off_pitch_rate": 0.001}


def test_details_defaults_to_empty_dict():
    assert ItemExcluded("r").details == {}


def test_it_is_an_exception():
    assert issubclass(ItemExcluded, Exception)
    with pytest.raises(ItemExcluded):
        raise ItemExcluded("r")


def test_empty_reason_is_refused():
    with pytest.raises(ValueError, match="reason"):
        ItemExcluded("")


def test_non_json_serializable_details_refused_at_construction():
    # A marker with unserializable details would otherwise fail only at write time, mid-corpus.
    with pytest.raises((TypeError, ValueError)):
        ItemExcluded("r", details={"x": object()})


def test_single_identity_across_import_paths():
    """`_driver` imports `scripts._item_outcome`; tests + some drivers use bare `_item_outcome`. Both
    MUST be the same class, or `except ItemExcluded` in `_driver` misses a loader's `MatchExcluded`."""
    import scripts._item_outcome as pkg_path

    assert pkg_path.ItemExcluded is ItemExcluded
    assert pkg_path is _item_outcome


def test_imports_nothing_from_scripts():
    """It is the leaf both `_driver` and the loaders import; importing the orchestration seam here
    would create the dependency cycle the module exists to avoid (spec §4.1, CDLS-SPEC-12)."""
    forbidden = {"scripts", "_driver", "_loader_pining", "_partition", "_sb_open_data", "_xt_corpus"}
    tree = ast.parse(inspect.getsource(_item_outcome))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden, alias.name
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden, node.module
