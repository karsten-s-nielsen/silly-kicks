"""The match-load seam fails loud on tokenless/empty resolution, returns on a real ref.

Exercises _load.load_match's OWN logic by faking its internals (list_match_refs / _pining_load_match),
so no network is touched.
"""

import pandas as pd
import pytest

from silly_kicks.mcp import _load


def test_raises_on_empty_resolution(monkeypatch):
    monkeypatch.setattr(_load, "list_match_refs", lambda **kw: [])
    with pytest.raises(RuntimeError):
        _load.load_match("any-ref")


def test_raises_without_token(monkeypatch):
    # Tokenless: an owner provider resolves to no refs (exit 0) -> the seam RAISES, never returns OK.
    monkeypatch.delenv("PINING_FOR_THE_DATA_TOKEN", raising=False)
    monkeypatch.setattr(_load, "list_match_refs", lambda **kw: [])
    with pytest.raises(RuntimeError):
        _load.load_match("any-ref", provider="gradientsports")


def test_raises_on_empty_frames(monkeypatch):
    from _loader_pining import LoadedMatch, MatchRef

    ref = MatchRef("idsse", "x", {})
    empty = LoadedMatch("idsse", "x", pd.DataFrame({"a": [1]}), pd.DataFrame(), "H", None, None)
    monkeypatch.setattr(_load, "list_match_refs", lambda **kw: [ref])
    monkeypatch.setattr(_load, "_pining_load_match", lambda r, **kw: empty)
    with pytest.raises(RuntimeError):
        _load.load_match("x")


def test_ok_on_fixture(monkeypatch):
    from _loader_pining import LoadedMatch, MatchRef

    ref = MatchRef("idsse", "x", {})
    loaded = LoadedMatch("idsse", "x", pd.DataFrame({"a": [1]}), pd.DataFrame({"x": [1.0]}), "H", None, None)
    monkeypatch.setattr(_load, "list_match_refs", lambda **kw: [ref])
    monkeypatch.setattr(_load, "_pining_load_match", lambda r, **kw: loaded)
    assert _load.load_match("x") is not None
