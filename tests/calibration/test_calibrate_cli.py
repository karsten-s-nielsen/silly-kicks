"""TF-24 calibrate CLI arg parsing + fold-loader threading (plan Task 9b).

The local TF-24 sweep OOMs when the loader pulls all matches at full tracking
depth; these flags bound memory. Verify ``--match-ids`` parsing and that
``_load_fold`` threads ``match_ids`` / ``tracking_limit`` / ``max_per_provider``
into ``load_matches`` instead of the old hardcoded ``match_ids=None``.
"""

from __future__ import annotations

import types

import pytest

import scripts.calibrate_tracking_defaults as C


def test_parse_match_ids_repeatable():
    out = C._parse_match_ids(["gradientsports:10517,10519", "idsse:M1"])
    assert out == {"gradientsports": ["10517", "10519"], "idsse": ["M1"]}


def test_parse_match_ids_none():
    assert C._parse_match_ids(None) is None
    assert C._parse_match_ids([]) is None


def test_parse_match_ids_rejects_malformed():
    with pytest.raises(ValueError, match="PROVIDER:id1"):
        C._parse_match_ids(["gradientsports"])  # no ':ids'


def test_load_fold_threads_memory_bounds(monkeypatch):
    captured = {}

    def _fake_load_matches(**kwargs):
        captured.update(kwargs)
        return iter([("gradientsports", "10517", "ACT", "FRM", "H")])

    import scripts._loader_pining as L

    monkeypatch.setattr(L, "load_matches", _fake_load_matches)
    args = types.SimpleNamespace(
        source="pining",
        providers=["gradientsports"],
        match_ids=["gradientsports:10517"],
        tracking_limit=200,
        max_matches_per_provider=3,
    )
    fold, used_ids = C._load_fold(args)

    assert captured["match_ids"] == {"gradientsports": ["10517"]}
    assert captured["tracking_limit"] == 200
    assert captured["max_per_provider"] == 3
    assert captured["providers"] == ["gradientsports"]
    assert used_ids == {"gradientsports": ["10517"]}
    assert fold["gradientsports"] == [("ACT", "FRM", "H")]
