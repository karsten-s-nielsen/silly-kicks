"""Task 8.5: the two tracking/open-data source factories (spec §4.1/§4.5, owner-ratified reuse).

``pining_source`` / ``open_data_source`` collapse the 3-line source construction every migrated
driver hand-repeated into one ``(refs, load)`` factory per load mode. They are thin: ``refs`` is
exactly ``list_match_refs`` / ``list_open_data_refs``, and ``load(ref)`` is exactly
``load_match(events_only=False, …)`` / ``load_open_data_match(…)`` with the cache dir resolved once.
"""

from __future__ import annotations

import pathlib

import scripts._loader_pining as lp
import scripts._sb_open_data as od


def test_pining_source_refs_are_list_match_refs(monkeypatch):
    sentinel = [lp.MatchRef("skillcorner", "1"), lp.MatchRef("skillcorner", "2")]
    seen: dict = {}

    def fake_list(**kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(lp, "list_match_refs", fake_list)
    monkeypatch.setattr(lp, "load_match", lambda *a, **k: None)

    refs, _load = lp.pining_source(
        ["skillcorner"], match_ids={"skillcorner": ["1"]}, max_per_provider=3, token="T", base_url="B"
    )
    assert refs is sentinel
    assert seen == {
        "providers": ["skillcorner"],
        "match_ids": {"skillcorner": ["1"]},
        "max_per_provider": 3,
        "token": "T",
        "base_url": "B",
    }


def test_pining_source_load_is_load_match_events_only_false(monkeypatch):
    calls: list[dict] = []

    def fake_load(ref, **kwargs):
        calls.append({"ref": ref, **kwargs})
        return "LOADED"

    monkeypatch.setattr(lp, "list_match_refs", lambda **k: [])
    monkeypatch.setattr(lp, "load_match", fake_load)

    _refs, load = lp.pining_source(["idsse"], tracking_limit=50, cache_dir="/cache", token="T", base_url="B")
    ref = lp.MatchRef("idsse", "9")
    assert load(ref) == "LOADED"
    (c,) = calls
    assert c["ref"] is ref
    assert c["events_only"] is False
    assert c["tracking_limit"] == 50
    assert c["cache_dir"] == lp.resolve_cache_dir("/cache")
    assert c["token"] == "T"  # noqa: S105
    assert c["base_url"] == "B"


def test_pining_source_resolves_cache_dir_once(monkeypatch):
    """The cache dir is resolved in the factory, not per load call (env/arg/None precedence applied once)."""
    monkeypatch.setenv(lp.CORPUS_CACHE_ENV, "/env-cache")
    monkeypatch.setattr(lp, "list_match_refs", lambda **k: [])
    seen: list = []
    monkeypatch.setattr(lp, "load_match", lambda ref, **k: seen.append(k["cache_dir"]))

    _refs, load = lp.pining_source(["gradientsports"])  # no cache_dir arg -> env
    load(lp.MatchRef("gradientsports", "1"))
    assert seen == [pathlib.Path("/env-cache")]


def test_open_data_source_refs_are_list_open_data_refs(monkeypatch):
    sentinel = [od.OpenDataRef(1, 2, "3", 7, "")]
    seen: dict = {}

    def fake_list(competitions, **kwargs):
        seen["competitions"] = competitions
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(od, "list_open_data_refs", fake_list)
    monkeypatch.setattr(od, "load_open_data_match", lambda *a, **k: None)

    refs, _load = od.open_data_source([(1, 2)], match_ids=["3"], max_matches=5)
    assert refs is sentinel
    assert seen == {"competitions": [(1, 2)], "match_ids": ["3"], "max_matches": 5}


def test_open_data_source_load_threads_preserve_native_and_cache(monkeypatch):
    calls: list[dict] = []

    def fake_load(ref, **kwargs):
        calls.append({"ref": ref, **kwargs})
        return "OD"

    monkeypatch.setattr(od, "list_open_data_refs", lambda *a, **k: [])
    monkeypatch.setattr(od, "load_open_data_match", fake_load)

    _refs, load = od.open_data_source([(1, 2)], preserve_native=("shot_type",), cache_dir="/od-cache")
    ref = od.OpenDataRef(1, 2, "3", 7, "")
    assert load(ref) == "OD"
    (c,) = calls
    assert c["ref"] is ref
    assert c["preserve_native"] == ("shot_type",)
    assert c["cache_dir"] == od.resolve_cache_dir("/od-cache")
