"""Open-data loader: `list_open_data_refs`, `load_open_data_match`, the shared cache, wrapper parity.

tests/scripts/ has NO __init__.py; conftest puts scripts/ on sys.path.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import _sb_open_data as sbod
import pandas as pd
import pytest

_360 = Path(__file__).resolve().parents[1] / "datasets" / "statsbomb" / "three-sixty"
_EVENTS = json.loads((_360 / "events.json").read_text(encoding="utf-8"))
if isinstance(_EVENTS, dict):
    _EVENTS = list(_EVENTS.values())
_HOME = int(next(e["team"]["id"] for e in _EVENTS if e.get("team")))


class _FakeSb:
    """A statsbombpy stand-in over the committed slice: counts events() fetches (cache round-trip)."""

    def __init__(self):
        self.event_calls: list[int] = []
        self.frame_calls: list[int] = []
        self.manifests = {
            (43, 106): {
                101: {"home_team": {"home_team_id": _HOME}, "match_date": "2022-11-20"},
                102: {"home_team": {"home_team_id": _HOME}, "match_date": "2022-11-21"},
            },
            (55, 43): {201: {"home_team": {"home_team_id": _HOME}, "match_date": "2021-06-11"}},
        }

    def matches(self, *, competition_id, season_id, fmt):
        assert fmt == "dict"
        return self.manifests[(competition_id, season_id)]

    def events(self, *, match_id, fmt):
        assert fmt == "dict"
        self.event_calls.append(match_id)
        return {e["id"]: e for e in _EVENTS}

    def frames(self, *, match_id, fmt):
        assert fmt == "dict"
        self.frame_calls.append(match_id)
        return [{"event_uuid": "u1", "visible_area": [], "freeze_frame": []}]


@pytest.fixture
def fake_sb(monkeypatch):
    monkeypatch.delenv("SB_USERNAME", raising=False)
    monkeypatch.delenv("SB_PASSWORD", raising=False)
    monkeypatch.delenv("SILLY_KICKS_CORPUS_CACHE_DIR", raising=False)
    sb = _FakeSb()
    monkeypatch.setitem(sys.modules, "statsbombpy", types.SimpleNamespace(sb=sb))
    return sb


# ---- list_open_data_refs -------------------------------------------------------------------------


def test_refs_walk_competitions_in_order_with_a_GLOBAL_cap(fake_sb):
    refs = sbod.list_open_data_refs([(43, 106), (55, 43)])
    assert [r.key for r in refs] == [("statsbomb", "101"), ("statsbomb", "102"), ("statsbomb", "201")]
    assert refs[0] == sbod.OpenDataRef(43, 106, "101", _HOME, "2022-11-20")
    capped = sbod.list_open_data_refs([(43, 106), (55, 43)], max_matches=2)
    assert [r.key for r in capped] == [("statsbomb", "101"), ("statsbomb", "102")], "cap is GLOBAL"


def test_refs_filter_by_match_ids_across_competitions(fake_sb):
    refs = sbod.list_open_data_refs([(43, 106), (55, 43)], match_ids=["201", "102"])
    assert [r.match_id for r in refs] == ["102", "201"], "manifest order, not request order"


def test_a_ref_compares_on_key_fields_not_date():
    assert sbod.OpenDataRef(43, 106, "101", 1, "a") == sbod.OpenDataRef(43, 106, "101", 1, "b")


def test_refs_fail_closed_on_credentials(fake_sb, monkeypatch):
    monkeypatch.setenv("SB_PASSWORD", "secret")
    with pytest.raises(SystemExit, match="public-only"):
        sbod.list_open_data_refs([(43, 106)])


# ---- load_open_data_match ------------------------------------------------------------------------


def test_load_open_data_match_is_event_only_with_names_and_xg(fake_sb):
    ref = sbod.list_open_data_refs([(43, 106)])[0]
    m = sbod.load_open_data_match(ref)
    assert (m.provider, m.match_id, m.home_team_id, m.visible_area, m.report) == ("statsbomb", "101", _HOME, None, None)
    assert m.frames is not None and m.frames.empty  # event-only: empty DataFrame
    assert {"player_name", "xg"} <= set(m.actions.columns) and len(m.actions) > 0


def test_the_cache_fetches_once(fake_sb, tmp_path):
    ref = sbod.list_open_data_refs([(43, 106)])[0]
    first = sbod.load_open_data_match(ref, cache_dir=tmp_path)
    second = sbod.load_open_data_match(ref, cache_dir=tmp_path)
    assert fake_sb.event_calls == [101], "the second load reads the cached file"
    assert (tmp_path / "statsbomb_open" / "101.json").is_file()
    pd.testing.assert_frame_equal(first.actions, second.actions)


def test_the_env_var_is_the_cache_when_no_argument(fake_sb, tmp_path, monkeypatch):
    monkeypatch.setenv("SILLY_KICKS_CORPUS_CACHE_DIR", str(tmp_path))
    ref = sbod.list_open_data_refs([(43, 106)])[0]
    sbod.load_open_data_match(ref)
    sbod.load_open_data_match(ref)
    assert fake_sb.event_calls == [101]


def test_no_cache_fetches_every_time_and_writes_nothing(fake_sb, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ref = sbod.list_open_data_refs([(43, 106)])[0]
    sbod.load_open_data_match(ref)
    sbod.load_open_data_match(ref)
    assert fake_sb.event_calls == [101, 101]


# ---- fetch_open_360_raw (the raw events + 360 fetch for build_sb360_coverage) --------------------


def test_fetch_open_360_raw_returns_both_and_caches_each_once(fake_sb, tmp_path):
    events, frames_raw = sbod.fetch_open_360_raw(101, cache_dir=tmp_path)
    assert isinstance(events, list) and events and isinstance(events[0], dict)
    assert isinstance(frames_raw, list) and "visible_area" in frames_raw[0]
    # a second call reads BOTH from the (distinct-filename) cache -- no re-fetch of either.
    sbod.fetch_open_360_raw(101, cache_dir=tmp_path)
    assert fake_sb.event_calls == [101] and fake_sb.frame_calls == [101]
    assert (tmp_path / "statsbomb_open" / "101.json").is_file()
    assert (tmp_path / "statsbomb_open" / "101.360.json").is_file()


def test_fetch_open_360_raw_fails_closed_on_credentials(fake_sb, monkeypatch, tmp_path):
    monkeypatch.setenv("SB_USERNAME", "u")
    with pytest.raises(SystemExit, match="public-only"):
        sbod.fetch_open_360_raw(101, cache_dir=tmp_path)
    assert not list(tmp_path.rglob("*.json"))  # refused before any fetch -> nothing cached


# ---- wrapper parity ------------------------------------------------------------------------------


@pytest.mark.parametrize("kw", [{}, {"match_ids": ["102"]}, {"max_matches": 1}])
def test_load_open_data_matches_wrapper_is_byte_identical_to_4ac26d0(fake_sb, kw):
    from _frozen_loader_4ac26d0 import load_open_data_matches_4ac26d0

    live = list(sbod.load_open_data_matches(competition_id=43, season_id=106, **kw))
    frozen = list(load_open_data_matches_4ac26d0(competition_id=43, season_id=106, **kw))
    assert [(p, m, h) for p, m, _a, _f, h in live] == [(p, m, h) for p, m, _a, _f, h in frozen]
    assert len(live) == len(frozen) and len(live) > 0
    for (_p, _m, a_live, f_live, _h), (_p2, _m2, a_frozen, f_frozen, _h2) in zip(live, frozen, strict=True):
        pd.testing.assert_frame_equal(a_live, a_frozen)
        assert f_live.empty and f_frozen.empty
