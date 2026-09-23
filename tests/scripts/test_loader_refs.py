"""Pining loader: `resolve_cache_dir`, `MatchRef`, `list_match_refs`, `load_match`, wrapper parity.

tests/scripts/ has NO __init__.py; conftest puts scripts/ on sys.path.
"""

from __future__ import annotations

import _loader_pining as lp
import pytest

# ---- Step 1: resolve_cache_dir --------------------------------------------------------------------


def test_resolve_cache_dir_argument_beats_env_beats_unset(tmp_path, monkeypatch):
    monkeypatch.delenv(lp.CORPUS_CACHE_ENV, raising=False)
    assert lp.resolve_cache_dir(None) is None
    monkeypatch.setenv(lp.CORPUS_CACHE_ENV, str(tmp_path / "env"))
    assert lp.resolve_cache_dir(None) == tmp_path / "env"
    assert lp.resolve_cache_dir(tmp_path / "arg") == tmp_path / "arg", "the argument beats the env var"


def test_blank_env_reads_as_unset(monkeypatch):
    monkeypatch.setenv(lp.CORPUS_CACHE_ENV, "   ")
    assert lp.resolve_cache_dir(None) is None, "a blank env value is a typo, not the working directory"


# ---- Step 2: MatchRef + list_match_refs -----------------------------------------------------------


def test_matchref_key_and_artifacts_out_of_identity():
    a = lp.MatchRef("skillcorner", "m1", {"events": "e.csv"})
    b = lp.MatchRef("skillcorner", "m1", {"events": "OTHER.csv"})
    assert a.key == ("skillcorner", "m1")
    assert a == b, "artifacts are excluded from equality -- a ref IS its (provider, match_id)"
    assert hash(a) == hash(b)
    assert "OTHER.csv" not in repr(b) and "events" not in repr(b)


def _fake_manifest(monkeypatch):
    manifests = {
        "skillcorner": [
            {"id": "1886347", "artifacts": {"events": "a_dynamic_events.csv"}},
            {"id": "1021404", "artifacts": {"events": "b_dynamic_events.csv"}},
        ],
        "idsse": [{"id": "DFL-MAT-01", "artifacts": {"tracking": "t.xml"}}],
    }
    monkeypatch.setattr(lp, "_list_matches", lambda provider, tok, base: manifests[provider])
    return manifests


def test_list_match_refs_agrees_with_select_match_ids(monkeypatch):
    _fake_manifest(monkeypatch)
    refs = lp.list_match_refs(providers=["skillcorner", "idsse"])
    assert [r.key for r in refs] == lp.select_match_ids(providers=["skillcorner", "idsse"])
    # the ref carries the manifest's artifact map, so a load needs no second manifest call
    assert refs[0].artifacts == {"events": "a_dynamic_events.csv"}


def test_list_match_refs_honours_selection(monkeypatch):
    _fake_manifest(monkeypatch)
    refs = lp.list_match_refs(providers=["skillcorner"], match_ids={"skillcorner": ["1021404"]})
    assert [r.match_id for r in refs] == ["1021404"]
    capped = lp.list_match_refs(providers=["skillcorner"], max_per_provider=1)
    assert [r.match_id for r in capped] == ["1886347"]


# ---- Step 3: load_match S1 exclusion ------------------------------------------------------------


class _Report:
    def __init__(self, *, excluded: bool, reason: str = "") -> None:
        self.geometry_excluded = excluded
        self.geometry_reason = reason
        self.player_off_pitch_rate = 0.34139
        self.ball_off_pitch_rate = 0.02


def test_load_match_raises_matchexcluded_with_both_rates_on_the_s1_gate(monkeypatch):
    def _fake_retry(provider, match_id, artifacts, tok, base, tracking_limit, *, cache_dir=None, events_only=False):
        return "act", "frm", "home", None, _Report(excluded=True, reason="player off-pitch 0.341 > 0.005")

    monkeypatch.setattr(lp, "_build_match_with_retry", _fake_retry)
    with pytest.raises(lp.MatchExcluded) as exc:
        lp.load_match(lp.MatchRef("skillcorner", "bad"), events_only=False)
    assert issubclass(lp.MatchExcluded, __import__("_item_outcome").ItemExcluded)
    assert exc.value.details == {"player_off_pitch_rate": 0.34139, "ball_off_pitch_rate": 0.02}
    assert "off-pitch" in exc.value.reason


def test_load_match_returns_a_seven_field_loaded_match(monkeypatch):
    def _fake_retry(provider, match_id, artifacts, tok, base, tracking_limit, *, cache_dir=None, events_only=False):
        return "act", "frm", "home", "va", _Report(excluded=False)

    monkeypatch.setattr(lp, "_build_match_with_retry", _fake_retry)
    m = lp.load_match(lp.MatchRef("skillcorner", "ok"), events_only=False)
    # one tuple compare (the fields are string sentinels) -- keeps the DataFrame-typed fields out of a
    # boolean context, which pyright rejects.
    assert (m.provider, m.match_id, m.actions, m.frames, m.home_team_id, m.visible_area) == (
        "skillcorner",
        "ok",
        "act",
        "frm",
        "home",
        "va",
    )
    assert m.report is not None


# ---- Step 5: wrapper parity vs the frozen 4ac26d0 bodies -----------------------------------------


class _Rep:
    def __init__(self, excluded: bool) -> None:
        self.geometry_excluded = excluded
        self.geometry_reason = "S1"
        self.player_off_pitch_rate = 0.34
        self.ball_off_pitch_rate = 0.02


def _install_parity_network(monkeypatch):
    manifests = {
        "skillcorner": [{"id": "good", "artifacts": {}}, {"id": "bad", "artifacts": {}}],
        "statsbomb": [{"id": "sb1", "artifacts": {}}],
    }
    monkeypatch.setattr(lp, "_list_matches", lambda provider, tok, base: manifests[provider])

    def _fake_retry(provider, match_id, artifacts, tok, base, tracking_limit, *, cache_dir=None, events_only=False):
        rep = _Rep(excluded=(match_id == "bad")) if provider == "skillcorner" else None
        return f"act-{match_id}", f"frm-{match_id}", f"home-{match_id}", f"va-{match_id}", rep

    monkeypatch.setattr(lp, "_build_match_with_retry", _fake_retry)


def test_load_matches_wrapper_is_byte_identical_to_4ac26d0(monkeypatch, capsys):
    from _frozen_loader_4ac26d0 import load_matches_4ac26d0

    _install_parity_network(monkeypatch)
    live = list(lp.load_matches(providers=["skillcorner"]))
    live_err = capsys.readouterr().err
    frozen = list(load_matches_4ac26d0(providers=["skillcorner"]))
    frozen_err = capsys.readouterr().err
    assert live == frozen, "the wrapper's tuples drifted from 4ac26d0"
    assert live_err == frozen_err, "the EXCLUDED line / summary drifted from 4ac26d0"
    # non-vacuous: the bad match WAS excluded and the summary said so.
    assert (
        [m for _p, m, *_ in live] == ["good"] and "EXCLUDED skillcorner/bad" in live_err and "excluded 1/2" in live_err
    )


def test_load_statsbomb_matches_wrapper_is_byte_identical_to_4ac26d0(monkeypatch, capsys):
    from _frozen_loader_4ac26d0 import load_statsbomb_matches_4ac26d0

    _install_parity_network(monkeypatch)
    live = list(lp.load_statsbomb_matches())
    capsys.readouterr()
    frozen = list(load_statsbomb_matches_4ac26d0())
    capsys.readouterr()
    assert live == frozen
    assert [t[:2] for t in live] == [("statsbomb", "sb1")] and live[0][5] == "va-sb1"  # 6-tuple w/ visible_area


# ---- Step 7: import direction -------------------------------------------------------------------


def test_loader_does_not_import_the_orchestration_seam():
    """The loader raises `MatchExcluded` ⊂ `ItemExcluded`, so it imports the leaf `_item_outcome` -- but
    NEVER `_driver` (spec §4.1: an adapter must not reach for the orchestration seam)."""
    import ast
    import inspect

    roots = set()
    for node in ast.walk(ast.parse(inspect.getsource(lp))):
        if isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.replace("scripts.", ""))
        elif isinstance(node, ast.Import):
            roots.update(a.name.replace("scripts.", "") for a in node.names)
    assert "_item_outcome" in roots
    assert "_driver" not in roots


# ---- Step 6: the _fake_corpus helper (consumed by Tasks 9-15) ------------------------------------


def test_fake_corpus_spyloader_records_and_raises():
    from _fake_corpus import SpyLoader, make_loaded, make_ref

    good, bad, boom = make_ref("gs", "1"), make_ref("gs", "2"), make_ref("gs", "3")
    spy = SpyLoader({good.key: make_loaded("gs", "1")}, fail=[boom.key], exclude={bad.key: "S1 gate"})
    assert spy(good, events_only=False).match_id == "1"
    assert spy.calls == [(("gs", "1"), {"events_only": False})]  # records key + kwargs
    with pytest.raises(lp.MatchExcluded, match="S1 gate"):
        spy(bad, events_only=False)
    with pytest.raises(RuntimeError):
        spy(boom, events_only=False)


def test_install_fake_corpus_refuses_a_stream_call(monkeypatch):
    import types

    from _fake_corpus import SpyLoader, install_fake_corpus, make_ref

    # SimpleNamespace stands in for a migrated driver module: it exposes `list_match_refs` + `load_match`
    # (which install patches, raising=True) plus a stream wrapper it still has (which install must refuse).
    mod = types.SimpleNamespace(
        list_match_refs=lp.list_match_refs,
        load_match=lp.load_match,
        load_matches=lambda **_kw: iter(()),
    )
    install_fake_corpus(monkeypatch, mod, refs=[make_ref("gs", "1")], loader=SpyLoader({}))
    assert [r.key for r in mod.list_match_refs()] == [("gs", "1")]
    with pytest.raises(AssertionError, match="still streams"):
        list(mod.load_matches())
