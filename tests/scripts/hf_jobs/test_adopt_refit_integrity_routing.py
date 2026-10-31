"""Task 3 (adopt-for-refit seam): verify routes the `_integrity`/`reproduces` gate by campaign mode.

A `refit` skips `_integrity` (new corpus/weights never reproduce the committed model); a `rematerialize`
keeps it (same model/corpus, values unchanged — the O-7 `_OWED` regen must keep its teeth)."""

from __future__ import annotations

import json
import types

from scripts.hf_jobs import adopt as ad


def _setup(monkeypatch, tmp_path, *, integrity_go: bool):
    """Stub verify()'s heavy collaborators around an xshot instance; return the _integrity call-spy list."""
    inst = types.SimpleNamespace(id="inst1", job="xshot_occurrence", tier="public", variant="default", args=())
    (tmp_path / "inst1").mkdir()
    (tmp_path / "inst1.record.json").write_text(json.dumps({"instance": "inst1"}), encoding="utf-8")
    (tmp_path / "inst1" / "metrics.json").write_text(json.dumps({"shipped_variant": "public"}), encoding="utf-8")
    ref = types.SimpleNamespace(kind="wheel", id="a1", job="xshot_occurrence", variant="default", tier="public")
    monkeypatch.setattr(ad, "instances", lambda c, commit, replica=None: [inst])
    monkeypatch.setattr("scripts.hf_jobs.registry.spec_for", lambda job: types.SimpleNamespace())
    monkeypatch.setattr(ad, "artifacts_for", lambda inst, man: [ref])
    monkeypatch.setattr(ad, "load_check", lambda *a, **k: None)
    monkeypatch.setattr(ad, "_certify", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(ad, "_committed_metrics", lambda *a, **k: {})
    monkeypatch.setattr(ad, "expectation_violations", lambda *a, **k: [])
    calls: list = []

    def integrity_spy(*a, **k):
        calls.append(1)
        return {"go": integrity_go}

    monkeypatch.setattr(ad, "_integrity", integrity_spy)
    return inst, calls


def _verify(c_mode: str, tmp_path):
    c = types.SimpleNamespace(name="tf46_refit", bucket="silly-kicks/hf-refit-scratch", mode=c_mode)
    return ad.verify(c, "deadbeef", staging=tmp_path)  # pyright: ignore[reportArgumentType]


def test_refit_skips_integrity(monkeypatch, tmp_path):
    _inst, calls = _setup(monkeypatch, tmp_path, integrity_go=False)
    rep = _verify("refit", tmp_path)
    assert calls == []  # _integrity NOT called in refit mode
    assert rep["instances"]["inst1"]["ok"] is True  # certify ok, integrity not a gate


def test_rematerialize_runs_integrity(monkeypatch, tmp_path):
    _inst, calls = _setup(monkeypatch, tmp_path, integrity_go=True)
    rep = _verify("rematerialize", tmp_path)
    assert calls == [1]  # _integrity IS called (keeps its teeth)
    assert rep["instances"]["inst1"]["ok"] is True


def test_rematerialize_integrity_still_fails_on_drift(monkeypatch, tmp_path):
    _inst, calls = _setup(monkeypatch, tmp_path, integrity_go=False)  # reproduces=False ⇒ go=False
    rep = _verify("rematerialize", tmp_path)
    assert calls == [1]
    assert rep["instances"]["inst1"]["ok"] is False  # re-materialize teeth: a drift fails the instance
