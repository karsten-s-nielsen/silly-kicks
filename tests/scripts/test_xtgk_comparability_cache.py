"""The cross-provider xt_gk comparability driver on the load seam (ADR-052 D14 / ADR-102).

The xT surface is fit by an EVENTS-ONLY count pass (no frames, no double fetch) and both providers
are scored in ONE resume-before-load ``for_each`` pass; ``--cache-dir`` threads to both loaders. The
old streaming ``_collect`` loop over ``load_matches`` is gone (Rules A/C).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
from _fake_corpus import make_ref

import scripts._xtgk_comparability as xc


def test_score_match_extracts_provider_dist_and_xtgk(monkeypatch):
    """The per-match work: (provider, dist, xt_gk, variant) over scored GK distributions only."""

    def _fake_add_xt_gk(actions, frames, xt):
        return pd.DataFrame(
            {
                "start_x": [0.0, 10.0, 3.0],
                "start_y": [0.0, 0.0, 4.0],
                "end_x": [3.0, 10.0, 3.0],
                "end_y": [4.0, 0.0, 4.0],
                "xt_gk": [0.1, float("nan"), 0.2],  # the NaN row is not a scored GK distribution
                "xt_gk_completion_variant": ["gs", "gs", "gs"],
            }
        )

    monkeypatch.setattr(xc, "add_xt_gk", _fake_add_xt_gk)
    out = xc._score_match("skillcorner", pd.DataFrame(), pd.DataFrame(), object())

    assert list(out.columns) == ["provider", "dist", "xt_gk", "variant"]
    assert len(out) == 2  # the NaN xt_gk row is dropped
    assert (out["provider"] == "skillcorner").all()
    # dist is hypot(end-start); the first scored row is (3,4) -> 5.0
    assert out["dist"].iloc[0] == 5.0


def test_main_threads_cache_dir_to_both_loaders_and_writes_report(monkeypatch, tmp_path):
    """--cache-dir reaches BOTH the events-only fit loader and the full-load scoring source, and the
    report is written. The corpus seam is faked, so no network and no real shards are needed."""
    seen: dict = {}
    ref_gs, ref_sc = make_ref("gradientsports", "g1"), make_ref("skillcorner", "s1")

    def _fake_pining_source(**kw):
        seen["pining"] = kw
        return [ref_gs, ref_sc], (lambda ref: object())

    def _fake_events_only_loader(refs, **kw):
        seen["events"] = kw
        admission = types.SimpleNamespace(digest="ADM", unmeasured_admitted=())
        return (lambda ref: object()), admission

    prov = types.SimpleNamespace(
        counts_digest="CD", admission_digest="ADM", unmeasured_admitted=(), fit_keys=("gradientsports__g1",)
    )
    # for_each is faked to write no shards -> shard_keys empty -> compare sees no overlap.
    fake_res = types.SimpleNamespace(shard_dir=tmp_path, shard_keys=(), failures={})

    monkeypatch.setattr(xc, "pining_source", _fake_pining_source)
    monkeypatch.setattr(xc, "events_only_loader", _fake_events_only_loader)
    monkeypatch.setattr(xc, "xt_count_pass", lambda *a, **kw: types.SimpleNamespace())
    monkeypatch.setattr(xc, "fit_xt_from_count_pass", lambda res, **kw: (object(), prov))
    monkeypatch.setattr(xc, "for_each", lambda *a, **kw: fake_res)
    argv = ["prog", "--cache-dir", "CACHE_SENTINEL", "--shard-root", str(tmp_path), "--out-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)

    assert xc.main() == 0

    assert seen["pining"]["cache_dir"] == Path("CACHE_SENTINEL")
    assert seen["events"]["cache_dir"] == Path("CACHE_SENTINEL")
    assert seen["pining"]["providers"] == ["gradientsports", "skillcorner"]
    report = (tmp_path / "comparability_report.json").read_text(encoding="utf-8")
    assert '"verdict": "insufficient_overlap"' in report
    assert '"counts_digest": "CD"' in report
