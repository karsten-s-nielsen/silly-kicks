"""CI-safe tests for scripts/validate_xs_probe.py:
- the pure helpers (re_gate, _render, _write) on a synthetic metrics dict;
- the run() ORCHESTRATION via a monkeypatched load_matches -- so the per-match loop, delta pooling,
  per_match accumulation, and empty-corpus SystemExit are covered by CI, not first exercised by the
  expensive owner run. No data, no network.

tests/scripts/ has NO __init__.py on purpose (it mirrors the sibling script tests): adding one makes
it a package that shadows the top-level `scripts` namespace and breaks `import scripts.<x>` elsewhere.
"""

from __future__ import annotations

import json
import sys
from typing import ClassVar

import pandas as pd
import pytest

import scripts.validate_xs_probe as mod


def _fake_metrics(verdict="fail"):
    return {
        "arm": "xs",
        "probe": {
            "verdict": verdict,
            "n_frames_used": 123,
            "gated_band_median": 0.003,
            "nearest_def_median": 0.0006,
            "placebo_p95": 0.02,
            "gated_band_n": 150,
            "gated_band_zero_fraction": 0.1,
            "off_pitch_control_fraction": 0.0,
            "dose_response_rho": 0.4,
            "dose_response_p": 0.03,
        },
        "regate_verdict": "gated_clean_fail",
        "entanglement": "inside_band",
        "reconciliation": {
            "total_targets": 200,
            "n_frames_used": 123,
            "n_distinct_games": 40,
            "gated_band_n": 150,
            "targets_to_used_drop_frac": 0.385,
        },
        "rule_constants": {"ratio": 2.0, "min_band_n": 100},
        "corpus": {"n_matches": 5, "match_ids": ["a"] * 5},
        "per_match": [],
        "seed": 42,
        "tracking_limit": None,
        "rng_discipline": "per-match placebo streams",
        "baseline_commit": "deadbeef",
    }


def test_re_gate_maps_fail_inside_band_to_gated_clean_fail():
    # Surfaces the real regate_verdict mapping; if it differs, the report wording must follow it.
    assert mod.re_gate("fail", "inside_band") == "gated_clean_fail"


def test_render_mentions_verdict_regate_and_reconciliation():
    out = mod._render(_fake_metrics(verdict="fail"))
    assert isinstance(out, str) and "fail" in out and "Re-gate" in out and "reconciliation" in out.lower()


def test_render_shows_na_on_unmeasurable_branch():
    # The early-return branch omits the prongs; they must render "n/a", not "None".
    m = _fake_metrics(verdict="unmeasurable_at_dose")
    for k in ("gated_band_median", "nearest_def_median", "placebo_p95", "dose_response_rho", "dose_response_p"):
        m["probe"][k] = None
    out = mod._render(m)
    assert "n/a (unmeasurable)" in out


def test_write_produces_both_files(tmp_path):
    mod._write(tmp_path, _fake_metrics())
    assert json.loads((tmp_path / "metrics.json").read_text())["arm"] == "xs"
    assert (tmp_path / "report.md").read_text().startswith("# TF-19 PR-3b")


class _FakeReport:
    n_frames_in = 1
    n_frames_scored = 1
    drop_reasons: ClassVar[dict] = {}


def _fake_deltas(game_id):
    # minimal tidy-deltas shape: two gk rows so pooling across matches is observable
    return pd.DataFrame(
        {
            "game_id": [game_id, game_id],
            "period_id": [1, 1],
            "frame_id": [10, 20],
            "actor_role": ["gk", "gk"],
            "replicate": [0, 0],
            "displacement_m": [2.5, 3.0],
            "delta_p": [0.01, 0.02],
            "ghost_clamped": [False, False],
            "ghost_out_of_box": [False, False],
            "moved_off_pitch": [False, False],
        }
    )


def test_run_pools_two_matches_with_DISTINCT_games(monkeypatch, tmp_path):
    # Each synthetic match must contribute a DISTINCT game_id, or the "pooling" assertion is vacuous.
    # Key the fake deltas off an iterator, NOT off the always-non-None `targets` arg.
    _gids = iter((100, 101))

    def fake_load_matches(**kwargs):
        for mid in ("m0", "m1"):
            yield ("gradientsports", mid, pd.DataFrame(), pd.DataFrame(), 1)

    fake_loader = type(sys)("_loader_pining")
    monkeypatch.setitem(sys.modules, "_loader_pining", fake_loader)
    monkeypatch.setattr(fake_loader, "load_matches", fake_load_matches, raising=False)
    monkeypatch.setattr(mod.GhostGkModel, "from_variant", staticmethod(lambda v="default": object()))
    monkeypatch.setattr(mod.XShotOccurrenceModel, "from_variant", staticmethod(lambda v="default": object()))
    monkeypatch.setattr(mod, "build_ghost_frames", lambda frames, **k: (None, pd.DataFrame(), _FakeReport()))
    monkeypatch.setattr(mod, "provenance_to_targets", lambda prov, **k: pd.DataFrame({"x": [0]}))
    monkeypatch.setattr(mod, "substitution_deltas", lambda *a, **k: _fake_deltas(next(_gids)))  # distinct id per call
    monkeypatch.setattr(mod, "evaluate_xs_probe", lambda d: {"verdict": "unmeasurable_at_dose", "gated_band_n": 0})

    m = mod.run(tmp_path, entanglement="inside_band", seed=7)
    assert m["corpus"]["n_matches"] == 2
    assert m["probe"]["verdict"] == "unmeasurable_at_dose"
    # POOLING IS REAL: run() re-computes n_frames_used over the POOLED deltas (driver, not the stubbed
    # evaluator) -- 2 frames x 2 DISTINCT games = 4 unique (game,period,frame) gk rows.
    assert m["probe"]["n_frames_used"] == 4
    assert m["reconciliation"]["n_distinct_games"] == 2  # R1 guard sees two games -> no collision warning
    assert m["reconciliation"]["total_targets"] == 2
    assert m["seed"] == 7 and (tmp_path / "metrics.json").exists()


def test_run_empty_corpus_raises_systemexit(monkeypatch, tmp_path):
    fake_loader = type(sys)("_loader_pining")
    monkeypatch.setitem(sys.modules, "_loader_pining", fake_loader)
    monkeypatch.setattr(fake_loader, "load_matches", lambda **k: iter(()), raising=False)
    monkeypatch.setattr(mod.GhostGkModel, "from_variant", staticmethod(lambda v="default": object()))
    monkeypatch.setattr(mod.XShotOccurrenceModel, "from_variant", staticmethod(lambda v="default": object()))
    with pytest.raises(SystemExit):
        mod.run(tmp_path)
