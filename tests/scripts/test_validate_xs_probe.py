"""CI-safe tests for scripts/validate_xs_probe.py:
- the pure helpers (re_gate, _render, _write) on a synthetic two-variant metrics dict;
- the run() ORCHESTRATION via a monkeypatched load_matches -- so the per-match loop, per-variant
  delta pooling, per_match accumulation, and empty-corpus SystemExit are covered by CI, not first
  exercised by the expensive owner run. No data, no network.

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


def _fake_probe(verdict="fail"):
    return {
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
    }


def _fake_metrics(v1_verdict="no_valid_placebo", v2_verdict="pass"):
    return {
        "arm": "xs",
        "variants": {
            "v1": {
                "probe": _fake_probe(v1_verdict),
                "regate_verdict": "unmeasurable_at_dose",
                "rule_constants": {"ratio": 2.0, "min_band_n": 100},
            },
            "v2": {
                "probe": {
                    **_fake_probe(v2_verdict),
                    "attacker_diag_p95": 0.05,
                    "rule": "xs-dose-banded-v2",
                    "placebo_pool": "model_relevant_def",
                },
                "regate_verdict": "joins_with_caveat",
                "rule_constants": {"ratio": 2.0, "min_band_n": 100, "placebo_pool": "model_relevant_def"},
            },
        },
        "entanglement": "inside_band",
        "reconciliation": {
            "total_targets": 200,
            "n_frames_used": 123,
            "n_distinct_games": 40,
            "gated_band_n": 150,
            "targets_to_used_drop_frac": 0.385,
        },
        "corpus": {"n_matches": 5, "match_ids": ["a"] * 5},
        "per_match": [],
        "seed": 42,
        "tracking_limit": None,
        "rng_discipline": "per-match placebo streams",
        "lock_commit": "1abc",
        "run_commit": "deadbeef",
    }


def test_re_gate_maps_fail_inside_band_to_gated_clean_fail():
    # Surfaces the real regate_verdict mapping; if it differs, the report wording must follow it.
    assert mod.re_gate("fail", "inside_band") == "gated_clean_fail"


def test_render_shows_both_variants_and_the_lock_commit():
    out = mod._render(_fake_metrics(v1_verdict="no_valid_placebo", v2_verdict="pass"))
    assert "v1" in out and "v2" in out
    assert "no_valid_placebo" in out and "pass" in out
    assert "re-gate" in out.lower()
    assert "reconciliation" in out.lower()
    assert "1abc" in out  # lock commit is auditable in the report


def test_render_shows_na_on_unmeasurable_branch():
    # The early-return branch omits the prongs; they must render "n/a", not "None".
    m = _fake_metrics(v2_verdict="unmeasurable_at_dose")
    for k in ("gated_band_median", "nearest_def_median", "placebo_p95", "dose_response_rho", "dose_response_p"):
        m["variants"]["v2"]["probe"][k] = None
    out = mod._render(m)
    assert "n/a (unmeasurable)" in out


def test_write_produces_both_files(tmp_path):
    mod._write(tmp_path, _fake_metrics())
    assert json.loads((tmp_path / "metrics.json").read_text())["arm"] == "xs"
    assert (tmp_path / "report.md").read_text().startswith("# TF-19")


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


def test_run_pools_two_matches_and_scores_both_variants(monkeypatch, tmp_path):
    # 2 matches x 2 variants = 4 substitution_deltas calls, each fed a DISTINCT game_id off an iterator
    # (not the always-non-None targets arg) so pooling is observable per variant.
    _gids = iter((100, 101, 200, 201))

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
    monkeypatch.setattr(mod, "substitution_deltas", lambda *a, **k: _fake_deltas(next(_gids)))
    monkeypatch.setattr(mod, "evaluate_xs_probe", lambda d: {"verdict": "unmeasurable_at_dose", "gated_band_n": 0})

    m = mod.run(tmp_path, entanglement="inside_band", seed=7, lock_commit="1abc")
    assert m["corpus"]["n_matches"] == 2
    assert set(m["variants"]) == {"v1", "v2"}
    assert m["variants"]["v1"]["probe"]["verdict"] == "unmeasurable_at_dose"
    assert m["variants"]["v2"]["probe"]["verdict"] == "unmeasurable_at_dose"
    assert m["variants"]["v2"]["probe"]["placebo_pool"] == "model_relevant_def"  # v2-only enrichment
    assert m["lock_commit"] == "1abc"
    assert m["seed"] == 7 and (tmp_path / "metrics.json").exists()


def test_run_single_variant_v1_only(monkeypatch, tmp_path):
    # --variant v1 runs one delta-compute per match and produces no v2 block (honest-framing guard).
    _gids = iter((100, 200))

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
    monkeypatch.setattr(mod, "substitution_deltas", lambda *a, **k: _fake_deltas(next(_gids)))
    monkeypatch.setattr(mod, "evaluate_xs_probe", lambda d: {"verdict": "unmeasurable_at_dose", "gated_band_n": 0})

    m = mod.run(tmp_path, variant="v1", seed=7)
    assert set(m["variants"]) == {"v1"}
    assert "The honest framing" not in (tmp_path / "report.md").read_text()  # v2-only prose gated


def test_run_empty_corpus_raises_systemexit(monkeypatch, tmp_path):
    fake_loader = type(sys)("_loader_pining")
    monkeypatch.setitem(sys.modules, "_loader_pining", fake_loader)
    monkeypatch.setattr(fake_loader, "load_matches", lambda **k: iter(()), raising=False)
    monkeypatch.setattr(mod.GhostGkModel, "from_variant", staticmethod(lambda v="default": object()))
    monkeypatch.setattr(mod.XShotOccurrenceModel, "from_variant", staticmethod(lambda v="default": object()))
    with pytest.raises(SystemExit):
        mod.run(tmp_path)


def _patch_probe_env(monkeypatch, *, calls):
    """The `run()` fixture above, with the loader RECORDING every match it is asked to build."""
    _gids = iter(range(100, 200))

    def fake_load_matches(**kwargs):
        for mid in ("m0", "m1"):
            yield ("gradientsports", mid, pd.DataFrame(), pd.DataFrame(), 1)

    def fake_build(frames, **k):
        calls.append("build")
        return (None, pd.DataFrame(), _FakeReport())

    fake_loader = type(sys)("_loader_pining")
    monkeypatch.setitem(sys.modules, "_loader_pining", fake_loader)
    monkeypatch.setattr(fake_loader, "load_matches", fake_load_matches, raising=False)
    monkeypatch.setattr(mod.GhostGkModel, "from_variant", staticmethod(lambda v="default": object()))
    monkeypatch.setattr(mod.XShotOccurrenceModel, "from_variant", staticmethod(lambda v="default": object()))
    monkeypatch.setattr(mod, "build_ghost_frames", fake_build)
    monkeypatch.setattr(mod, "provenance_to_targets", lambda prov, **k: pd.DataFrame({"x": [0]}))
    monkeypatch.setattr(mod, "substitution_deltas", lambda *a, **k: _fake_deltas(next(_gids)))
    monkeypatch.setattr(mod, "evaluate_xs_probe", lambda d: {"verdict": "unmeasurable_at_dose", "gated_band_n": 0})


def test_a_RESUMED_probe_run_recomputes_NOTHING_and_still_reports_the_corpus(monkeypatch, tmp_path):
    """THE motivating driver: ~80 matches, 14 hours serial, one write at the end.

    Two properties in one place because they fail independently. (1) The second pass must not
    re-enter `build_ghost_frames` -- that is resume. (2) It must still report `n_matches == 2` and a
    full `per_match` block -- that is the counters sidecar, without which a resumed pass writes a
    cited artifact claiming a corpus of zero matches while every shard sits on disk beside it.
    """
    calls: list[str] = []
    _patch_probe_env(monkeypatch, calls=calls)
    first = mod.run(tmp_path, seed=7)
    assert calls == ["build", "build"]
    assert first["corpus"]["n_matches"] == 2

    calls.clear()
    second = mod.run(tmp_path, seed=7)
    assert calls == [], f"a resumed pass re-entered the engine for {calls}"
    assert second["corpus"]["n_matches"] == 2, "resume lost the corpus record"
    assert [m["match_id"] for m in second["per_match"]] == [m["match_id"] for m in first["per_match"]]
    assert [m["n_targets"] for m in second["per_match"]] == [m["n_targets"] for m in first["per_match"]]
    assert second["reconciliation"]["total_targets"] == first["reconciliation"]["total_targets"]


def test_per_match_drop_reasons_survive_the_flat_counter_round_trip():
    """`_merge_counters` carries ints and FLAT {str: int} dicts only, so the per-match breakdown is
    encoded on a composite key. `metrics["per_match"]` is a published field of a cited artifact --
    coarsening it corpus-wide would be a schema break, so the round trip is pinned."""
    flat = {}
    flat.update(mod._flatten_by_match("m0", {"no_gk": 3, "ball_dead": 1}))
    flat.update(mod._flatten_by_match("m1", {"no_gk": 2}))
    records = mod._per_match_records({"drop_reasons_by_match": flat, "n_targets_by_match": {"m0": 5, "m1": 0}})
    assert [r["match_id"] for r in records] == ["m0", "m1"]
    assert records[0]["drop_reasons"] == {"no_gk": 3, "ball_dead": 1}
    assert records[1]["drop_reasons"] == {"no_gk": 2}
    # A zero-target match KEEPS its record: `n_contributing` counts exactly those, and its shard is
    # empty so the combined frame cannot recover it.
    assert records[1]["n_targets"] == 0


def test_an_AMBIGUOUS_match_id_is_REFUSED_rather_than_mis_attributed():
    """The other side: a component holding the separator would split wrong and silently attribute
    one match's drop reasons to another. Same discipline as `_driver.join_key`."""
    with pytest.raises(ValueError, match="split ambiguously"):
        mod._flatten_by_match("a::b", {"no_gk": 1})
