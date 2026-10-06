"""TF-58 Task 23 (D3, §8.5): ``validate_team_coordination`` -- pure reduce kernels + driver wiring.

The corpus passes are exercised with a fake corpus (resume-before-load); the report kernels are pinned directly.
Thresholds are referenced from ``_coordination_thresholds`` -- an AST guard proves the driver inlines none.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import validate_team_coordination as d
from _fake_corpus import SpyLoader, make_loaded, make_ref

_CLEAN = {"commit": "0" * 40, "dirty": False, "tree_state": "clean"}


@pytest.fixture(autouse=True)
def _private_manifest(monkeypatch):
    """ADR-038 label lookups never reach the network here: the stub manifest lists nothing, so every match is
    private (fail-closed)."""
    monkeypatch.setattr("scripts._loader_pining.match_visibility", lambda providers, *, token=None, base_url=None: {})


def _args(tmp_path, **over):
    base = dict(
        out=str(tmp_path),
        providers=("sportec",),
        match_ids_json=None,
        max_matches=None,
        cache_dir=None,
        token=None,
        allow_dirty=True,
        list_matches=False,
        corpus_json=None,
        derivation=None,
        calibration=None,
        in_package_params=True,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _write_share(dest, name, df, *, tag="all", stage_seconds=None, n_attempted=None, **extra):
    """One worker's share of a D3 pass, in the layout ``write_worker_partial`` writes (table + manifest)."""
    keys = []
    if len(df) and "provider" in df.columns:
        ids = df["match_id"] if "match_id" in df.columns else df["game_id"]
        keys = sorted({f"{p}__{m}" for p, m in zip(df["provider"], ids, strict=True)})
    df.to_parquet(dest / f"{name}.{tag}.parquet", index=False)
    manifest = {
        "generation": "g1",
        "n_attempted": len(keys) if n_attempted is None else n_attempted,
        "stage_seconds": stage_seconds or {},
        "listed": keys,
        "produced": keys,
        "excluded_keys": [],
        "failed_keys": [],
        "run_commit": _CLEAN["commit"],
        "run_tree_dirty": False,
        "run_tree_state": "clean",
        "params_source": "in_package",
        **extra,
    }
    (dest / f"manifest_{name}.{tag}.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_all_shares(dest, metrics_df, **metrics_extra):
    _write_share(dest, "metrics", metrics_df, **metrics_extra)
    _write_share(dest, "stoppage", pd.DataFrame())
    _write_share(dest, "occlusion", pd.DataFrame(), width_m=40.0)


# --------------------------------------------------------------------------- stoppage interval agreement (D20)
def test_stoppage_interval_agreement_precision_recall_iou():
    # ball_state truth: two dead spans; events: one exact hit, one miss (no overlap), so precision 1/2, recall 1/2.
    key = ("g", 1)
    ball_state: dict[tuple[object, object], np.ndarray] = {key: np.array([[10.0, 40.0], [100.0, 140.0]])}
    events: dict[tuple[object, object], np.ndarray] = {key: np.array([[12.0, 38.0], [200.0, 230.0]])}
    ag = d.stoppage_interval_agreement(events, ball_state)
    assert ag["n_event"] == 2 and ag["n_ball_state"] == 2
    assert ag["precision"] == 0.5  # one of two event intervals overlaps a ball-state interval
    assert ag["recall"] == 0.5  # one of two ball-state intervals is recalled
    assert 0.0 < ag["mean_iou"] <= 1.0  # the single matched pair has a real IoU


def test_stoppage_interval_agreement_empty_is_nan_not_crash():
    ag = d.stoppage_interval_agreement({}, {})
    assert np.isnan(ag["precision"]) and np.isnan(ag["recall"]) and ag["n_event"] == 0


# --------------------------------------------------------------------------- liveness (real-data)
def test_liveness_block_flags_a_dead_column():
    rows = []
    for i in range(4):
        rows.append(
            {
                "table": "spectral",
                "game_id": "g",
                "period_id": 1,
                "window_kind": "period",
                "window_id": i,
                "team_id": "A",
                "signal": "centroid_x",
                "coord_median_freq_cpm": 0.4 + 0.01 * i,  # live: varies
                "coord_duration_s": 90.0,  # dead: constant
            }
        )
    live = d.liveness_block(d.split_tables(pd.DataFrame(rows)))
    assert live["columns"]["spectral.coord_median_freq_cpm"]["live"] is True
    assert live["columns"]["spectral.coord_duration_s"]["live"] is False
    assert live["columns"]["spectral.coord_duration_s"]["reason"] == "constant"
    assert "spectral.coord_duration_s" in live["dead"]
    assert "reason" not in live["columns"]["spectral.coord_median_freq_cpm"]  # a live column carries no finding


# --------------------------------------------------------------------------- hypotheses: recorded, not dropped
def _failing_spectral_metrics_df():
    # median frequency ABOVE the ceiling in every half -> H4 fails (below_share 0 < H4_BELOW_CEIL_SHARE).
    rows = []
    for t in range(3):
        for period in (1, 2):
            rows.append(
                {
                    "provider": "skillcorner",
                    "table": "spectral",
                    "game_id": f"g{t}",
                    "period_id": period,
                    "window_kind": "period",
                    "window_id": 0,
                    "team_id": f"T{t}",
                    "signal": "centroid_x",
                    "coord_median_freq_cpm": 5.0,  # far above the 1 cpm ceiling
                }
            )
    return pd.DataFrame(rows)


def test_failed_hypothesis_recorded_not_dropped():
    report = d.build_report(_failing_spectral_metrics_df(), pd.DataFrame(), pd.DataFrame(), _CLEAN)
    assert set(report["hypotheses"]) == {"H1", "H2", "H3", "H4", "H5", "H6", "H7"}
    assert report["hypotheses"]["H4"]["pass"] is False  # a real failure, present -- not dropped
    assert report["gated_pass"] is False


def test_build_report_emits_the_per_construct_section_not_the_old_per_family_blocks():
    # A-09: reliability/poolability/coverage are now the per-construct `constructs` list + honesty + summary.
    report = d.build_report(_failing_spectral_metrics_df(), pd.DataFrame(), pd.DataFrame(), _CLEAN)
    assert report["schema_version"] == d._METRICS_SCHEMA_VERSION
    assert isinstance(report["constructs"], list)
    assert {c["column"] for c in report["constructs"]} == {"coord_median_freq_cpm"}  # the one scored spectral metric
    for c in report["constructs"]:
        assert set(c) >= {
            "column",
            "construct_key",
            "unit",
            "kind",
            "reliability",
            "diagnostics",
            "split_mode",
            "deciles",
            "poolability",
            "provider",
        }
    assert report["honesty"] and len(report["honesty"]) == 2
    assert "coord_median_freq_cpm" in report["column_summary"]
    assert "run_tree_hash" in report  # A-55
    assert "d2_gate_population" in report  # A-35
    for stale in ("reliability", "poolability", "coverage_stratification"):
        assert stale not in report


# --------------------------------------------------------------------------- stage timing (§7.15 cost model)
def test_stage_timing_summarises_the_combined_passes():
    # `n_attempted` counts work done (a resumed skip did none); `compute` is the R8 per-stage compute counter.
    timings = d._stage_timing({"metrics": {"stage_seconds": {"corpus": 90.0, "compute": 60.0}, "n_attempted": 3}})
    assert timings["metrics"]["n_matches"] == 3
    assert timings["metrics"]["seconds_per_match"] == 30.0  # incl. load/download
    assert timings["metrics"]["compute_seconds_per_match"] == 20.0  # the §7.15 budget comparand
    assert "stoppage" not in timings


def test_reduce_includes_the_stage_timing_summary(tmp_path):
    _write_all_shares(
        tmp_path, _failing_spectral_metrics_df(), stage_seconds={"corpus": 20.0, "compute": 12.0}, n_attempted=2
    )
    d._reduce(_args(tmp_path), _CLEAN)
    report = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert report["stage_timing"]["metrics"]["seconds_per_match"] == 10.0
    assert report["stage_timing"]["metrics"]["compute_seconds_per_match"] == 6.0
    assert report["population"]["metrics"]["n_workers"] == 1
    # review m8: metrics.json must carry the ADR-038 corpus_visibility label, like the other three artifacts
    assert report["corpus_visibility"] and isinstance(report["corpus_visibility"], str)


def test_reduce_refuses_a_missing_pass(tmp_path):
    # every leg is part of the validation artifact: a reduce without the stoppage pass refuses, never reports less
    _write_share(tmp_path, "metrics", _failing_spectral_metrics_df())
    _write_share(tmp_path, "occlusion", pd.DataFrame(), width_m=40.0)
    with pytest.raises(SystemExit, match="no worker manifest for pass 'stoppage'"):
        d._reduce(_args(tmp_path), _CLEAN)
    assert not (tmp_path / "metrics.json").exists()


def test_reduce_refuses_shares_computed_with_different_params(tmp_path):
    # M-5: one artifact pair for the whole corpus -- two workers that read different derivations are refused
    df = _failing_spectral_metrics_df()
    _write_share(tmp_path, "metrics", df[df["game_id"] == "g0"], tag="w0", derivation_sha256="aaa")
    _write_share(tmp_path, "metrics", df[df["game_id"] != "g0"], tag="w1", derivation_sha256="bbb")
    _write_share(tmp_path, "stoppage", pd.DataFrame())
    _write_share(tmp_path, "occlusion", pd.DataFrame(), width_m=40.0)
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({"skillcorner": ["g0", "g1", "g2"]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="disagree on 'derivation_sha256'"):
        d._reduce(_args(tmp_path, corpus_json=str(corpus), providers=("skillcorner",)), _CLEAN)


# --------------------------------------------------------------------------- params: the artifact handoff (M-5)
def test_params_resolver_requires_the_artifacts_unless_told_otherwise(tmp_path):
    with pytest.raises(SystemExit, match="--derivation"):
        d.params_resolver(_args(tmp_path, in_package_params=False))
    params_for, src = d.params_resolver(_args(tmp_path))  # the dev opt-out says so
    assert src == {"params_source": "in_package"} and params_for("sportec") == d.CoordinationParams.for_provider(
        "sportec"
    )


def test_params_resolver_reads_the_artifact_pair_and_records_their_digests(tmp_path):
    import _coordination_params_codegen as cg

    derivation = {"pooled": {**dict(cg.INTERIM_BASE), "welch_segment_s": 120.0}, "providers": {}}
    calibration = {"confirmation": {"gate_cleared": False}, "moved_multipliers": {}}
    (tmp_path / "derivation.json").write_text(json.dumps(derivation), encoding="utf-8")
    (tmp_path / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
    args = _args(
        tmp_path,
        in_package_params=False,
        derivation=str(tmp_path / "derivation.json"),
        calibration=str(tmp_path / "calibration.json"),
    )
    params_for, src = d.params_resolver(args)
    assert params_for("sportec").welch_segment_s == 120.0  # the artifact's value, not the committed module's
    assert src["params_source"] == "artifacts" and len(src["derivation_sha256"]) == 64


# --------------------------------------------------------------------------- reduce: every hypothesis + AST guard
def test_reduce_reports_every_hypothesis(tmp_path):
    _write_all_shares(tmp_path, _failing_spectral_metrics_df())
    d._reduce(_args(tmp_path), _CLEAN)
    report = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert set(report["hypotheses"]) == {"H1", "H2", "H3", "H4", "H5", "H6", "H7"}
    assert report["corpus_visibility"] == "sc_extended"  # ADR-038: SkillCorner only, nothing public
    assert (tmp_path / "report.md").is_file()


def test_driver_inlines_no_hypothesis_threshold_literal():
    """Plan Task 23: the driver has NO numeric literal equal to a threshold constant -- every threshold is
    referenced via ``thr.<NAME>`` (TF58-IMPL-08: guard ints too, not only floats).

    A FLOAT literal equal to any threshold is flagged. An INT literal is flagged only at a non-trivial magnitude
    (``|v| >= 2``): ``0``/``1`` are universal structural constants (array indices, counts) and the sole threshold
    colliding with them (``1.0``) is still caught in its float form -- so an inlined ``25``/``30``/``95``/``10`` is
    caught while ``interval[1]`` is not a false positive.
    """
    import scripts._coordination_thresholds as thr

    threshold_values = {float(v) for v in vars(thr).values() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    guarded_ints = {v for v in threshold_values if v == int(v) and abs(v) >= 2}
    source = Path(d.__file__).read_text(encoding="utf-8")
    offenders = []
    for node in ast.walk(ast.parse(source)):
        if not (
            isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
        ):
            continue
        val = float(node.value)
        if isinstance(node.value, float) and val in threshold_values:
            offenders.append(node.value)
        elif isinstance(node.value, int) and val in guarded_ints:
            offenders.append(node.value)
    assert offenders == [], f"driver inlines threshold literal(s): {offenders}"


# --------------------------------------------------------------------------- calibrated occlusion width
def test_calibrated_width_reads_derivation(tmp_path):
    (tmp_path / "derivation.json").write_text(json.dumps({"occlusion": {"width_m": 42.5}}), encoding="utf-8")
    assert d._calibrated_width(_args(tmp_path, derivation=str(tmp_path / "derivation.json"))) == 42.5


def test_calibrated_width_refuses_without_width(tmp_path):
    (tmp_path / "derivation.json").write_text(json.dumps({"occlusion": {}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="width_m"):
        d._calibrated_width(_args(tmp_path, derivation=str(tmp_path / "derivation.json")))


# --------------------------------------------------------------------------- corpus wiring (resume + dirty + IC)
def test_pass_metrics_resumes_before_load_and_excludes(tmp_path, monkeypatch):
    frames = pd.DataFrame({"game_id": ["m1"]})
    matches = {
        ("sportec", "m1"): make_loaded("sportec", "m1", frames=frames, actions=None),
        ("sportec", "m2"): make_loaded("sportec", "m2", frames=frames, actions=None),
    }
    refs = [make_ref("sportec", "m1"), make_ref("sportec", "m2"), make_ref("sportec", "bad")]
    loader = SpyLoader(matches, exclude={("sportec", "bad"): "planted visibility gate"})
    monkeypatch.setattr(d, "corpus_source", lambda args: (refs, loader))
    monkeypatch.setattr(
        d,
        "match_tables",
        lambda loaded, params, **kw: pd.DataFrame(
            [{"provider": loaded.provider, "match_id": str(loaded.match_id), "table": "rsi", "variant": "base"}]
        ),
    )

    res1 = d._pass_metrics(_args(tmp_path), _CLEAN)
    assert res1.excluded == 1 and res1.skipped == 0
    n_loads = len(loader.calls)
    res2 = d._pass_metrics(_args(tmp_path), _CLEAN)
    assert res2.attempted == 0 and res2.skipped == 2 and res2.excluded == 1
    assert len(loader.calls) == n_loads  # resume-before-load: nothing re-loaded
    assert (tmp_path / "metrics.all.parquet").is_file()


def test_pass_metrics_manifest_carries_the_per_stage_breakdown(tmp_path, monkeypatch):
    # R8 (plan L2770, Task 23 L3271): the driver threads its StageTimer into match_tables, so the per-stage breakdown
    # of a match's compute lands in manifest_metrics.json (the official artifact), not in a scratch harness.
    frames = pd.DataFrame({"game_id": ["m1"]})
    loader = SpyLoader({("sportec", "m1"): make_loaded("sportec", "m1", frames=frames, actions=None)})
    monkeypatch.setattr(d, "corpus_source", lambda args: ([make_ref("sportec", "m1")], loader))
    seen = {}

    def fake_match_tables(loaded, params, *, timer=None, **kw):
        seen["timer"] = timer
        if timer is not None:
            with timer("family.cluster_phase"):
                pass
        return pd.DataFrame(
            [{"provider": loaded.provider, "match_id": str(loaded.match_id), "table": "rsi", "variant": "base"}]
        )

    monkeypatch.setattr(d, "match_tables", fake_match_tables)
    d._pass_metrics(_args(tmp_path), _CLEAN)
    assert seen["timer"] is not None
    manifest = json.loads((tmp_path / "manifest_metrics.all.json").read_text(encoding="utf-8"))
    assert {"corpus", "compute", "family.cluster_phase"} <= set(manifest["stage_seconds"])


def test_reduce_refuses_a_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts._provenance.git_provenance",
        lambda: {"commit": "abc", "dirty": True, "tree_state": "dirty", "dirty_files": ["x.py"]},
    )
    monkeypatch.setattr("sys.argv", ["validate_team_coordination.py", "--pass", "reduce", "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="DIRTY"):
        d.main()


def test_input_contract_declared_and_written(tmp_path):
    ic = d.input_contract()
    assert ic["driver"] == "validate_team_coordination" and len(ic["digest"]) == 64
    _write_all_shares(tmp_path, pd.DataFrame())
    d._reduce(_args(tmp_path), _CLEAN)
    report = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert report["input_contract"]["driver"] == "validate_team_coordination"


def test_occlusion_pass_computes_with_the_final_params(tmp_path, monkeypatch):
    # review A-10 / R2-3: D3's occlusion leg ran at bare CoordinationParams() defaults with no params in its token;
    # it computes with the artifact params (M-5) and records their provenance like every other D3 pass
    import json

    from scripts._coordination_params_codegen import INTERIM_BASE

    derivation = tmp_path / "derivation.json"
    derivation.write_text(
        json.dumps(
            {
                "pooled": {**INTERIM_BASE, "butterworth_cutoff_hz": 0.55},
                "providers": {},
                "occlusion": {"width_m": 40.0},
            }
        ),
        encoding="utf-8",
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"confirmation": {"gate_cleared": False}}), encoding="utf-8")
    loader = SpyLoader({("sportec", "m1"): make_loaded("sportec", "m1", frames=pd.DataFrame({"game_id": ["m1"]}))})
    monkeypatch.setattr(d, "corpus_source", lambda args: ([make_ref("sportec", "m1")], loader))
    seen = []

    def spy(loaded, width_m, *, params):
        seen.append(params)
        return pd.DataFrame({"provider": [loaded.provider], "match_id": [loaded.match_id], "value": [1.0]})

    monkeypatch.setattr(d, "occlusion_metrics", spy)
    args = _args(tmp_path, derivation=str(derivation), calibration=str(calibration), in_package_params=False)
    d._pass_occlusion(args, {"commit": "0" * 40, "dirty": False, "tree_state": "clean"})
    assert seen and seen[0].butterworth_cutoff_hz == 0.55
    manifest = json.loads((tmp_path / "manifest_occlusion.all.json").read_text(encoding="utf-8"))
    assert manifest["params_source"] == "artifacts" and len(manifest["derivation_sha256"]) == 64
