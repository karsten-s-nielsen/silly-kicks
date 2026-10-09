"""ADR-111: ``validate_coordination_numerics`` -- the corpus no-flip gate for the coordination numerics.

``compare_numerics`` is pinned on hand-built frames (every violation class is detected, every clean column reports its
max |delta|); ``numerics_match`` runs BOTH real legs on a synthetic match; map + reduce run over a fake corpus
(resume-before-load) and write the aggregate-only verdict with its input contract.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import validate_coordination_numerics as d
from _fake_corpus import SpyLoader, make_loaded, make_ref

from silly_kicks.coordination._compute import REFERENCE_NUMERICS_ENV
from tests.coordination._fixtures import make_coordination_actions, make_coordination_match

pytestmark = pytest.mark.filterwarnings("ignore:vx/vy columns not found")
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
        in_package_params=True,  # the gate-mechanics tests run on the committed module (a recorded dev run)
    )
    base.update(over)
    return SimpleNamespace(**base)


def _long(values, tokens, percentiles, keys=("a", "b", "c")):
    """A melted two-table frame shaped like ``match_tables``' output (lead columns + the union of table columns)."""
    return pd.DataFrame(
        {
            "provider": "sportec",
            "match_id": "m1",
            "variant": "base",
            "table": ["cluster_team"] * 3 + ["rsi"],
            "team_id": [*keys, pd.NA],
            "coord_rho_group_mean_surrogate_mean": [*values, np.nan],
            "coord_rho_group_mean_percentile": [*percentiles, np.nan],
            "coord_cluster_surrogate_source": [*tokens, pd.NA],
            "coord_rsi_mean_m": [np.nan, np.nan, np.nan, 3.0],
        }
    )


_BASE = ([0.40, 0.50, np.nan], ["computed", "computed", "segment_too_short"], [0.96, 0.30, np.nan])


def _entry(agg, column):
    return agg[agg["column"] == column].iloc[0]


def test_compare_numerics_clean_legs_report_only_deviations():
    agg = d.compare_numerics(_long(*_BASE), _long([0.40 + 1e-15, 0.50, np.nan], _BASE[1], _BASE[2]))
    assert int(agg[["source_flips", "nan_changes", "pct_changed", "crossings"]].to_numpy().sum()) == 0
    assert _entry(agg, "coord_rho_group_mean_surrogate_mean")["max_abs_dev"] == pytest.approx(1e-15, rel=0.2)
    assert set(agg["table"]) == {"cluster_team", "rsi"}  # each table's own columns only
    assert "coord_rsi_mean_m" not in set(agg.loc[agg["table"] == "cluster_team", "column"])
    report = d.reduce_numerics(agg.assign(provider="sportec", match_id="m1"))
    assert report["no_flip"] is True and report["n_matches"] == 1


def test_compare_numerics_excludes_nan_nan_cells_from_n_compared():
    # A-55: n counts every table row; n_compared counts only cells finite in at least one leg, so a NaN-NaN cell of the
    # melted union is not a comparison. The surrogate-mean column has one such row in the cluster_team slice.
    agg = d.compare_numerics(_long(*_BASE), _long(*_BASE))
    e = _entry(agg, "coord_rho_group_mean_surrogate_mean")
    assert e["n"] == 3 and e["n_compared"] == 2  # the NaN-in-both row is not compared
    s = _entry(agg, "coord_cluster_surrogate_source")
    assert s["n"] == s["n_compared"] == 3  # a source/token column compares every row
    report = d.reduce_numerics(agg.assign(provider="sportec", match_id="m1"))
    assert report["n_cells_total"] > report["n_values_compared"]  # the dropped NaN-NaN cells show in the totals


def test_compare_numerics_counts_a_source_token_flip():
    tokens = ["computed", "segment_too_short", "segment_too_short"]
    agg = d.compare_numerics(_long(*_BASE), _long(_BASE[0], tokens, _BASE[2]))
    assert _entry(agg, "coord_cluster_surrogate_source")["source_flips"] == 1
    assert d.reduce_numerics(agg.assign(provider="sportec", match_id="m1"))["no_flip"] is False


def test_compare_numerics_counts_percentile_moves_and_threshold_crossings():
    # 0.96 -> 0.94 crosses 0.95; 0.30 -> 0.31 moves without crossing any threshold.
    agg = d.compare_numerics(_long(*_BASE), _long(_BASE[0], _BASE[1], [0.94, 0.31, np.nan]))
    pct = _entry(agg, "coord_rho_group_mean_percentile")
    assert pct["pct_changed"] == 2 and pct["crossings"] == 1
    report = d.reduce_numerics(agg.assign(provider="sportec", match_id="m1"))
    assert report["no_flip"] is False and report["percentile_threshold_crossings"] == 1


def test_compare_numerics_counts_a_nan_pattern_change():
    agg = d.compare_numerics(_long(*_BASE), _long([0.40, np.nan, np.nan], _BASE[1], _BASE[2]))
    assert _entry(agg, "coord_rho_group_mean_surrogate_mean")["nan_changes"] == 1
    assert d.reduce_numerics(agg.assign(provider="sportec", match_id="m1"))["no_flip"] is False


def test_compare_numerics_refuses_legs_describing_different_rows():
    with pytest.raises(ValueError, match="different rows"):
        d.compare_numerics(_long(*_BASE), _long(*_BASE, keys=("a", "b", "z")))
    with pytest.raises(ValueError, match="reference rows"):
        d.compare_numerics(_long(*_BASE), _long(*_BASE).iloc[1:])


def test_reduce_without_matches_is_not_a_pass():
    empty = pd.DataFrame(columns=["provider", "match_id", *d._AGG_COLUMNS])
    assert d.reduce_numerics(empty)["no_flip"] is False


def test_reference_numerics_context_restores_the_environment(monkeypatch):
    monkeypatch.delenv(REFERENCE_NUMERICS_ENV, raising=False)
    with d.reference_numerics():
        import os

        assert os.environ[REFERENCE_NUMERICS_ENV] == "1"
    assert REFERENCE_NUMERICS_ENV not in __import__("os").environ


def _synthetic_loaded():
    frames = make_coordination_match(seconds=450.0, provider="sportec")
    return SimpleNamespace(provider="sportec", match_id="m1", frames=frames, actions=make_coordination_actions(frames))


def test_numerics_match_runs_both_legs_and_finds_no_flip(monkeypatch):
    # Both REAL legs on a synthetic match: non-vacuous (surrogate percentiles were compared) and clean.
    monkeypatch.setattr(d, "D3_N_SURROGATES", 19)
    timer = d.StageTimer()
    agg = d.numerics_match(_synthetic_loaded(), timer=timer)
    assert {"reference", "production"} <= set(timer.as_dict())
    pct = agg[agg["column"].str.endswith("_percentile")]
    assert int(pct["n"].sum()) > 0
    report = d.reduce_numerics(agg)
    assert report["no_flip"] is True, report
    print("synthetic no-flip max |delta| by column:", agg.groupby("column")["max_abs_dev"].max().to_dict())


def test_map_and_reduce_write_the_aggregate_verdict(tmp_path, monkeypatch):
    frames = pd.DataFrame({"game_id": ["m1"]})
    loader = SpyLoader({("sportec", "m1"): make_loaded("sportec", "m1", frames=frames, actions=None)})
    monkeypatch.setattr(d, "corpus_source", lambda args: ([make_ref("sportec", "m1")], loader))
    calls = []

    def fake_match_tables(loaded, params, **kw):
        calls.append(__import__("os").environ.get(REFERENCE_NUMERICS_ENV))
        return _long(*_BASE)

    monkeypatch.setattr(d, "match_tables", fake_match_tables)
    monkeypatch.chdir(tmp_path)  # a relative write would land here, not in the repo
    d._pass_map(_args(tmp_path), _CLEAN)
    assert calls == ["1", None]  # the reference leg under the switch, production without it
    d._reduce(_args(tmp_path), _CLEAN)
    report = json.loads((tmp_path / "numerics_noflip.json").read_text(encoding="utf-8"))
    assert report["no_flip"] is True and report["n_matches"] == 1
    assert report["corpus_visibility"] == "sc_extended"  # ADR-038: one private Sportec match
    assert {"corpus", "reference", "production"} <= set(report["stage_seconds"])
    assert report["input_contract"]["driver"] == "validate_coordination_numerics"
    assert all(set(row) >= {"table", "column", "n"} for row in report["by_column"])  # aggregates only, no match rows
    assert not (tmp_path / "docs").exists()  # M-5: the verdict lands in --out only (commit 2 copies it in)


def _two_worker_corpus(tmp_path, monkeypatch):
    """A fake two-match corpus whose ``corpus_source`` serves exactly the slice in ``--match-ids-json``."""
    frames = pd.DataFrame({"game_id": ["m1"]})
    loader = SpyLoader({("sportec", m): make_loaded("sportec", m, frames=frames, actions=None) for m in ("m1", "m2")})

    def source(args):
        ids = json.loads(Path(args.match_ids_json).read_text(encoding="utf-8"))["sportec"]
        return [make_ref("sportec", m) for m in ids], loader

    monkeypatch.setattr(d, "corpus_source", source)
    monkeypatch.setattr(d, "match_tables", lambda loaded, params, **kw: _long(*_BASE).assign(match_id=loaded.match_id))
    monkeypatch.chdir(tmp_path)  # a relative write would land here, not in the repo
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({"sportec": ["m1", "m2"]}), encoding="utf-8")
    slices = {}
    for worker, match in (("w0", "m1"), ("w1", "m2")):
        slices[worker] = tmp_path / f"match_ids_{worker}.json"
        slices[worker].write_text(json.dumps({"sportec": [match]}), encoding="utf-8")
    return corpus, slices


def _population_with_exclusion():
    return {
        "consistent": {},
        "stage_seconds": {"corpus": 1.0},
        "n_excluded": 1,
        "excluded_keys": ["sportec__m2"],
        "n_workers": 1,
    }


def test_reduce_refuses_an_undeclared_exclusion(tmp_path, monkeypatch):
    # exclusions nit: a PASS over a corpus with an UNDECLARED excluded match is a silent subset (R2-1 class) -> refuse.
    agg = d.compare_numerics(_long(*_BASE), _long(*_BASE)).assign(provider="sportec", match_id="m1")
    monkeypatch.setattr(d, "combine_workers", lambda *a, **k: (agg, _population_with_exclusion()))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="not DECLARED"):
        d._reduce(_args(tmp_path), _CLEAN)
    assert not (tmp_path / "numerics_noflip.json").exists()  # no verdict over a silent subset


def test_reduce_passes_a_declared_exclusion_and_records_it(tmp_path, monkeypatch):
    # the other side: a DECLARED exclusion (an input, with a vocabulary reason) passes, and the verdict records it.
    agg = d.compare_numerics(_long(*_BASE), _long(*_BASE)).assign(provider="sportec", match_id="m1")
    monkeypatch.setattr(d, "combine_workers", lambda *a, **k: (agg, _population_with_exclusion()))
    monkeypatch.chdir(tmp_path)
    decl = tmp_path / "declared.json"
    decl.write_text(json.dumps({"sportec__m2": "no_tracking"}), encoding="utf-8")
    d._reduce(_args(tmp_path, declared_excluded=str(decl)), _CLEAN)
    report = json.loads((tmp_path / "numerics_noflip.json").read_text(encoding="utf-8"))
    assert report["n_excluded"] == 1 and report["excluded"] == {"sportec__m2": "no_tracking"}


def test_reduce_records_the_run_tree_hash(tmp_path, monkeypatch):
    # A-55: the verdict ties to the exact tree CONTENT (git_tree_hash), not only to a commit + a dirty flag.
    frames = pd.DataFrame({"game_id": ["m1"]})
    loader = SpyLoader({("sportec", "m1"): make_loaded("sportec", "m1", frames=frames, actions=None)})
    monkeypatch.setattr(d, "corpus_source", lambda args: ([make_ref("sportec", "m1")], loader))
    monkeypatch.setattr(d, "match_tables", lambda loaded, params, **kw: _long(*_BASE))
    monkeypatch.setattr("scripts._provenance.git_tree_hash", lambda: "deadbeef" * 5)
    monkeypatch.chdir(tmp_path)
    d._pass_map(_args(tmp_path), _CLEAN)
    d._reduce(_args(tmp_path), _CLEAN)
    report = json.loads((tmp_path / "numerics_noflip.json").read_text(encoding="utf-8"))
    assert report["run_tree_hash"] == "deadbeef" * 5


def test_two_worker_slices_reduce_over_the_whole_corpus(tmp_path, monkeypatch):
    # B-1: the DGX runs 16 workers into one --out; the verdict must judge ALL of their matches, not the last writer's.
    corpus, slices = _two_worker_corpus(tmp_path, monkeypatch)
    for worker in ("w0", "w1"):
        d._pass_map(_args(tmp_path, match_ids_json=str(slices[worker]), corpus_json=str(corpus)), _CLEAN)
    d._reduce(_args(tmp_path, corpus_json=str(corpus)), _CLEAN)
    report = json.loads((tmp_path / "numerics_noflip.json").read_text(encoding="utf-8"))
    assert report["n_matches"] == 2 and report["no_flip"] is True
    assert (
        report["population"]["n_workers"] == 2 and report["population"]["population_checked_against"] == "corpus_json"
    )


def test_workers_handed_different_provider_lists_still_share_one_generation(tmp_path, monkeypatch):
    # Found by the 2026-10-03 MEDIA-PC no-flip: each worker was launched with its own slice's --providers, the shard
    # token hashed only THOSE providers' params, so the workers wrote different shard generations and the reduce
    # (rightly) refused to combine them. The token must not depend on which providers a worker's slice holds.
    corpus, slices = _two_worker_corpus(tmp_path, monkeypatch)
    d._pass_map(
        _args(tmp_path, match_ids_json=str(slices["w0"]), corpus_json=str(corpus), providers=("sportec",)), _CLEAN
    )
    d._pass_map(
        _args(tmp_path, match_ids_json=str(slices["w1"]), corpus_json=str(corpus), providers=("sportec", "idsse")),
        _CLEAN,
    )
    d._reduce(_args(tmp_path, corpus_json=str(corpus)), _CLEAN)
    report = json.loads((tmp_path / "numerics_noflip.json").read_text(encoding="utf-8"))
    assert report["n_matches"] == 2 and report["population"]["n_workers"] == 2


def test_reduce_refuses_a_worker_that_never_ran(tmp_path, monkeypatch):
    corpus, slices = _two_worker_corpus(tmp_path, monkeypatch)
    d._pass_map(_args(tmp_path, match_ids_json=str(slices["w0"]), corpus_json=str(corpus)), _CLEAN)  # w1 never ran
    with pytest.raises(SystemExit, match="miss 1 corpus key"):
        d._reduce(_args(tmp_path, corpus_json=str(corpus)), _CLEAN)
    assert not (tmp_path / "numerics_noflip.json").exists()  # no verdict on a partial corpus


def test_reduce_refuses_a_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts._provenance.git_provenance",
        lambda: {"commit": "abc", "dirty": True, "tree_state": "dirty", "dirty_files": ["x.py"]},
    )
    monkeypatch.setattr("sys.argv", ["validate_coordination_numerics.py", "--pass", "reduce", "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="DIRTY"):
        d.main()


def test_input_contract_is_declared():
    ic = d.input_contract()
    assert ic["driver"] == "validate_coordination_numerics" and len(ic["digest"]) == 64
    assert ic["percentile_thresholds"] == list(d.PERCENTILE_THRESHOLDS)


# --------------------------------------------------------------------------- review A-16 / R2-2: the shipped params
def _artifacts(tmp_path, cutoff=0.55):
    from scripts._coordination_params_codegen import INTERIM_BASE

    derivation = tmp_path / "derivation.json"
    derivation.write_text(
        json.dumps({"pooled": {**INTERIM_BASE, "butterworth_cutoff_hz": cutoff}, "providers": {}}), encoding="utf-8"
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"confirmation": {"gate_cleared": False}}), encoding="utf-8")
    return derivation, calibration


def test_the_gate_computes_with_the_artifact_params_and_records_them(tmp_path, monkeypatch):
    # The HARD gate certifies the configuration that SHIPS: the D1/D2 artifacts (final params), not the interim
    # in-package module (under M-5 the DGX chain never rewrites it). Their digests ride in the token, the manifest
    # and the verdict, and every worker must have used the same pair.
    import hashlib

    corpus, slices = _two_worker_corpus(tmp_path, monkeypatch)
    seen = []

    def spy(loaded, params, **kw):
        seen.append(params.butterworth_cutoff_hz)
        return _long(*_BASE).assign(match_id=loaded.match_id)

    monkeypatch.setattr(d, "match_tables", spy)
    derivation, calibration = _artifacts(tmp_path)
    art = {"derivation": str(derivation), "calibration": str(calibration), "in_package_params": False}
    for worker in ("w0", "w1"):
        d._pass_map(_args(tmp_path, match_ids_json=str(slices[worker]), corpus_json=str(corpus), **art), _CLEAN)
    assert seen and set(seen) == {0.55}  # both legs of every match computed with the artifact's cutoff
    d._reduce(_args(tmp_path, corpus_json=str(corpus), **art), _CLEAN)
    verdict = json.loads((tmp_path / "numerics_noflip.json").read_text(encoding="utf-8"))
    assert verdict["params"]["params_source"] == "artifacts"
    assert verdict["params"]["derivation_sha256"] == hashlib.sha256(derivation.read_bytes()).hexdigest()
    assert verdict["params"]["calibration_sha256"] == hashlib.sha256(calibration.read_bytes()).hexdigest()


def test_the_gate_refuses_to_run_without_the_artifacts(tmp_path, monkeypatch):
    corpus, slices = _two_worker_corpus(tmp_path, monkeypatch)
    args = _args(tmp_path, match_ids_json=str(slices["w0"]), corpus_json=str(corpus), in_package_params=False)
    with pytest.raises(SystemExit, match="--derivation"):
        d._pass_map(args, _CLEAN)
