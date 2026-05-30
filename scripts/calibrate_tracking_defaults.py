"""TF-24 calibration CLI — orchestrates the two Optuna studies + diagnostics.

Pure objectives/CV/gates live in silly_kicks.calibration; this script owns I/O (loaders), study
orchestration, the frozen-xT artifact, and the report + data/version manifest.

Usage (all three providers come from pining: SkillCorner+IDSSE public, GS via owner token):
    python scripts/calibrate_tracking_defaults.py --stage 1 --source pining \
        --providers skillcorner idsse gradientsports --n-trials 100 --store tc3_stage1.db
    python scripts/calibrate_tracking_defaults.py --stage 2 --source pining \
        --providers skillcorner idsse gradientsports --n-trials 60 --store tc3_stage2.db \
        --xt-artifact calibration_xt.npz --carrier-best carrier_best.json

The frozen-xT corpus (bronze.spadl_actions) is fetched via Databricks regardless of --source
(it is the disjoint exogenous corpus, not calibration data) unless --xt-corpus-source pining is
used (default — id-space-safe held-out fit); DATABRICKS_* env vars are needed only for the
databricks xT corpus, not for the pining match loads.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from importlib.metadata import version

import pandas as pd
from ruthless import InProcessBackend, render_json, render_summary_md
from ruthless.strategies.optuna_ import OptunaStrategy

import silly_kicks
from silly_kicks.calibration import stage1_config, stage2_config
from silly_kicks.calibration._carrier_objective import CarrierAccuracyObjective
from silly_kicks.calibration._vaep_brier_objective import AugmentedVaepBrierObjective

_XT_COLS = ["game_id", "start_x", "start_y", "end_x", "end_y", "type_id", "result_id"]


def build_manifest(*, source, seed, n_trials, match_ids, xt, stage, diagnostics=None) -> dict:
    """Data + version manifest for auditability (spec §6 R3)."""
    manifest = {
        "stage": stage,
        "source": source,
        "seed": seed,
        "n_trials": n_trials,
        "match_ids": match_ids,
        "silly_kicks_version": silly_kicks.__version__,  # source truth (editable-install safe)
        "ruthless_version": version("ruthless-efficiency"),
        "xgboost_version": version("xgboost"),
        "generated_date": _dt.date.today().isoformat(),
    }
    if xt is not None:
        manifest["xt_artifact"] = xt.manifest()  # includes n_excluded (H2 audit)
    if diagnostics:
        manifest["diagnostics"] = diagnostics  # excluded_providers (M1) + das_degraded (M8)
    return manifest


def run_stage(*, stage, fold, n_trials, seed, store_path, xt, carrier_params):
    """Run one Optuna stage on an already-loaded fold (the testable seam — no I/O).

    Returns (result, objective) so the caller can read objective.diagnostics (M1/M8) for the
    manifest and the Stage-1 best params for the carrier_best.json handoff (M9).
    """
    if stage == 1:
        objective = CarrierAccuracyObjective(fold)
        config = stage1_config(n_trials=n_trials, store_path=store_path)
    elif stage == 2:
        objective = AugmentedVaepBrierObjective(fold=fold, xt=xt, carrier_params=carrier_params, seed=seed)
        config = stage2_config(n_trials=n_trials, store_path=store_path)
    else:
        raise ValueError(f"unknown stage {stage}")
    result = OptunaStrategy(config, seed=seed).run(objective, backend=InProcessBackend())
    return result, objective


def _load_fold(args):
    """Wire the chosen loader into the {provider: [(actions, frames, home)]} fold + match_ids."""
    if args.source == "pining":
        import scripts._loader_pining as loader
    else:
        import scripts._loader_databricks as loader
    fold: dict[str, list[tuple]] = {}
    used_ids: dict[str, list[str]] = {}
    for provider, mid, actions, frames, home in loader.load_matches(providers=args.providers, match_ids=None):
        fold.setdefault(provider, []).append((actions, frames, home))
        used_ids.setdefault(provider, []).append(mid)
    return fold, used_ids


def _resolve_xt(args, fold, used_ids):
    """Fit-and-freeze the xT artifact on a disjoint corpus, or load + sha256-verify (N1)."""
    from pathlib import Path

    from silly_kicks.calibration._xt import fit_frozen_xt, load_xt, save_xt

    if args.xt_artifact and Path(args.xt_artifact).exists():
        return load_xt(args.xt_artifact)

    calib_ids = {str(m) for ids in used_ids.values() for m in ids}
    if args.xt_corpus_source == "pining":
        corpus, corpus_ids = _load_xt_corpus_pining(args, calib_ids)
        overlap = corpus_ids & calib_ids
        if overlap:  # held-out corpus must be disjoint by construction
            raise ValueError(f"pining xT corpus overlaps calibration matches: {sorted(overlap)[:5]}")
        frozen = fit_frozen_xt(
            corpus,
            exclude_match_ids=set(),
            match_id_col="game_id",
            source="pining:held-out",
            fit_date=_dt.date.today().isoformat(),
        )
    else:  # databricks
        corpus = _load_xt_corpus_databricks()
        frozen = fit_frozen_xt(
            corpus,
            exclude_match_ids=calib_ids,
            match_id_col="game_id",
            source="bronze.spadl_actions",
            fit_date=_dt.date.today().isoformat(),
        )
    if args.xt_artifact:
        save_xt(frozen, args.xt_artifact)
    return frozen


def _load_xt_corpus_pining(args, calib_ids) -> tuple[pd.DataFrame, set[str]]:
    """Load actions from pining matches NOT in the calibration set (id-space-safe corpus, N1)."""
    import scripts._loader_pining as pining_loader

    token, base_url = pining_loader._resolve_token(None), pining_loader._base_url()
    parts, corpus_ids = [], set()
    per_provider_cap = 8  # bound the corpus; enough matches for a stable xT grid
    for provider in args.providers:
        manifest = pining_loader._list_matches(provider, token, base_url)
        held_out = [m["id"] for m in manifest if str(m["id"]) not in calib_ids][:per_provider_cap]
        for _p, mid, actions, _frames, _home in pining_loader.load_matches(
            providers=[provider], match_ids={provider: held_out}, token=token, tracking_limit=50
        ):
            parts.append(actions[[c for c in _XT_COLS if c in actions.columns]])
            corpus_ids.add(str(mid))
    return pd.concat(parts, ignore_index=True), corpus_ids


def _load_xt_corpus_databricks() -> pd.DataFrame:
    """Load the xT-fit corpus from bronze.spadl_actions — ONLY the columns ExpectedThreat needs (N3)."""
    import scripts._loader_databricks as databricks_loader

    conn = databricks_loader._connect()
    try:
        cur = conn.cursor()
        cols = ", ".join(_XT_COLS)
        return databricks_loader._query_param(cur, f"SELECT {cols} FROM soccer_analytics.bronze.spadl_actions")  # noqa: S608
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="TF-24 tracking-defaults calibration")
    ap.add_argument("--stage", choices=["1", "2", "diagnostics"], required=True)
    ap.add_argument("--source", choices=["pining", "databricks", "auto"], default="pining")
    ap.add_argument("--providers", nargs="+", required=True)
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--store", required=True)
    ap.add_argument("--xt-artifact", default=None)
    ap.add_argument("--xt-corpus-source", choices=["pining", "databricks"], default="pining")
    ap.add_argument("--carrier-best", default=None, help="JSON with the Stage-1 optimum (for Stage 2)")
    ap.add_argument("--report-out", default="calibration_report")
    args = ap.parse_args()

    fold, used_ids = _load_fold(args)

    xt = None
    carrier_params = None
    if args.stage == "2":
        xt = _resolve_xt(args, fold, used_ids)  # pining held-out (default) / databricks bronze
        with open(args.carrier_best, encoding="utf-8") as fh:
            carrier_params = json.load(fh)
        missing = {"tolerance_m", "beta", "gamma"} - set(carrier_params)  # N4a: validate up front
        if missing:
            raise ValueError(f"carrier_best.json missing keys {sorted(missing)} — run Stage 1 first")

    result, objective = run_stage(
        stage=int(args.stage) if args.stage != "diagnostics" else "diagnostics",
        fold=fold,
        n_trials=args.n_trials,
        seed=args.seed,
        store_path=args.store,
        xt=xt,
        carrier_params=carrier_params,
    )

    # M9: Stage 1 writes carrier_best.json so Stage 2 consumes a RECORDED artifact (not hand-typed).
    if args.stage == "1" and result.best is not None:
        best_carrier = {k: result.best.candidate.params[k] for k in ("tolerance_m", "beta", "gamma")}
        with open(args.carrier_best or "carrier_best.json", "w", encoding="utf-8") as fh:
            json.dump(best_carrier, fh, indent=2)
        print(f"Wrote carrier_best.json: {best_carrier}")

    manifest = build_manifest(
        source=args.source,
        seed=args.seed,
        n_trials=args.n_trials,
        match_ids=used_ids,
        xt=xt,
        stage=args.stage,
        diagnostics=getattr(objective, "diagnostics", None),
    )
    report = {"ruthless": json.loads(render_json(result)), "calibration_manifest": manifest}
    with open(f"{args.report_out}.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(f"{args.report_out}.md", "w", encoding="utf-8") as fh:
        fh.write(render_summary_md(result))
        fh.write("\n\n## Calibration manifest\n\n```json\n")
        fh.write(json.dumps(manifest, indent=2))
        fh.write("\n```\n")
    print(render_summary_md(result))
    print(f"Best: {result.best.metrics if result.best else None}")


if __name__ == "__main__":
    main()
