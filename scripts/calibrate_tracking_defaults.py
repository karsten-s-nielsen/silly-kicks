"""TF-24 calibration CLI -- orchestrates the two Optuna studies + diagnostics.

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
used (default -- id-space-safe held-out fit); DATABRICKS_* env vars are needed only for the
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
    """Data + version manifest for auditability (spec section 6 R3)."""
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
    """Run one Optuna stage on an already-loaded fold (the testable seam -- no I/O).

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


def _parse_match_ids(raw):
    """Parse repeatable ``--match-ids PROVIDER:id1,id2`` into ``{provider: [ids]}`` (or None)."""
    if not raw:
        return None
    parsed: dict[str, list[str]] = {}
    for entry in raw:
        provider, _, ids = entry.partition(":")
        provider = provider.strip()
        if not provider or not ids.strip():
            raise ValueError(f"--match-ids expects 'PROVIDER:id1,id2', got {entry!r}")
        parsed.setdefault(provider, []).extend(i.strip() for i in ids.split(",") if i.strip())
    return parsed


def _assert_match_game_id_consistent(provider: str, mid: str, actions, frames) -> None:
    """Fail loud if a match's actions and frames disagree on ``game_id``.

    Every tracking-feature join (ball carrier, DAS, defensive line, team shape) keys on
    ``(game_id, period_id, frame_id)``. A mismatch silently drops EVERY row for the match, so the
    provider contributes ~0 signal and is quietly excluded by ``signal_sanity`` -- calibrating on
    fewer providers than the operator requested (the IDSSE ``game_id=None`` regression). Surface it
    instead of degrading the fold in silence; the loader must stamp a consistent game_id on both.
    """
    a_ids = set(actions["game_id"].dropna().unique()) if "game_id" in actions.columns else set()
    f_ids = set(frames["game_id"].dropna().unique()) if "game_id" in frames.columns else set()
    if not a_ids or not f_ids or {str(x) for x in a_ids} != {str(x) for x in f_ids}:
        raise ValueError(
            f"{provider} match {mid}: actions game_id {sorted(map(str, a_ids)) or '[none]'} != frames "
            f"game_id {sorted(map(str, f_ids)) or '[none]'}. Tracking-feature joins key on game_id, so "
            f"this silently drops the whole match (0 carrier signal -> provider excluded). The loader "
            f"must stamp a consistent game_id on both actions and frames."
        )


def _load_fold(args):
    """Wire the chosen loader into the {provider: [(actions, frames, home)]} fold + match_ids.

    WHY THIS LOOP HAS NO SHARDS, unlike every other corpus walk in `scripts/` (ADR-052). What it
    accumulates is not a RESULT -- it is the Optuna objective's INPUT. ``fold`` holds the raw
    ``(actions, frames, home)`` triples, which `CachedObjective` prepares once and then re-reads on
    every trial; there is no per-match output to persist, so a shard here would be a second copy of
    the tracking corpus in a second format, and resuming it would still have to hand the whole
    thing to the objective. The resilience this loop can actually have is paying the DOWNLOAD only
    once, which is what ``--cache-dir`` does through the loader's own artifact cache; the xT-corpus
    walk in `_load_xt_corpus_pining` is the one whose per-match result IS a table, and that is the
    one that shards.
    """
    if args.source == "pining":
        import scripts._loader_pining as loader
    else:
        import scripts._loader_databricks as loader
    match_ids = _parse_match_ids(getattr(args, "match_ids", None))
    # Memory bounds: tracking_limit caps frames/match; max_per_provider caps matches/provider.
    # Both default to None (load everything) for back-compat; set them to run the sweep locally.
    load_kwargs = dict(
        providers=args.providers,
        match_ids=match_ids,
        tracking_limit=getattr(args, "tracking_limit", None),
        max_per_provider=getattr(args, "max_matches_per_provider", None),
    )
    # Pining-only: the artifact cache is a downloaded-file store, which bronze has no analogue for.
    if args.source == "pining" and getattr(args, "cache_dir", None):
        load_kwargs["cache_dir"] = args.cache_dir
    fold: dict[str, list[tuple]] = {}
    used_ids: dict[str, list[str]] = {}
    for provider, mid, actions, frames, home in loader.load_matches(**load_kwargs):  # type: ignore[reportArgumentType]
        _assert_match_game_id_consistent(provider, mid, actions, frames)
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
    """Load actions from pining matches NOT in the calibration set (id-space-safe corpus, N1).

    Walked as an ID LIST rather than as a stream, deliberately (ADR-052). `for_each` resumes
    ``work``, never the PRODUCTION of its items -- and a streamed ``load_matches`` downloads and
    parses a match INSIDE the generator, before yielding it. Here ``work`` is a column slice, i.e.
    unambiguously trivial next to the load, so streaming would make a resumed run re-pay the whole
    corpus in order to skip a handful of trivial writes. The manifest listing that yields the ids
    already happens, so inverting costs nothing.
    """
    from pathlib import Path

    import scripts._loader_pining as pining_loader
    from scripts._driver import for_each, shard_path

    token, base_url = pining_loader._resolve_token(None), pining_loader._base_url()
    per_provider_cap = 8  # bound the corpus; enough matches for a stable xT grid
    items: list[tuple[str, str]] = []
    for provider in args.providers:
        manifest = pining_loader._list_matches(provider, token, base_url)
        held_out = [m["id"] for m in manifest if str(m["id"]) not in calib_ids][:per_provider_cap]
        items.extend((provider, str(mid)) for mid in held_out)

    def _work(item):
        provider, mid = item
        for _p, _mid, actions, _frames, _home in pining_loader.load_matches(
            providers=[provider],
            match_ids={provider: [mid]},
            token=token,
            tracking_limit=50,
            cache_dir=getattr(args, "cache_dir", None),
        ):
            return actions[[c for c in _XT_COLS if c in actions.columns]]
        # The loader yielded nothing -- a geometry-excluded skillcorner match. An EMPTY shard, so a
        # resume records "walked, contributed nothing" instead of re-downloading it every time.
        return None

    res = for_each(
        items,
        key=lambda item: (str(item[0]), str(item[1])),
        work=_work,
        shard_root=Path(f"{args.report_out}_xt_corpus_shards"),
        # What determines a shard's CONTENT: the columns kept, and the frame cap the loader parses
        # under. The held-out SELECTION is not declared -- it is derived from the calibration set,
        # so narrowing the calibration corpus widens this one, and re-downloading a match that is
        # already on disk purely because a different match joined the set would be pure waste.
        token_inputs={"xt_cols": list(_XT_COLS), "tracking_limit": 50},
        tag="xt_corpus",
        label="match",
    )
    if res.failures:
        raise RuntimeError(f"{len(res.failures)} xT-corpus match(es) failed: {res.failures}")

    # Combined from THIS PASS'S keys (no partition surface here -- see `reconcile`'s precondition).
    # `corpus_ids` counts matches that CONTRIBUTED ACTIONS, where it previously counted matches the
    # loader yielded. The two differ only for a held-out match whose actions are empty, and the set
    # feeds one disjointness assertion over ids that exclude `calib_ids` by construction -- a match
    # contributing zero actions is not in the corpus in the sense that check is about.
    parts, corpus_ids = [], set()
    for k, (_provider, mid) in zip(res.keys, items, strict=True):
        frame = pd.read_parquet(shard_path(res.shard_dir, k))
        if len(frame):
            parts.append(frame)
            corpus_ids.add(str(mid))
    if not parts:
        raise ValueError("the held-out xT corpus is empty -- no match contributed any actions")
    return pd.concat(parts, ignore_index=True), corpus_ids


def _load_xt_corpus_databricks() -> pd.DataFrame:
    """Load the xT-fit corpus from bronze.spadl_actions -- ONLY the columns ExpectedThreat needs (N3)."""
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
    ap.add_argument(
        "--match-ids",
        action="append",
        default=None,
        metavar="PROVIDER:id1,id2",
        help="Restrict to specific match ids per provider (repeatable), e.g. "
        "--match-ids gradientsports:10517,10519 --match-ids idsse:M1",
    )
    ap.add_argument(
        "--max-matches-per-provider",
        type=int,
        default=None,
        help="Cap the number of matches loaded per provider (bounds memory; prevents the "
        "TF-24 sweep OOM when loading all matches at full tracking depth locally).",
    )
    ap.add_argument(
        "--tracking-limit",
        type=int,
        default=None,
        help="Cap frames loaded per match (passed to the kloppy parsers; bounds memory).",
    )
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="Persist every downloaded pining artifact here and reuse it on re-runs. The "
        "calibration fold cannot be sharded (it IS the objective's input, see _load_fold), so "
        "this is the resilience that loop can have: a crashed sweep re-reads from disk instead "
        "of re-downloading the tracking corpus. Ignored for --source databricks.",
    )
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit a dev run from a modified tree; the report still records dirty: true",
    )
    args = ap.parse_args()

    # ADR-037: refuse BEFORE `_load_fold` downloads the pining tracking corpus. The harness
    # RECOMMENDS library defaults, so its manifest is cited in an apply-PR -- a recommendation
    # nobody can trace back to a commit cannot be reproduced or audited.
    from scripts._provenance import git_provenance, require_clean_tree

    provenance = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)

    fold, used_ids = _load_fold(args)

    xt = None
    carrier_params = None
    if args.stage == "2":
        xt = _resolve_xt(args, fold, used_ids)  # pining held-out (default) / databricks bronze
        with open(args.carrier_best, encoding="utf-8") as fh:
            carrier_params = json.load(fh)
        missing = {"tolerance_m", "beta", "gamma"} - set(carrier_params)  # N4a: validate up front
        if missing:
            raise ValueError(f"carrier_best.json missing keys {sorted(missing)} -- run Stage 1 first")

    result, objective = run_stage(
        stage=int(args.stage) if args.stage != "diagnostics" else "diagnostics",
        fold=fold,
        n_trials=args.n_trials,
        seed=args.seed,
        store_path=args.store,
        xt=xt,  # FrozenXt artifact; the Stage-2 objective unwraps the inner ExpectedThreat itself
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
    report = {
        "ruthless": json.loads(render_json(result)),
        "calibration_manifest": manifest,
        "run_commit": provenance["commit"],
        "run_tree_dirty": provenance["dirty"],
    }
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
