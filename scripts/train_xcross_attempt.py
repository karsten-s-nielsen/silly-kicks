#!/usr/bin/env python
"""Train the xCrossAttempt (xCross) model (TF-17 weights run, PR-B).

Two match sources:
  --data-dir DIR     parquet dirs DIR/*/{frames,actions}.parquet (smoke / local corpus)
  --providers a,b,c  pining loader (skillcorner,idsse,gradientsports) for the maintainer run

Streams per match, caches features, and (on a public/owner mix with Gradient Sports) runs the
common-public-held-out PAIRED data-effect comparison over THREE candidates (public / sc_extended
/ full) with NESTED HPO -- each candidate re-tuned per outer fold with that fold's public games
excluded (spec 4.1, reviewer M4) -- then selects the shipped corpus via the registered fixed
sequence (scripts/_paired.py). Computes FAIL-CLOSED acceptance gates, and writes a pickle-free
artifact ONLY if the gates pass. Quality numbers in metrics.json are CV/protocol estimates, not
the shipped all-data fit.

Mirror of scripts/train_xshot_occurrence.py. Requires: silly-kicks[train,xgboost]
(+ [kloppy] for --providers).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from silly_kicks.tracking._xcross_attempt import XCrossFeatureSet

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]


def _corpus_fingerprint(args) -> str:
    """Fingerprint of the corpus THIS run requests, for cache validity (ADR-050).

    Mirror of the xS trainer's helper -- see that docstring. Keyed on the REQUESTED corpus via the
    same ``select_match_ids`` / ``_wanted_for_provider`` rule ``load_matches`` applies, so the
    fingerprint cannot describe a corpus the extraction never loaded.
    """
    sys.path.insert(0, "scripts")
    from _cache import corpus_fingerprint

    if not args.providers:
        d = Path(args.data_dir)
        rows = [("local", p.name, "private") for p in sorted(d.iterdir()) if p.is_dir()]
        return corpus_fingerprint(rows)

    from _loader_pining import match_visibility, select_match_ids

    providers = args.providers.split(",")
    allowlist = json.load(open(args.match_ids_json)) if args.match_ids_json else None
    pairs = select_match_ids(providers=providers, match_ids=allowlist, max_per_provider=args.max_per_provider)
    vis = match_visibility(providers)
    return corpus_fingerprint([(p, m, vis.get((p, m), "private")) for p, m in pairs])


def _iter_matches_from_dir(data_dir: Path):
    for game_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        frames = pd.read_parquet(game_dir / "frames.parquet")
        actions = pd.read_parquet(game_dir / "actions.parquet")
        prov = str(frames["source_provider"].iloc[0]) if "source_provider" in frames.columns else "unknown"
        yield prov, game_dir.name, actions, frames, frames["team_id"].dropna().iloc[0]


def _source_key(item):
    """The `for_each` key for BOTH sources: a `MatchRef` (pining) -> its `.key`, a --data-dir tuple
    -> `(provider, match_id)`. Module-level so the key-pin gate can see it (_KEY_EXCEPTIONS)."""
    key = getattr(item, "key", None)
    return key if key is not None else (str(item[0]), str(item[1]))


def _new_probe_cohort() -> dict:
    """One TF-19 probe cohort's capture state (M5): bounded frames/actions copies + provenance."""
    return {"frames": [], "actions": [], "home": None, "matches": [], "match_groups": {}}


#: The four per-row arrays `_extract` returns alongside the feature matrix, carried as COLUMNS so
#: one match is one tidy shard. Underscore-prefixed and collision-checked: a feature named `_y`
#: would be silently overwritten, and the model would train on its own label.
_SIDE_COLS = ("_y", "_group", "_provider", "_match_id")


def _extract(
    source,
    horizon_seconds,
    *,
    shard_root,
    probe_keep=2,
    probe_providers=("gradientsports",),
    probe_comparison_providers=("skillcorner",),
    feature_set: XCrossFeatureSet = "faithful",
    load=None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[dict, dict, int]]:
    from scripts._driver import for_each, shard_path
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
    from silly_kicks.tracking._xcross_attempt import (
        XCROSS_FEATURE_NAMES_FAITHFUL,
        XCROSS_FEATURE_NAMES_POSITION_ONLY,
        prepare_xcross_training_data,
    )

    _feature_names = (
        XCROSS_FEATURE_NAMES_POSITION_ONLY if feature_set == "position_only" else XCROSS_FEATURE_NAMES_FAITHFUL
    )

    collision = set(_SIDE_COLS) & set(XCROSS_FEATURE_NAMES_FAITHFUL)
    if collision:
        raise ValueError(f"side columns {sorted(collision)} collide with feature names")

    # TF-19 substitution-probe samples (M1/M3/M5): the GATED cohort is provider-CONTROLLED
    # (--probe-providers; default the gradientsports gated cohort); a second reported-not-gated
    # comparison cohort (--probe-comparison-providers) persists to _probe_sample_comparison/.
    probe, comparison = _new_probe_cohort(), _new_probe_cohort()

    def _work(item):
        prov, mid, actions, frames, home, *_ = item
        X, y, groups = prepare_xcross_training_data(
            frames,
            actions,
            home_team_id=home,
            feature_set=feature_set,
            horizon_seconds=horizon_seconds,
            wide_area_only=True,
            carrier_params=DEFAULT_CARRIER_PARAMS,  # 4.7.0 values; shared constant (anti-drift)
        )
        if not len(X):
            del frames
            return None  # still writes an EMPTY shard: "ran, produced no usable row"
        cohort = probe if prov in probe_providers else comparison if prov in probe_comparison_providers else None
        if cohort is not None and len(cohort["frames"]) < probe_keep:  # M3: capture a COPY before del frames
            # N3 (memory): keeps up to `probe_keep` matches' frames+actions resident per cohort for
            # the whole loop (deliberate, bounded -- vs the original's immediate del). Fine at tracking
            # scale on the box; probe_keep caps it. The per-match `del frames` still frees all others.
            cohort["frames"].append(frames.copy())
            cohort["actions"].append(actions.copy())
            cohort["home"] = home
            cohort["matches"].append([prov, str(mid)])
            # groups == game_id per row (prepare_xcross_training_data contract), recorded so the
            # gate can compute per-match training-fold membership (M6) + filter the probe frames.
            cohort["match_groups"][str(mid)] = sorted({str(g) for g in np.asarray(groups).tolist()})
        out = X.assign(
            _y=np.asarray(y, int),
            _group=np.asarray(groups),
            _provider=str(prov),
            _match_id=str(mid),  # per-row pining match_id (visibility key)
        )
        del frames
        return out

    res = for_each(
        source,
        key=_source_key,
        load=load,
        work=_work,
        shard_root=shard_root,
        # Mirrors the xS trainer: extractor, horizon, domain filter, carrier params. The probe
        # provider filters are NOT declared -- they select which matches are COPIED into the gate
        # cohort, not what a feature row contains.
        token_inputs={
            "extractor": "prepare_xcross_training_data",
            # feature_set changes the X columns (16 vs 15) -> MUST key the shard generation (4.77.1).
            "feature_set": feature_set,
            "horizon_seconds": horizon_seconds,
            "wide_area_only": True,
            "carrier_params": dict(DEFAULT_CARRIER_PARAMS),
        },
        tag="xcross_features",
        label="match",
    )
    if res.failures:
        raise RuntimeError(f"{len(res.failures)} match(es) failed: {res.failures}. Re-run to retry only them.")

    parts = [f for f in (pd.read_parquet(shard_path(res.shard_dir, k)) for k in res.shard_keys) if len(f)]
    if not parts:
        raise SystemExit("No usable training data.")
    combined = pd.concat(parts, ignore_index=True)
    return (
        combined[_feature_names],
        combined["_y"].to_numpy(int),
        combined["_group"].to_numpy(),
        combined["_provider"].to_numpy(),
        combined["_match_id"].to_numpy(),
        # `res.skipped` rides along because the probe cohort CANNOT be rebuilt from the shards: it
        # holds whole tracking frames, which no tidy shard carries. A resumed pass therefore returns
        # an EMPTY cohort, and `_write_probe_sample` no-ops on empty -- so without this count the
        # TF-19 gate cohort would silently never be written. The caller turns that into a raise.
        (probe, comparison, res.skipped),
    )


def _write_probe_sample(ps: Path, cohort: dict, provider_filter: list) -> None:
    """Persist one probe cohort + its provenance meta.json (M5). No-op on an empty cohort
    (no match from the filtered providers seen). Extracted so the write is unit-testable
    without Databricks."""
    if not cohort["frames"]:
        return
    ps.mkdir(parents=True, exist_ok=True)
    pd.concat(cohort["frames"], ignore_index=True).to_parquet(ps / "frames.parquet")
    pd.concat(cohort["actions"], ignore_index=True).to_parquet(ps / "actions.parquet")
    meta = {
        "home_team_id": str(cohort["home"]),
        "probe_matches": cohort["matches"],  # [[provider, match_id], ...]
        "probe_providers": list(provider_filter),  # the capture filter used
        "match_groups": cohort["match_groups"],  # match_id -> [game_id, ...] (M6 fold membership)
    }
    json.dump(meta, open(ps / "meta.json", "w"), indent=2)


def _gated_probe_matches(meta: dict, admitted: bool) -> list:
    """Pure M6 gate: the probe matches valid for the GATED tf19 statistic.

    ``admitted`` = the paired test admitted the probe provider into the SHIPPED training
    corpus. Not admitted -> every probe match is held-out by construction -> all pass
    through. Admitted -> only matches recorded OUTSIDE the shipped training folds
    (``meta["in_training_folds"]``; unknown membership counts as in-training) are valid,
    and the gate FAILS LOUD rather than emit an in-sample tf19_ready: missing provenance
    (a pre-plan probe sample) and zero held-out matches both refuse.
    """
    matches = list(meta.get("probe_matches", []))
    if not admitted:
        return matches
    if not matches:
        raise SystemExit(
            "Probe provenance missing from _probe_sample/meta.json (pre-plan probe sample) while "
            "the paired test ADMITTED the probe provider to training -> held-out status cannot be "
            "verified. Refusing to emit tf19_ready from potentially in-sample frames. Delete the "
            "feature cache + probe sample and re-extract."
        )
    membership = meta.get("in_training_folds", {})
    held = [m for m in matches if not membership.get(str(m[1]), True)]
    if not held:
        raise SystemExit(
            "Held-out gated statistic impossible (M6): the paired test admitted the probe provider "
            f"to training and every probe match {[m[1] for m in matches]} sits in the shipped "
            "training folds. Refusing to emit tf19_ready from in-sample frames. Re-extract with "
            "probe matches excluded from training, or ship the public candidate."
        )
    return held


def _hpo_once(
    X,
    y,
    groups,
    out_dir,
    tag,
    n_trials,
    *,
    objective_inputs,
    prov,
    negative_subsample=None,
    seed=42,
    study_shard_dir=None,
) -> dict:
    """Run ruthless HPO once for one candidate; return the frozen best-params dict.

    ``objective_inputs`` (D21) carries the run's shared identity parts (driver/args/match_ids); this
    seam COMPLETES them with its per-fold ``tag`` and ``prov`` into the store's ``objective_id``, so
    each fold's study keys on its own tag (the trainer's per-site extra) and a dirty tree never resumes.

    ``study_shard_dir`` (opt-in) caches the frozen params to ``<dir>/<tag>.study.json`` -- a shard
    written under THIS call's ``objective_id`` and ``n_trials`` is returned (resume), else HPO runs and the
    shard is (re)written: the store's own D21 resume rule, so a shard from another run is never served
    stale and a dirty tree never resumes from one (a dirty-tree parallel run recomputes in the reduce).
    Deterministic (seeded TPE + tag-keyed sqlite store), so a cached study is byte-identical to an
    in-process one; this is what lets the nested studies run as independent parallel workers (5c). JSON
    round-trips finite floats exactly.
    """
    from scripts._input_contract import declare_inputs
    from scripts._provenance import objective_id, store_path_for
    from scripts._study_shared import read_study_shard, write_study_shard
    from silly_kicks.tracking._xcross_attempt_objective import XCrossAttemptObjective

    # The id depends on the objective CLASS, never its data, so it is known before the shard check.
    oid = objective_id(XCrossAttemptObjective, declare_inputs(**objective_inputs, tag=tag), prov=prov)
    if study_shard_dir is not None:
        cached = read_study_shard(study_shard_dir, tag, objective_id=oid, n_trials=n_trials)
        if cached is not None:
            return cached

    from ruthless import Direction, FloatRange, InProcessBackend, OptunaConfig
    from ruthless.config.common import StoreConfig
    from ruthless.strategies.optuna_ import OptunaStrategy

    obj = XCrossAttemptObjective(
        fold={tag: [(X, pd.Series(y), groups)]}, negative_subsample=negative_subsample, subsample_seed=seed
    )
    cfg = OptunaConfig(
        kind="optuna",
        metric="logloss",
        direction=Direction.MINIMIZE,
        n_trials=n_trials,
        sampler="tpe",
        param_space={
            "n_estimators": FloatRange(kind="float", lo=50.0, hi=400.0),
            "max_depth": FloatRange(kind="float", lo=2.0, hi=6.0),
            "learning_rate": FloatRange(kind="float", lo=0.02, hi=0.4, log=True),
            "min_child_weight": FloatRange(kind="float", lo=1.0, hi=20.0),
            "reg_lambda": FloatRange(kind="float", lo=0.0, hi=5.0),
        },
        # a dirty id opens a FRESH store (store_path_for): never resumes, never reopens a worker's store
        store=StoreConfig(
            kind="sqlite",
            path=store_path_for(out_dir / f"study_{tag}.db", oid),
            objective_id=oid,  # pyright: ignore[reportCallIssue]  # B m4: ruthless-efficiency 0.7.0 stub omits objective_id
        ),
    )
    result = OptunaStrategy(cfg, seed=42).run(obj, backend=InProcessBackend())
    if result.best is None:
        raise RuntimeError("HPO produced no best candidate")
    params = dict(result.best.candidate.params)
    if study_shard_dir is not None:
        write_study_shard(study_shard_dir, tag, params, objective_id=oid, n_trials=n_trials)
    return params


def _cv_metrics(X, y, groups, params, *, negative_subsample=None, seed=42) -> dict:
    """Label-stratified, match-grouped CV at FIXED params -> gate metrics on the TRUE balance.

    M4: scoring is delegated to silly_kicks.tracking._xcross_eval._cv_score so the acceptance gate
    and the GK-block ablation share ONE implementation (identical folds for any seed/negative_subsample
    -- no drift). `_cv_metrics` re-adds only its two gate-specific keys (positive_rate, base_rate_brier).
    """
    from silly_kicks.tracking import _xcross_eval as ev

    s = ev._cv_score(X, y, groups, params, seed=seed, negative_subsample=negative_subsample)
    base = float(np.asarray(y, dtype=int).mean())
    return {**s, "positive_rate": base, "base_rate_brier": base * (1 - base)}


def _gates(m: dict) -> dict:
    pr = m["pr_auc"]
    br = m["brier"]
    return {
        "enough_usable_folds": m.get("n_usable_folds", 0) >= 2,
        "pr_auc_gt_base_rate": bool(pr == pr and pr > m["positive_rate"]),  # NaN-safe strict
        "brier_lt_base_rate_brier": bool(br == br and br < m["base_rate_brier"]),
        "log_loss_lt_uniform": m["log_loss"] < float(np.log(2)),
    }


def _fit_score(X_tr, y_tr, X_te, y_te, params, *, negative_subsample=None, seed=42) -> float:
    """Fit XGBoost at ``params`` on (X_tr, y_tr); return PR-AUC on (X_te, y_te).

    Module-level extraction of the old ``_paired_data_effect`` closure with ``params`` + the eval
    slice made EXPLICIT, so ONE path serves both protocols: the candidate's OWN tuned params
    (nested) and the public params (shared). Degenerate (single-class) folds return NaN (the caller
    drops them). Preserves the base_score override + train-only negative subsampling. Mirrors the xS
    twin exactly EXCEPT ``_pinned_params`` comes from ``_xcross_attempt`` (``subsample_negatives`` is
    single-sourced from ``_xshot_occurrence`` -- xCross has none of its own, as the old closure did).
    """
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import average_precision_score

    from silly_kicks.tracking._xcross_attempt import _pinned_params
    from silly_kicks.tracking._xshot_occurrence import subsample_negatives

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return float("nan")
    if negative_subsample:  # TRAIN only; the held-out fold (X_te, y_te) is never subsampled
        X_tr, y_tr, _ = subsample_negatives(X_tr, y_tr, y_tr, fraction=negative_subsample, seed=seed)
        if len(np.unique(y_tr)) < 2:
            return float("nan")
    p_ = dict(_pinned_params(params))
    p_["base_score"] = float(y_tr.mean())  # XGBoost's default base_score is wrong for this balance
    clf = xgb.XGBClassifier(**p_)
    clf.fit(X_tr.to_numpy(float), y_tr)
    return float(average_precision_score(y_te, clf.predict_proba(X_te.to_numpy(float))[:, 1]))


def _public_folds(X, y, groups, is_public):
    """The public held-out CV split, single-sourced so a parallel study worker and the serial paired
    loop derive the SAME fold-k ``trainable`` mask. Returns ``(Xp, yp, [(te_idx, trainable_mask), ...])``.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    Xp, yp, gp = X[is_public], y[is_public], groups[is_public]
    k = max(2, min(5, len(np.unique(gp))))
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
    folds = []
    for _fold, (_tr, te) in enumerate(skf.split(Xp, yp, gp)):
        te_games = set(np.asarray(gp)[te].tolist())
        trainable = ~(is_public & np.isin(groups, list(te_games)))  # drop fold-k public games from ALL arms
        folds.append((te, trainable))
    return Xp, yp, folds


def _paired_data_effect(
    X,
    y,
    groups,
    is_public,
    match_ids,  # accepted for caller-signature symmetry (Task 11 Step 4); masks arrive pre-built
    *,
    candidates,
    n_trials,
    out_dir,
    objective_inputs,
    prov,
    negative_subsample=None,
    seed=42,
    study_shard_dir=None,
) -> dict:
    """Nested-HPO paired comparison on the common public held-out folds (spec 4.1, reviewer M4).

    The historical version tuned HPO ONCE, outside the outer CV, on the public arm -- so `public`
    tuned on exactly the matches that ARE the evaluation universe (differential leakage favouring
    `public`, deciding the ship). Here, for each outer fold k, EVERY candidate is tuned on its OWN
    training data with fold k's public games EXCLUDED, then fitted at those params and scored on
    fold k. `candidates` maps name -> row mask. Returns, per candidate, per-fold PR-AUC deltas vs
    `public` under both protocols:
      * "nested"        -- PRIMARY, decides the ship (each candidate at ITS OWN tuned params)
      * "shared_params" -- REPORTED for comparability with 4.9.0/4.18.0 (candidate at PUBLIC params)
    """
    Xp, yp, folds = _public_folds(X, y, groups, is_public)
    out = {name: {"nested": [], "shared_params": []} for name in candidates}

    for fold, (te, trainable) in enumerate(folds):
        X_te, y_te = Xp.iloc[te], yp[te]  # the PUBLIC held-out fold (positional)

        fold_params = {
            name: _hpo_once(
                X[mask & trainable],
                y[mask & trainable],
                groups[mask & trainable],
                out_dir,
                f"{name}_f{fold}",  # real dir + unique tag -> no study-db collision
                n_trials,
                objective_inputs=objective_inputs,
                prov=prov,
                negative_subsample=negative_subsample,
                seed=seed,
                study_shard_dir=study_shard_dir,
            )
            for name, mask in candidates.items()
        }
        d_pub = _fit_score(
            X[candidates["public"] & trainable],
            y[candidates["public"] & trainable],
            X_te,
            y_te,
            fold_params["public"],
            negative_subsample=negative_subsample,
            seed=seed,
        )
        for name, mask in candidates.items():
            if name == "public":
                continue
            m = mask & trainable
            d_nested = _fit_score(
                X[m], y[m], X_te, y_te, fold_params[name], negative_subsample=negative_subsample, seed=seed
            )
            d_shared = _fit_score(
                X[m], y[m], X_te, y_te, fold_params["public"], negative_subsample=negative_subsample, seed=seed
            )
            if not (np.isnan(d_pub) or np.isnan(d_nested)):
                out[name]["nested"].append(float(d_nested - d_pub))
            if not (np.isnan(d_pub) or np.isnan(d_shared)):
                out[name]["shared_params"].append(float(d_shared - d_pub))
    return out


class AcceptanceGatesFailedError(RuntimeError):
    """The shipped candidate failed a fail-closed acceptance gate. main() maps this to exit 1."""


def _build_candidate_masks(X, providers, is_public) -> dict:
    """The 3 paired candidates, single-sourced so main, ``run_one_study`` and ``assemble_studies``
    build identical row masks (the ``gradientsports`` owner rows make ``full`` != ``sc_extended``)."""
    is_sc_private = (providers == "skillcorner") & ~is_public
    return {
        "public": is_public,
        "sc_extended": is_public | is_sc_private,
        "full": np.ones(len(X), bool),
    }


def _fit_study_for_test(X, y, groups, tag, n_trials, seed=42):
    """Test-only seam: run one HPO study on ``(X, y, groups)`` and return ``(params, booster)``,
    mirroring the shipped fit path so a shared-mmap study can be proven byte-identical to an in-memory one."""
    import tempfile

    import xgboost as xgb

    from silly_kicks.tracking._xcross_attempt import _pinned_params

    y = np.asarray(y)
    d = tempfile.mkdtemp(prefix="xcross_study_")
    # D21: a fixed clean identity -- this seam is test-only and its store is a fresh tempdir.
    params = _hpo_once(
        X,
        y,
        np.asarray(groups),
        Path(d),
        tag,
        n_trials,
        objective_inputs={"driver": "train_xcross_attempt._fit_study_for_test"},
        prov={"commit": "test", "dirty": False, "tree_state": "clean"},
        seed=seed,
    )
    p_ = dict(_pinned_params(params))
    p_["base_score"] = float(y.mean())
    clf = xgb.XGBClassifier(**p_)
    clf.fit(X.to_numpy(dtype=float), y)
    booster = clf.get_booster()
    booster.feature_names = list(X.columns)
    return params, booster


def enumerate_studies(shard_root) -> list[str]:
    """The parallel study tags ``{candidate}_f{fold}`` for a run (empty for a single-candidate corpus)."""
    from scripts._study_shared import load_study_inputs

    inp = load_study_inputs(shard_root)
    if not inp.config.get("run_paired"):
        return []
    masks = _build_candidate_masks(inp.X, inp.providers, inp.is_public)
    _, _, folds = _public_folds(inp.X, inp.y, inp.groups, inp.is_public)
    return [f"{name}_f{fold}" for fold in range(len(folds)) for name in masks]


def run_one_study(shard_root, tag: str):
    """Run ONE nested-HPO study ``tag`` from the shared corpus (5c worker); writes ``<root>/<tag>.study.json``."""
    from scripts._study_shared import load_study_inputs

    inp = load_study_inputs(shard_root)
    cfg = inp.config
    masks = _build_candidate_masks(inp.X, inp.providers, inp.is_public)
    name, _sep, fold_s = tag.rpartition("_f")
    fold = int(fold_s)
    _, _, folds = _public_folds(inp.X, inp.y, inp.groups, inp.is_public)
    trainable = folds[fold][1]
    m = masks[name] & trainable
    _hpo_once(
        inp.X[m],
        inp.y[m],
        inp.groups[m],
        Path(cfg["study_db_dir"]),
        tag,
        cfg["n_trials"],
        objective_inputs=cfg["objective_inputs"],
        prov=cfg["run_prov"],
        negative_subsample=cfg["negative_subsample"],
        seed=cfg["seed"],
        study_shard_dir=shard_root,
    )
    return Path(shard_root) / f"{tag}.study.json"


def assemble_studies(shard_root, *, study_shard_dir=None, run_probe: bool = True):
    """Reduce: ship decision + final fit (+ TF-19 probe) over the studies (5c reduce), single-sourcing
    main's Phase 2/3. With ``study_shard_dir`` the nested studies come from the per-study cache (parallel
    workers); with it None they are computed inline -- so serial and parallel produce byte-identical
    weights. ``run_probe`` runs the (deterministic, model-derived) TF-19 reporting probe; the weight
    parity test sets it False since the probe needs a real ``_probe_sample`` on disk. Returns
    ``(metrics, model)``; raises :class:`AcceptanceGatesFailedError` instead of exiting."""
    sys.path.insert(0, "scripts")
    from _corpus import artifact_label, check_shipped_variant, corpus_identity, reproducibility
    from _paired import fixed_sequence_ship

    from scripts._study_shared import load_study_inputs
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
    from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel
    from silly_kicks.tracking._xshot_occurrence import subsample_negatives

    inp = load_study_inputs(shard_root)
    X, y, groups = inp.X, inp.y, inp.groups
    providers, match_ids, is_public = inp.providers, inp.match_ids, inp.is_public
    cfg = inp.config
    ns, seed = cfg["negative_subsample"], cfg["seed"]
    n_trials = cfg["n_trials"]
    out = Path(cfg["study_db_dir"])
    art = Path(cfg["artifact_dir"])
    run_prov = cfg["run_prov"]
    objective_inputs = cfg["objective_inputs"]  # D21, persisted by the prep
    ship_variant = cfg.get("ship_variant")
    run_paired = cfg["run_paired"]
    provset = {str(p) for p in providers.tolist()}

    # score_differential range probe (B6): guard the phantom-owngoal signature.
    sd = X["score_differential"].to_numpy(dtype=float)
    sd_fin = sd[np.isfinite(sd)]
    sd_probe = {
        "coverage": float(np.isfinite(sd).mean()),
        "min": float(sd_fin.min()) if sd_fin.size else float("nan"),
        "max": float(sd_fin.max()) if sd_fin.size else float("nan"),
        "abs_ge_12_count": int((np.abs(sd_fin) >= 12).sum()),
    }
    if sd_probe["abs_ge_12_count"] > 0:
        raise SystemExit(
            f"score_differential range probe FAILED (impossible |sd|>=12): {sd_probe}. "
            "Rebuild the feature cache on clean 4.13.0 GS events."
        )
    if sd_fin.size and np.abs(sd_fin).max() > 6:
        print(f"WARN score_differential |max|>6 (legit blowout possible): {sd_probe}", file=sys.stderr)

    candidates: dict = {}
    if run_paired:
        cand_masks = _build_candidate_masks(X, providers, is_public)
        paired = _paired_data_effect(
            X,
            y,
            groups,
            is_public,
            match_ids,
            candidates=cand_masks,
            n_trials=n_trials,
            out_dir=out,
            objective_inputs=objective_inputs,
            prov=run_prov,
            negative_subsample=ns,
            seed=seed,
            study_shard_dir=study_shard_dir,
        )
        full_vs_sc = [f - s for f, s in zip(paired["full"]["nested"], paired["sc_extended"]["nested"], strict=True)]
        shipped, why = fixed_sequence_ship(
            sc_extended=paired["sc_extended"]["nested"], full=paired["full"]["nested"], full_vs_sc=full_vs_sc
        )
        print(f"Fixed-sequence verdict: ship {shipped} -- {why}")
        if ship_variant is not None:
            why = f"operator --ship-variant override (fixed-sequence gate verdict was: {shipped} -- {why})"
            shipped = ship_variant
            print(f"Ship-variant override: forcing ship {shipped}. {why}")
        check_shipped_variant(cfg.get("expect_variant"), shipped)
        ship_mask = cand_masks[shipped]
        shipped_params = _hpo_once(
            X[ship_mask],
            y[ship_mask],
            groups[ship_mask],
            out,
            shipped,
            n_trials,
            objective_inputs=objective_inputs,
            prov=run_prov,
            negative_subsample=ns,
            seed=seed,
            study_shard_dir=study_shard_dir,
        )
        candidates[shipped] = {
            "params": shipped_params,
            "metrics": _cv_metrics(
                X[ship_mask], y[ship_mask], groups[ship_mask], shipped_params, negative_subsample=ns, seed=seed
            ),
            "providers": sorted(set(providers[ship_mask].tolist())),
        }
        candidates["paired"] = {
            "nested": {n: paired[n]["nested"] for n in cand_masks},
            "shared_params": {n: paired[n]["shared_params"] for n in cand_masks},
            "full_vs_sc": full_vs_sc,
            "shipped": shipped,
            "why": why,
        }
    else:
        if ship_variant is not None:
            raise SystemExit(
                "--ship-variant requires the multi-candidate (run_paired) corpus (public + owner "
                "SkillCorner + gradientsports) so the variant masks and the TF-19 probe cohort exist."
            )
        ship_mask = np.ones(len(X), bool)
        ship_provs = set(providers[ship_mask].tolist())
        shipped = artifact_label(providers=ship_provs, all_public=bool(is_public[ship_mask].all()))
        check_shipped_variant(cfg.get("expect_variant"), shipped)  # before the study: a refusal costs no fit
        params_all = _hpo_once(
            X,
            y,
            groups,
            out,
            "single",
            n_trials,
            objective_inputs=objective_inputs,
            prov=run_prov,
            negative_subsample=ns,
            seed=seed,
            study_shard_dir=study_shard_dir,
        )
        candidates[shipped] = {
            "params": params_all,
            "metrics": _cv_metrics(X, y, groups, params_all, negative_subsample=ns, seed=seed),
            "providers": sorted(provset),
        }

    shipped_metrics = candidates[shipped]["metrics"]
    acceptance = _gates(shipped_metrics)
    print(f"Shipped variant: {shipped}; gates: {acceptance}")
    art.mkdir(parents=True, exist_ok=True)
    if not all(acceptance.values()):
        json.dump(
            {"candidates": candidates, "acceptance": acceptance, "shipped_variant": shipped},
            open(art / "metrics_FAILED.json", "w"),
            indent=2,
        )
        print("ACCEPTANCE GATES FAILED -- refusing to write the bundled artifact.", file=sys.stderr)
        raise AcceptanceGatesFailedError(shipped)

    Xfit, yfit, _ = (
        subsample_negatives(X[ship_mask], y[ship_mask], y[ship_mask], fraction=ns, seed=seed)
        if ns
        else (X[ship_mask], y[ship_mask], None)
    )
    model = XCrossAttemptModel(params=candidates[shipped]["params"], feature_set=cfg["feature_set"])
    model.shipped_variant = shipped
    model.provider_list = candidates[shipped]["providers"]
    model.training_commit = run_prov["commit"]
    model.fit(Xfit, pd.Series(yfit), carrier_params=DEFAULT_CARRIER_PARAMS, horizon_seconds=cfg["horizon_seconds"])
    model.save(art)
    reloaded = XCrossAttemptModel.load(art)
    np.testing.assert_allclose(
        model.predict_proba(X[ship_mask].head(50)), reloaded.predict_proba(X[ship_mask].head(50)), rtol=0, atol=0
    )

    metrics = {
        "run_commit": run_prov["commit"],
        "run_tree_dirty": run_prov["dirty"],
        "run_tree_state": run_prov["tree_state"],
        "shipped_variant": shipped,
        "n_rows": len(X),
        "n_positive": int(np.asarray(y).sum()),
        "providers": sorted(provset),
        # Corpus IDENTITY (spec section 5): exact ids only for an all-public corpus, else a digest
        # (Hub publishes copy metrics.json). n_rows alone could not tell 17 public matches from 27.
        **corpus_identity(providers.tolist(), match_ids.tolist(), all_public=bool(is_public.all())),
        # ADR-067 M4 caveat, emitted here -- never hand-added at bundling (spec 0.11).
        **reproducibility(shipped, candidates[shipped]["providers"], training_commit=run_prov["commit"]),
        "candidates": candidates,
        "acceptance": acceptance,
        "estimates_are_cv_not_shipped_fit": True,
        "artifact_size_bytes": sum(f.stat().st_size for f in art.glob("*") if f.is_file()),
    }

    if run_probe:
        # Headline GK validations on the SHIPPED candidate (PR-B). Deterministic given (model, sample):
        # a parallel and a serial run produce the SAME probe. Requires the real _probe_sample on disk.
        from silly_kicks.tracking import _xcross_eval as ev

        shipped_params = candidates[shipped]["params"]
        gk_ablation = ev.gk_block_ablation(
            X[ship_mask], y[ship_mask], groups[ship_mask], shipped_params, seed=seed, negative_subsample=ns
        )
        perm_imp = ev.permutation_importance_report(
            X[ship_mask], y[ship_mask], groups[ship_mask], shipped_params, n_repeats=10, seed=seed
        )
        ps = art / "_probe_sample"
        if not (ps / "frames.parquet").exists():
            raise SystemExit(
                "Feature cache present but _probe_sample/ absent -> cannot run the TF-19 substitution "
                "probe (the headline deliverable). Delete the feature cache and re-extract, or restore "
                "the probe sample. Refusing to ship a spurious tf19_ready=False."
            )
        pf = pd.read_parquet(ps / "frames.parquet")
        pa = pd.read_parquet(ps / "actions.parquet")
        probe_meta = json.load(open(ps / "meta.json"))
        phome = probe_meta["home_team_id"]
        probe_matches = probe_meta.get("probe_matches", [])
        match_groups = probe_meta.get("match_groups", {})
        train_groups = set(groups[ship_mask].tolist())
        in_training = {mid: bool(set(g) & train_groups) for mid, g in match_groups.items()}
        probe_meta["in_training_folds"] = in_training
        json.dump(probe_meta, open(ps / "meta.json", "w"), indent=2)
        shipped_providers = set(candidates[shipped].get("providers", []))
        probe_provs_seen = {prov for prov, _mid in probe_matches}
        admitted = bool(run_paired and (not probe_provs_seen or probe_provs_seen & shipped_providers))
        gated_matches = _gated_probe_matches(probe_meta, admitted)
        if admitted:
            held_groups = {g for _prov, mid in gated_matches for g in match_groups.get(str(mid), [])}
            pf = pf[pf["game_id"].astype(str).isin(held_groups)]
            pa = pa[pa["game_id"].astype(str).isin(held_groups)]
            if pf.empty:
                raise SystemExit(
                    "Held-out probe matches resolved ZERO frames -- the probe sample is inconsistent "
                    "with meta.json match_groups. Delete the feature cache + probe sample and re-extract."
                )
        elif probe_matches and all(in_training.get(str(m[1]), False) for m in probe_matches):
            print(
                "NOTE: every probe match sits in the shipped training corpus (single-candidate path) -- "
                "the substitution-probe statistic below is NOT held-out.",
                file=sys.stderr,
            )
        probe = ev.gk_substitution_probe(model, pf, actions=pa, home_team_id=phome, n_frames=200, seed=seed)
        metrics.update(
            {
                "gk_block_ablation": gk_ablation,
                "gk_substitution_probe": probe,
                "permutation_importance": perm_imp,
                "score_differential_range_probe": sd_probe,
                "probe_sample_matches": probe_matches,
                "probe_sample_in_training_folds": in_training,
                "probe_gated_on_held_out": bool(admitted),
                "tf19_ready": probe.get("tf19_ready", False),
            }
        )
        if not probe.get("tf19_ready", False):
            print(
                f"NOTE: tf19_ready=False ({probe.get('tf19_reason')}) -- surface ships, but flagged "
                "NOT TF-19-ready (loud, not silent).",
                file=sys.stderr,
            )
    else:
        metrics["score_differential_range_probe"] = sd_probe

    json.dump(metrics, open(art / "metrics.json", "w"), indent=2)
    print(f"Wrote artifact + metrics to {art}")
    return metrics, model


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=False)  # not required for --study/--assemble workers
    src.add_argument("--data-dir")
    src.add_argument("--providers")
    ap.add_argument("--output-dir")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--max-per-provider", type=int, default=None)
    ap.add_argument("--horizon-seconds", type=float, default=1.0)
    ap.add_argument(
        "--negative-subsample",
        type=float,
        default=None,
        help="Thin this fraction of negatives in TRAIN folds only (never eval). Default OFF "
        "(crosses have a healthy base rate -- PA-M4 -- so subsampling is usually unnecessary).",
    )
    ap.add_argument("--seed", type=int, default=42, help="Seed for --negative-subsample (deterministic).")
    ap.add_argument(
        "--probe-providers",
        default="gradientsports",
        help="Comma list of providers eligible for the TF-19 substitution-probe capture (M5: the "
        "GATED cohort, persisted to _probe_sample/). Default: the gated gradientsports cohort.",
    )
    ap.add_argument(
        "--probe-comparison-providers",
        default="skillcorner",
        help="Comma list captured to _probe_sample_comparison/ -- the reported-not-gated "
        "same-population comparison leg (M5).",
    )
    ap.add_argument(
        "--probe-match-ids-json",
        default=None,
        help='JSON {"gradientsports": ["10502", ...]}: matches loaded ONLY for the TF-19 substitution probe, in '
        "a separate pass whose rows are discarded -- a held-out probe for a public-only fit (combined-cycle D5c).",
    )
    ap.add_argument(
        "--match-ids-json",
        default=None,
        help="JSON file mapping {provider: [match_id, ...]} -- a per-provider allowlist threaded to "
        "load_matches(match_ids=) (--providers path only). Default None (load every listed match).",
    )
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="Persist downloaded pining artifacts under CACHE_DIR/{provider}/{match_id}/ and reuse "
        "them on later runs over the same corpus. Default None re-downloads every run (~24-90 s per "
        "match). The cache is keyed on (provider, match_id) ONLY, so it would serve stale bytes if "
        "an upstream artifact were ever revised; these are immutable historical matches.",
    )
    ap.add_argument(
        "--feature-set",
        choices=["faithful", "position_only"],
        default="faithful",
        help="'faithful' (velocity-bearing, 16 feats) or 'position_only' (velocity dropped, 15 feats) "
        "for a model that scores on velocity-less SB360 freeze frames. 'extended' is NOT exposed here.",
    )
    ap.add_argument(
        "--ship-variant",
        choices=["public", "sc_extended", "full"],
        default=None,
        help="Operator override: force-ship THIS variant regardless of the fixed-sequence "
        "bundle-selection verdict. The gate decides the WHEEL bundle; the Hub sc_extended repo is the "
        "owner-tier ARCHIVE and holds sc_extended even when it does not clear the improvement gate "
        "(a marginal, noise-sensitive fold-consistency bar). The paired test still runs and its "
        "verdict + deltas are recorded in metrics.json (candidates.paired), so the override is fully "
        "audited; only the shipped model changes. Uses the SAME corpus mask the gate would, so the "
        "fitted model is identical to a gate-selected ship of that variant. Requires the "
        "multi-candidate (run_paired) corpus. Default None = normal gate-selected ship.",
    )
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Train from a modified working tree. The run still records run_tree_dirty=true in "
        "metrics.json -- the hatch permits a dev run, it never launders the fact.",
    )
    ap.add_argument(
        "--expect-variant",
        choices=["public", "sc_extended", "full"],
        default=None,
        help="G1 guard: refuse BEFORE extraction unless the requested corpus can ship this variant "
        "(public => every requested match is public), and refuse at ship time if the shipped variant "
        "differs. Default off (unchanged behaviour).",
    )
    ap.add_argument(
        "--shard-root",
        default=None,
        help="Study-shard root for the parallel study path (5c): a worker reads it via "
        "--study, the reduce via --assemble. Written by a serial run's prep.",
    )
    ap.add_argument(
        "--study",
        default=None,
        help="Run ONE study TAG ({candidate}_f{fold}) from --shard-root and exit (parallel worker).",
    )
    ap.add_argument(
        "--assemble",
        action="store_true",
        help="Reduce: ship decision + final fit + TF-19 probe over the studies in --shard-root, then exit.",
    )
    ap.add_argument(
        "--prep-only",
        action="store_true",
        help="extract + persist the study inputs, print {study_root, studies}, exit (launcher f1b mode)",
    )
    ap.add_argument("--list-studies", action="store_true", help="print the study tags under --shard-root as JSON")
    ap.add_argument(
        "--study-list", default=None, help="JSON list of study tags to run from --shard-root (launcher worker)"
    )
    args = ap.parse_args(argv)

    # Parallel study path (5c): a worker runs one study, the reduce assembles them -- both operate on an
    # already-persisted --shard-root (the serial prep enforced clean-tree + provenance).
    if args.study or args.assemble or args.study_list or args.list_studies:
        if not args.shard_root:
            ap.error("--study/--study-list/--list-studies/--assemble require --shard-root")
        root = Path(args.shard_root)
        if args.list_studies:
            print(json.dumps(enumerate_studies(root)))
        elif args.study_list:
            for tag in json.loads(Path(args.study_list).read_text(encoding="utf-8")):
                run_one_study(root, tag)
        elif args.study:
            run_one_study(root, args.study)
        else:
            try:
                assemble_studies(root, study_shard_dir=root, run_probe=True)
            except AcceptanceGatesFailedError:
                sys.exit(1)
        return

    if not args.output_dir:
        ap.error("--output-dir is required")
    if not (args.data_dir or args.providers):
        ap.error("one of --data-dir / --providers is required")

    # FIRST, before any corpus work. This trainer writes BUNDLED weights, and an artifact whose
    # provenance is unknown is one nobody can reproduce or audit later. ADR-052 enrolled all five
    # trainers at once, deliberately: a partial roll-out is how the same rule failed twice before.
    from scripts._provenance import git_provenance, require_clean_tree

    run_prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    ns, seed = args.negative_subsample, args.seed
    probe_provs = [p for p in args.probe_providers.split(",") if p]
    comparison_provs = [p for p in args.probe_comparison_providers.split(",") if p]
    if set(probe_provs) & set(comparison_provs):
        raise SystemExit("--probe-providers and --probe-comparison-providers must be disjoint.")
    if args.probe_match_ids_json:
        if not args.providers:
            ap.error("--probe-match-ids-json needs --providers (probe matches are listed via pining)")
        _probe = json.load(open(args.probe_match_ids_json))
        _pkeys = set(_probe)
        if not _pkeys <= set(probe_provs):
            ap.error(f"--probe-match-ids-json providers {sorted(_pkeys)} must all be in --probe-providers")
        # Held out by construction (B r4 CCC-PLAN-39): refuse any probe match the TRAINING pass could also load
        # -- a training provider with no allowlist loads its whole manifest, so every probe id under it overlaps.
        _train_provs = {p for p in args.providers.split(",") if p}
        _train_allow = json.load(open(args.match_ids_json)) if args.match_ids_json else None
        _overlap = [
            (p, str(m))
            for p, ids in _probe.items()
            for m in ids
            if p in _train_provs and (_train_allow is None or str(m) in {str(x) for x in _train_allow.get(p, [])})
        ]
        if _overlap:
            ap.error(
                f"--probe-match-ids-json names {len(_overlap)} match(es) the training corpus can also load; "
                "a held-out probe must be disjoint from training"
            )

    out = Path(args.output_dir)
    art = out / "xcross_attempt_v1"
    cache = art / "_feature_cache"
    sys.path.insert(0, "scripts")
    from _cache import cache_is_valid, write_cache_meta

    # G1 launch preflight (combined-cycle-completion spec section 5): BEFORE any corpus work, so a run
    # that would train a public bundle on owner-tier data never starts.
    if args.expect_variant is not None:
        if not args.providers:
            ap.error("--expect-variant needs --providers (only the pining path lists a requested corpus)")
        from _corpus import check_expected_variant, requested_is_all_public
        from _loader_pining import match_visibility, select_match_ids

        _provs = args.providers.split(",")
        _allow = json.load(open(args.match_ids_json)) if args.match_ids_json else None
        _pairs = select_match_ids(providers=_provs, match_ids=_allow, max_per_provider=args.max_per_provider)
        check_expected_variant(
            args.expect_variant, all_public=requested_is_all_public(_pairs, match_visibility(_provs))
        )

    # --- Phase 1: stream + extract + cache ---
    # M1: bound on BOTH branches (cache-hit never calls _extract).
    # Cache-validity guard: a pre-schema cache (no cache_meta.json, no match_ids.npy) MISSES, so the
    # DGX-populated caches that predate the visibility taxonomy are never silently reused. As of
    # ADR-050 the fingerprint is a LIVE per-corpus hash, so a cache built from a different corpus
    # under the same --output-dir also MISSES.
    probe_bundle = (_new_probe_cohort(), _new_probe_cohort(), 0)
    _fingerprint = _corpus_fingerprint(args)
    if cache_is_valid(cache, fingerprint=_fingerprint) and (cache / "match_ids.npy").exists():
        print(f"Loading cached features from {cache}")
        X = pd.read_parquet(cache / "features.parquet")
        y = np.load(cache / "labels.npy")
        groups = np.load(cache / "groups.npy", allow_pickle=True)
        providers = np.load(cache / "providers.npy", allow_pickle=True)
        match_ids = np.load(cache / "match_ids.npy", allow_pickle=True)
        if args.probe_match_ids_json:
            _meta_path = cache.parent / "_probe_sample" / "meta.json"
            if not _meta_path.is_file():
                raise SystemExit(
                    "no cached _probe_sample/meta.json for --probe-match-ids-json; use a fresh --output-dir"
                )
            _meta = json.load(open(_meta_path))
            _probe = json.load(open(args.probe_match_ids_json))
            _want = sorted([p, m] for p, ids in _probe.items() for m in ids)
            if sorted(_meta.get("probe_matches", [])) != _want:
                raise SystemExit("cached _probe_sample does not match --probe-match-ids-json; use a fresh --output-dir")
    else:
        if args.providers:
            allowlist = json.load(open(args.match_ids_json)) if args.match_ids_json else None
            sys.path.insert(0, "scripts")
            from _loader_pining import pining_source

            source, load = pining_source(
                args.providers.split(","),
                max_per_provider=args.max_per_provider,
                match_ids=allowlist,
                cache_dir=args.cache_dir,
            )
        else:
            source, load = _iter_matches_from_dir(Path(args.data_dir)), None
        t0 = time.time()
        X, y, groups, providers, match_ids, probe_bundle = _extract(
            source,
            args.horizon_seconds,
            feature_set=args.feature_set,
            load=load,
            # Shards live BESIDE the feature cache, under the same per-corpus `--output-dir`, so
            # the "fresh --output-dir per corpus" discipline the fingerprint enforces covers them.
            shard_root=art / "shards",
            probe_providers=() if args.probe_match_ids_json else tuple(probe_provs),
            probe_comparison_providers=tuple(comparison_provs),
        )
        if args.probe_match_ids_json:
            # D5(c): the TF-19 probe cohort comes from a SEPARATE pass over probe-only matches (e.g. GS
            # 10502/10503) whose feature rows are discarded -- they can never enter training.
            from _loader_pining import pining_source

            probe_allow = json.load(open(args.probe_match_ids_json))
            p_refs, p_load = pining_source(sorted(probe_allow), match_ids=probe_allow, cache_dir=args.cache_dir)
            *_discarded, p_bundle = _extract(
                p_refs,
                args.horizon_seconds,
                feature_set=args.feature_set,
                load=p_load,
                shard_root=art / "probe_shards",
                probe_providers=tuple(probe_provs),
                probe_comparison_providers=(),
            )
            probe_bundle = (p_bundle[0], probe_bundle[1], probe_bundle[2] + p_bundle[2])
        print(f"Extracted {len(X)} rows ({int(y.sum())} positives) in {time.time() - t0:.0f}s")
        cache.mkdir(parents=True, exist_ok=True)
        X.to_parquet(cache / "features.parquet")
        np.save(cache / "labels.npy", y)
        np.save(cache / "groups.npy", groups)
        np.save(cache / "providers.npy", providers)
        np.save(cache / "match_ids.npy", match_ids)
        # visibility.npy is deliberately NOT persisted: is_public is recomputed live every run from
        # cached providers + match_ids + the live manifest (below), so a persisted arm split would be
        # redundant AND could go stale. The schema bump is what invalidates pre-Task-11 caches.
        write_cache_meta(cache, fingerprint=_fingerprint)
        probe_cohort, comparison_cohort, n_skipped = probe_bundle  # TF-19 probe samples (fresh-extract only)
        # The probe cohort holds whole TRACKING FRAMES, which no tidy shard carries -- so a resumed
        # pass returns it EMPTY, and `_write_probe_sample` no-ops on empty. Left unguarded, resuming
        # a crashed extraction would silently produce a run with no TF-19 gate cohort at all: the
        # numbers would look complete and the gate would have nothing to stand on. Refuse instead,
        # unless the earlier pass already wrote the sample (in which case there is nothing to lose).
        for ps, cohort, provs in (
            (cache.parent / "_probe_sample", probe_cohort, probe_provs),
            (cache.parent / "_probe_sample_comparison", comparison_cohort, comparison_provs),
        ):
            if provs and n_skipped and not cohort["frames"] and not (ps / "meta.json").is_file():
                raise RuntimeError(
                    f"{n_skipped} match(es) were resumed from shards, so the probe cohort for "
                    f"{sorted(provs)} could not be captured and {ps} does not already exist. The "
                    f"probe needs whole tracking frames, which the shards do not carry. Re-run "
                    f"against a fresh --output-dir to rebuild it, or restore the earlier sample."
                )
            _write_probe_sample(ps, cohort, provs)

    groups = np.asarray(groups).astype(str)
    provset = {str(p) for p in providers.tolist()}
    sys.path.insert(0, "scripts")
    from _corpus import assert_public_corpus, is_public_row
    from _loader_pining import match_visibility

    # Public-vs-owner is keyed on the manifest visibility field, NEVER the provider name (spec 3.2):
    # the 98 owner-tier SkillCorner matches carry provider `skillcorner` but are non-redistributable.
    # The manifest is only fetchable on the --providers (pining) path; --data-dir has none -> {} ->
    # fail-closed all-private (is_public_row's default), which is correct for a local smoke corpus.
    vis = match_visibility(sorted(set(providers.tolist()))) if args.providers else {}
    loads_full_public_arm = {"skillcorner", "idsse"} <= set(providers.tolist()) and args.max_per_provider is None
    assert_public_corpus(vis, expect_full_public_arm=loads_full_public_arm)
    is_public = is_public_row(providers=providers, match_ids=match_ids, visibility=vis)
    # Outer gate for the 3-candidate nested paired test (Task 11 Step 4). Kept identical to the
    # pre-Task-11 predicate (NOT the plan's bare `mix` form): the `gradientsports` clause is what
    # makes `full` != `sc_extended` (owner GS rows), and it also keeps a public/owner SkillCorner-
    # only mix -- e.g. the Task-9 slow test's 1-public-game corpus -- out of the paired path, where
    # StratifiedGroupKFold(2) on a single public group would raise. The real maintainer run always
    # carries GS, so its behaviour is unchanged; only the paired-test INTERNALS became nested-HPO.
    # (`run_paired` is also the `admitted`-gate input for the TF-19 held-out probe, further below.)
    run_paired = bool(is_public.any() and (~is_public).any() and "gradientsports" in provset)

    # D21: shared store-identity parts for every HPO study this run opens; `_hpo_once` completes them
    # with its per-fold `tag`. Output / resume / orchestration knobs (output_dir, n_trials, allow_dirty,
    # and the 5c parallel-path shard_root/study/assemble) are excluded so they never fork the store key;
    # every other CLI arg is an input the tuning depends on. Persisted in the study config below so the
    # parallel workers and the reduce key their stores on the SAME identity as a serial run.
    resume_knobs = {"output_dir", "n_trials", "allow_dirty", "shard_root", "study", "assemble"}
    objective_inputs = {
        "driver": "train_xcross_attempt",
        "args": {k: v for k, v in vars(args).items() if k not in resume_knobs},
        "match_ids": sorted(map(str, match_ids.tolist())),
    }

    # --- Phase 2/3 (single-sourced with the parallel study path, 5c): persist the trial-invariant
    # inputs (the probe sample is already on disk beside the feature cache), then assemble. A serial run
    # computes the studies inline via the empty cache; a parallel launcher pre-fills them with
    # run_one_study workers -- either way assemble_studies produces the SAME shipped weights + probe
    # (studies deterministic; the reduce single-sourced). ---
    study_root = art / "studies"
    config = {
        "n_trials": args.n_trials,
        "negative_subsample": ns,
        "seed": seed,
        "feature_set": args.feature_set,
        "horizon_seconds": args.horizon_seconds,
        "study_db_dir": str(out),
        "artifact_dir": str(art),
        "run_paired": run_paired,
        "run_prov": run_prov,
        "objective_inputs": objective_inputs,
        "ship_variant": args.ship_variant,
        "expect_variant": args.expect_variant,
    }
    from scripts._study_shared import persist_study_inputs

    persist_study_inputs(
        study_root,
        X=X,
        y=y,
        groups=groups,
        providers=providers,
        match_ids=match_ids,
        is_public=is_public,
        config=config,
    )
    if args.prep_only:
        print(json.dumps({"study_root": str(study_root), "studies": enumerate_studies(study_root)}))
        return
    try:
        assemble_studies(study_root, study_shard_dir=study_root, run_probe=True)
    except AcceptanceGatesFailedError:
        sys.exit(1)


if __name__ == "__main__":
    main()
