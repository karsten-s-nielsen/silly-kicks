#!/usr/bin/env python
"""Train the xShotOccurrence (xS) model (TF-16 weights run, PR-S80).

Two match sources:
  --data-dir DIR     parquet dirs DIR/*/{frames,shots}.parquet (smoke / local corpus)
  --providers a,b,c  pining loader (skillcorner,idsse,gradientsports) for the maintainer run

Streams per match, caches features, and (on a public/owner mix with Gradient Sports) runs the
common-public-held-out PAIRED data-effect comparison over THREE candidates (public / sc_extended
/ full) with NESTED HPO -- each candidate re-tuned per outer fold with that fold's public games
excluded (spec 4.1, reviewer M4) -- then selects the shipped corpus via the registered fixed
sequence (scripts/_paired.py). Computes FAIL-CLOSED acceptance gates (spec S3, N3) and writes a
pickle-free artifact ONLY if the gates pass. Quality numbers in metrics.json are CV/protocol
estimates, not the shipped all-data fit (N7).

Requires: silly-kicks[train,xgboost]  (+ [kloppy] for --providers).
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
    from silly_kicks.tracking._xshot_occurrence import XShotFeatureSet

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]


def _corpus_fingerprint(args) -> str:
    """Fingerprint of the corpus THIS run requests, for cache validity (ADR-050).

    Replaces a constant ``"schema-v2"`` token that could invalidate a pre-schema cache but was
    blind to corpus DRIFT: a cache built from corpus A and reused under the same ``--output-dir``
    with a different ``--match-ids-json`` was silently accepted, which is why the operating rule
    had to be "use a fresh --output-dir per corpus" -- a discipline, not a guard.

    Keyed on the REQUESTED corpus (``select_match_ids``), not the extracted one, and on the same
    selection helper ``load_matches`` uses, so the fingerprint cannot describe a corpus the
    extraction never loaded. Costs one manifest listing on the cache-hit path; that was the reason
    it was deferred, and it is worth paying for a guard that actually detects the failure.
    """
    sys.path.insert(0, "scripts")
    from _cache import corpus_fingerprint

    if not args.providers:
        # --data-dir: a local smoke corpus with no manifest. Fingerprint the directory contents.
        d = Path(args.data_dir)
        rows = [("local", p.name, "private") for p in sorted(d.iterdir()) if p.is_dir()]
        return corpus_fingerprint(rows)

    from _loader_pining import match_visibility, select_match_ids

    providers = args.providers.split(",")
    allowlist = json.load(open(args.match_ids_json)) if args.match_ids_json else None
    pairs = select_match_ids(providers=providers, match_ids=allowlist, max_per_provider=args.max_per_provider)
    vis = match_visibility(providers)
    # visibility is part of the identity, not decoration: the same match can move public->private
    # in the manifest, which changes which training arm it belongs to.
    return corpus_fingerprint([(p, m, vis.get((p, m), "private")) for p, m in pairs])


def _iter_matches_from_dir(data_dir: Path):
    for game_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        frames = pd.read_parquet(game_dir / "frames.parquet")
        shots = pd.read_parquet(game_dir / "shots.parquet")
        prov = str(frames["source_provider"].iloc[0]) if "source_provider" in frames.columns else "unknown"
        yield prov, game_dir.name, shots, frames, frames["team_id"].dropna().iloc[0]


def _source_key(item):
    """The `for_each` key for BOTH sources: a `MatchRef` (pining) -> its `.key`, a --data-dir tuple
    -> `(provider, match_id)`. Module-level so the key-pin gate can see it (_KEY_EXCEPTIONS)."""
    key = getattr(item, "key", None)
    return key if key is not None else (str(item[0]), str(item[1]))


#: The four per-row arrays `_extract` returns alongside the feature matrix, carried as COLUMNS so
#: one match is one tidy shard. Underscore-prefixed and collision-checked below: a feature named
#: `_y` would be silently overwritten, and the model would train on the label.
_SIDE_COLS = ("_y", "_group", "_provider", "_match_id")


def _extract(
    source, horizon_seconds, *, shard_root, feature_set: XShotFeatureSet = "faithful", load=None
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scripts._driver import for_each, shard_path
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
    from silly_kicks.tracking._xshot_occurrence import (
        XSHOT_FEATURE_NAMES_FAITHFUL,
        XSHOT_FEATURE_NAMES_POSITION_ONLY,
        prepare_xshot_training_data,
    )

    _feature_names = (
        XSHOT_FEATURE_NAMES_POSITION_ONLY if feature_set == "position_only" else XSHOT_FEATURE_NAMES_FAITHFUL
    )

    collision = set(_SIDE_COLS) & set(XSHOT_FEATURE_NAMES_FAITHFUL)
    if collision:
        raise ValueError(f"side columns {sorted(collision)} collide with feature names")

    def _work(item):
        prov, mid, actions_or_shots, frames, home, *_ = item
        X, y, groups = prepare_xshot_training_data(
            frames,
            actions_or_shots,
            home_team_id=home,
            feature_set=feature_set,
            horizon_seconds=horizon_seconds,
            attacking_third_only=True,
            carrier_params=DEFAULT_CARRIER_PARAMS,  # 4.7.0 values; shared constant (anti-drift)
        )
        del frames
        if not len(X):
            return None  # still writes an EMPTY shard: "ran, produced no usable row"
        return X.assign(
            _y=np.asarray(y, int),
            _group=np.asarray(groups),
            _provider=str(prov),
            _match_id=str(mid),  # per-row pining match_id (visibility key)
        )

    # The `_cache.py` layer above this is a WHOLE-CORPUS fast path -- all-or-nothing, and it only
    # helps a run that already completed once. These shards make the extraction ITSELF resumable, so
    # the two nest rather than compete (same arrangement as `cohort_cache` over `for_each` in
    # `calibrate_xt_bandwidth`). A crash at match 70 of 80 now costs 10 matches, not 80.
    res = for_each(
        source,
        key=_source_key,
        load=load,
        work=_work,
        shard_root=shard_root,
        # What determines a shard's CONTENT: the extractor, the label horizon, the domain filter,
        # and the carrier params the features are built against. `--providers` /
        # `--max-per-provider` / `--match-ids-json` only choose WHICH matches are walked, and the
        # key separates them; the HPO and the gates consume these rows downstream.
        token_inputs={
            "extractor": "prepare_xshot_training_data",
            # feature_set changes the X columns (27 vs 26), so it MUST key the shard generation, or a
            # faithful run's shards get reused for a position_only run (the 4.77.1 stale-shard trap).
            "feature_set": feature_set,
            "horizon_seconds": horizon_seconds,
            "attacking_third_only": True,
            "carrier_params": dict(DEFAULT_CARRIER_PARAMS),
        },
        tag="xshot_features",
        label="match",
    )
    if res.failures:
        raise RuntimeError(f"{len(res.failures)} match(es) failed: {res.failures}. Re-run to retry only them.")

    # Combined from THIS PASS'S keys, not `_driver.reconcile`: no partition surface here, so a
    # whole-generation read would fold in matches from a wider earlier run. See its docstring.
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
    )


def _hpo_once(X, y, groups, out_dir, tag, n_trials, *, negative_subsample=None, seed=42, study_shard_dir=None) -> dict:
    """Run ruthless HPO once for one candidate; return the frozen best-params dict.

    ``negative_subsample`` thins negatives in TRAIN folds only (never eval) inside the objective.

    ``study_shard_dir`` (opt-in) caches the frozen params to ``<dir>/<tag>.study.json``: an existing
    shard is loaded and returned (resume), else the computed params are written after HPO. The study
    is deterministic (seeded TPE + a tag-keyed sqlite store), so a cached result is byte-identical to
    an in-process one -- this is what lets the ~15 nested studies run as independent parallel workers
    (5c) and be assembled with no loss of quality (spec 6). JSON round-trips finite floats exactly.
    """
    if study_shard_dir is not None:
        shard = Path(study_shard_dir) / f"{tag}.study.json"
        if shard.exists():
            return dict(json.loads(shard.read_text(encoding="utf-8"))["params"])

    from ruthless import Direction, FloatRange, InProcessBackend, OptunaConfig
    from ruthless.config.common import StoreConfig
    from ruthless.strategies.optuna_ import OptunaStrategy

    from silly_kicks.tracking._xshot_occurrence_objective import XShotOccurrenceObjective

    obj = XShotOccurrenceObjective(
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
        store=StoreConfig(kind="sqlite", path=str(out_dir / f"study_{tag}.db")),
    )
    result = OptunaStrategy(cfg, seed=42).run(obj, backend=InProcessBackend())
    if result.best is None:
        raise RuntimeError("HPO produced no best candidate")
    params = dict(result.best.candidate.params)
    if study_shard_dir is not None:
        shard = Path(study_shard_dir) / f"{tag}.study.json"
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_text(json.dumps({"tag": tag, "params": params}), encoding="utf-8")
    return params


def _cv_metrics(X, y, groups, params, *, negative_subsample=None, seed=42) -> dict:
    """Label-stratified, match-grouped CV at FIXED params -> gate metrics on the TRUE balance.

    ``negative_subsample`` thins negatives in the TRAIN fold only; the held-out fold (and hence
    every reported metric + the base-rate baselines) always uses the true, unsubsampled balance
    (PR-S80 M3 -- the fix for the prior pre-split contamination footgun).
    """
    import xgboost as xgb
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
    from sklearn.model_selection import StratifiedGroupKFold

    from silly_kicks.tracking._xshot_occurrence import _pinned_params, subsample_negatives

    n_splits = max(2, min(5, len(np.unique(groups))))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    prs, brs, lls = [], [], []
    for fold_i, (tr, te) in enumerate(skf.split(X, y, groups)):
        if len(np.unique(y[tr])) < 2:
            continue
        Xtr, ytr = X.iloc[tr], y[tr]
        if negative_subsample:  # TRAIN fold only; eval fold (te) keeps the true balance
            Xtr, ytr, _ = subsample_negatives(Xtr, ytr, ytr, fraction=negative_subsample, seed=seed + fold_i)
            if len(np.unique(ytr)) < 2:
                continue
        p_ = dict(_pinned_params(params))
        p_["base_score"] = float(ytr.mean())
        clf = xgb.XGBClassifier(**p_)
        clf.fit(Xtr.to_numpy(float), ytr)
        p = clf.predict_proba(X.iloc[te].to_numpy(float))[:, 1]
        lls.append(log_loss(y[te], p, labels=[0, 1]))
        brs.append(brier_score_loss(y[te], p))
        if len(np.unique(y[te])) == 2:
            prs.append(average_precision_score(y[te], p))
    base = float(y.mean())
    return {
        "pr_auc": float(np.mean(prs)) if prs else float("nan"),
        "brier": float(np.mean(brs)) if brs else float("nan"),
        "log_loss": float(np.mean(lls)) if lls else float("inf"),
        "pr_auc_std": float(np.std(prs)) if prs else float("nan"),
        "positive_rate": base,
        "base_rate_brier": base * (1 - base),
        "n_usable_folds": len(lls),  # P5
    }


def _gates(m: dict) -> dict:
    pr = m["pr_auc"]
    br = m["brier"]
    return {
        "enough_usable_folds": m.get("n_usable_folds", 0) >= 2,  # P5
        "pr_auc_gt_base_rate": bool(pr == pr and pr > m["positive_rate"]),  # NaN-safe strict
        "brier_lt_base_rate_brier": bool(br == br and br < m["base_rate_brier"]),
        "log_loss_lt_uniform": m["log_loss"] < float(np.log(2)),
    }


def _fit_score(X_tr, y_tr, X_te, y_te, params, *, negative_subsample=None, seed=42) -> float:
    """Fit XGBoost at ``params`` on (X_tr, y_tr); return PR-AUC on (X_te, y_te).

    A module-level extraction of the old ``_paired_data_effect`` closure, with two things made
    EXPLICIT that the closure hardcoded:
      * ``params``  -- so ONE code path serves both protocols: the candidate's OWN tuned params
                       (nested, PRIMARY, decides the ship) and the public params (shared, reported).
      * the eval slice (X_te, y_te) -- the closure captured Xp/yp from its enclosing scope.

    Degenerate (single-class) folds return NaN; the caller drops them. Preserves the closure's
    base_score override and train-only negative subsampling (PR-S80 M3).
    """
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import average_precision_score

    from silly_kicks.tracking._xshot_occurrence import _pinned_params, subsample_negatives

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


def _fit_study_for_test(X, y, groups, tag, n_trials, seed=42):
    """Test-only seam: run one HPO study on ``(X, y, groups)`` and return ``(params, booster)``.

    Mirrors the shipped fit path (``.to_numpy(dtype=float)`` then ``booster.feature_names =
    list(columns)``, _xshot_occurrence.py:480-482) so a study run on the shared-mmap corpus can be
    proven byte-identical to an in-memory one -- values AND feature-names. Not used in production.
    """
    import tempfile

    import xgboost as xgb

    from silly_kicks.tracking._xshot_occurrence import _pinned_params

    y = np.asarray(y)
    d = tempfile.mkdtemp(prefix="xshot_study_")  # test seam: OS temp, no lock-sensitive cleanup
    params = _hpo_once(X, y, np.asarray(groups), Path(d), tag, n_trials, seed=seed)
    p_ = dict(_pinned_params(params))
    p_["base_score"] = float(y.mean())
    clf = xgb.XGBClassifier(**p_)
    clf.fit(X.to_numpy(dtype=float), y)
    booster = clf.get_booster()
    booster.feature_names = list(X.columns)
    return params, booster


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
    negative_subsample=None,
    seed=42,
    study_shard_dir=None,
) -> dict:
    """Nested-HPO paired comparison on the common public held-out folds (spec 4.1, reviewer M4).

    The historical version tuned HPO ONCE, outside the outer CV, on the public arm -- so `public`
    tuned on exactly the matches that ARE the evaluation universe (differential leakage favouring
    `public`, deciding the ship). Here, for each outer fold k, EVERY candidate is tuned on its OWN
    training data with fold k's public games EXCLUDED, then fitted at those params and scored on
    fold k. No candidate's params ever see the fold they are scored on. `candidates` maps
    name -> row mask. Returns, per candidate, per-fold PR-AUC deltas vs `public` under both
    protocols:
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
    """The shipped candidate failed a fail-closed acceptance gate (N3). main() maps this to exit 1."""


def _build_candidate_masks(X, providers, is_public) -> dict:
    """The 3 paired candidates, single-sourced so main, ``run_one_study`` and ``assemble_studies``
    build identical row masks (the ``gradientsports`` owner rows are what make ``full`` != ``sc_extended``).
    """
    is_sc_private = (providers == "skillcorner") & ~is_public  # owner-tier SkillCorner rows
    return {
        "public": is_public,
        "sc_extended": is_public | is_sc_private,
        "full": np.ones(len(X), bool),
    }


def enumerate_studies(shard_root) -> list[str]:
    """The parallel study tags ``{candidate}_f{fold}`` for a run, derived from the persisted inputs.

    Empty for a single-candidate (non-paired) corpus -- there is nothing to fan out there.
    """
    from scripts._study_shared import load_study_inputs

    inp = load_study_inputs(shard_root)
    if not inp.config.get("run_paired"):
        return []
    masks = _build_candidate_masks(inp.X, inp.providers, inp.is_public)
    _, _, folds = _public_folds(inp.X, inp.y, inp.groups, inp.is_public)
    return [f"{name}_f{fold}" for fold in range(len(folds)) for name in masks]


def run_one_study(shard_root, tag: str):
    """Run ONE nested-HPO study ``tag`` (``{candidate}_f{fold}``) from the shared corpus (5c worker).

    Reconstructs exactly the rows the serial ``_paired_data_effect`` would feed ``_hpo_once`` for that
    ``(candidate, fold)`` and writes ``<shard_root>/<tag>.study.json`` (the study-shard cache). Running
    every enumerated tag then :func:`assemble_studies` is byte-identical to the serial run.
    """
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
        negative_subsample=cfg["negative_subsample"],
        seed=cfg["seed"],
        study_shard_dir=shard_root,
    )
    return Path(shard_root) / f"{tag}.study.json"


def assemble_studies(shard_root, *, study_shard_dir=None):
    """Reduce: the ship decision + final fit over the (cached-or-computed) studies (5c reduce).

    Single-sources main's Phase 2/3. ``study_shard_dir`` lets the nested studies come from the
    per-study cache (pre-filled by parallel :func:`run_one_study` workers); with it None the studies
    are computed inline here -- so a serial ``main`` call and a parallel launcher call run the SAME
    code and produce byte-identical weights. Returns ``(metrics, model)``; raises
    :class:`AcceptanceGatesFailedError` instead of exiting so it is safe to call as a library.
    """
    sys.path.insert(0, "scripts")
    from _corpus import artifact_label, check_shipped_variant, corpus_identity, reproducibility
    from _paired import fixed_sequence_ship

    from scripts._study_shared import load_study_inputs
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel, subsample_negatives

    inp = load_study_inputs(shard_root)
    X, y, groups = inp.X, inp.y, inp.groups
    providers, match_ids, is_public = inp.providers, inp.match_ids, inp.is_public
    cfg = inp.config
    ns, seed = cfg["negative_subsample"], cfg["seed"]
    n_trials = cfg["n_trials"]
    out = Path(cfg["study_db_dir"])
    art = Path(cfg["artifact_dir"])
    run_prov = cfg["run_prov"]
    provset = {str(p) for p in providers.tolist()}

    candidates: dict = {}
    if cfg["run_paired"]:
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
            negative_subsample=ns,
            seed=seed,
            study_shard_dir=study_shard_dir,
        )
        full_vs_sc = [f - s for f, s in zip(paired["full"]["nested"], paired["sc_extended"]["nested"], strict=True)]
        shipped, why = fixed_sequence_ship(
            sc_extended=paired["sc_extended"]["nested"], full=paired["full"]["nested"], full_vs_sc=full_vs_sc
        )
        print(f"Fixed-sequence verdict: ship {shipped} -- {why}")
        check_shipped_variant(cfg.get("expect_variant"), shipped)
        ship_mask = cand_masks[shipped]
        shipped_params = _hpo_once(
            X[ship_mask],
            y[ship_mask],
            groups[ship_mask],
            out,
            shipped,
            n_trials,
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
        ship_mask = np.ones(len(X), bool)
        ship_provs = set(providers[ship_mask].tolist())
        shipped = artifact_label(providers=ship_provs, all_public=bool(is_public[ship_mask].all()))
        check_shipped_variant(cfg.get("expect_variant"), shipped)  # before the study: a refusal costs no fit
        params_all = _hpo_once(
            X, y, groups, out, "single", n_trials, negative_subsample=ns, seed=seed, study_shard_dir=study_shard_dir
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
    model = XShotOccurrenceModel(params=candidates[shipped]["params"], feature_set=cfg["feature_set"])
    model.shipped_variant = shipped
    model.provider_list = candidates[shipped]["providers"]
    model.training_commit = run_prov["commit"]
    model.fit(Xfit, pd.Series(yfit), carrier_params=DEFAULT_CARRIER_PARAMS, horizon_seconds=cfg["horizon_seconds"])
    model.save(art)
    reloaded = XShotOccurrenceModel.load(art)
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
        help="Thin this fraction of negatives in TRAIN folds only (never eval), for wall-clock/memory "
        "control on very large corpora. Default OFF -- the maintainer run uses the full true balance.",
    )
    ap.add_argument("--seed", type=int, default=42, help="Seed for --negative-subsample (deterministic).")
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
        help="'faithful' (velocity-bearing, 27 feats) or 'position_only' (velocity dropped, 26 feats) "
        "for a model that scores on velocity-less SB360 freeze frames. 'extended' is NOT exposed here "
        "(it raises in prepare).",
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
        help="Study-shard root for the parallel study path (5c). Written by a serial run's prep; a "
        "worker reads it via --study, the reduce via --assemble.",
    )
    ap.add_argument(
        "--study",
        default=None,
        help="Run ONE study TAG ({candidate}_f{fold}) from --shard-root and exit (parallel worker). "
        "The corpus + masks must already be persisted there.",
    )
    ap.add_argument(
        "--assemble",
        action="store_true",
        help="Reduce: ship decision + final fit over the studies in --shard-root, then exit.",
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

    # Parallel study path (5c): a worker runs one study, the reduce assembles them. Both operate on an
    # already-persisted --shard-root (the serial prep enforced clean-tree + provenance), so they skip
    # the extraction pipeline entirely.
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
                assemble_studies(root, study_shard_dir=root)
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

    out = Path(args.output_dir)
    art = out / "xshot_occurrence_v1"
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
    # Cache-validity guard: a pre-schema cache (no cache_meta.json, no match_ids.npy) MISSES, so the
    # DGX-populated caches that predate the visibility taxonomy are never silently reused. As of
    # ADR-050 the fingerprint is a LIVE per-corpus hash, so a cache built from a different corpus
    # under the same --output-dir also MISSES -- the "fresh --output-dir per corpus" discipline is
    # now enforced rather than documented.
    _fingerprint = _corpus_fingerprint(args)
    if cache_is_valid(cache, fingerprint=_fingerprint) and (cache / "match_ids.npy").exists():
        print(f"Loading cached features from {cache}")
        X = pd.read_parquet(cache / "features.parquet")
        y = np.load(cache / "labels.npy")
        groups = np.load(cache / "groups.npy", allow_pickle=True)
        providers = np.load(cache / "providers.npy", allow_pickle=True)
        match_ids = np.load(cache / "match_ids.npy", allow_pickle=True)
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
        # Shards live BESIDE the feature cache, under the same per-corpus `--output-dir`, so the
        # "fresh --output-dir per corpus" discipline the fingerprint enforces covers them too.
        X, y, groups, providers, match_ids = _extract(
            source, args.horizon_seconds, shard_root=art / "shards", feature_set=args.feature_set, load=load
        )
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

    # game_id dtype is provider-asymmetric (kloppy str vs GS int) -> normalize cross-provider
    # groups to str so np.unique / StratifiedGroupKFold can sort them (the model never uses game_id).
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
    run_paired = bool(is_public.any() and (~is_public).any() and "gradientsports" in provset)

    # --- Phase 2/3 (single-sourced with the parallel study path, 5c): persist the trial-invariant
    # inputs, then assemble. A serial run computes the studies inline via the empty cache; a parallel
    # launcher pre-fills them with run_one_study workers -- either way assemble_studies produces the
    # SAME shipped weights (studies are deterministic; the reduce is single-sourced). ---
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
        assemble_studies(study_root, study_shard_dir=study_root)
    except AcceptanceGatesFailedError:
        sys.exit(1)


if __name__ == "__main__":
    main()
