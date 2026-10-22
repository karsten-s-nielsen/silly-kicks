#!/usr/bin/env python
"""Train the xShotOccurrence (xS) model (TF-16 weights run, PR-S80).

Usage (ADR-128 stage contract; the uniform CLI from ``scripts._stages.main``)::

    python -m scripts.train_xshot_occurrence --dataset tc3 --arms-json '<launcher payload>' \
        --store <DIR> --data-root <DATA> --run-all --out <BUNDLE_DIR>

Pipeline: ``extract[match]`` (an ``ItemStage`` over the staged flat dataset: frames + the ``_actions`` shot
rows) -> ``prep`` (the study inputs + the ``--arms-json`` candidate masks) -> ``study[tag]`` (nested HPO;
each trial's CV folds run on the InnerPool) -> ``cv[arm, fold]`` -> ``select`` -> ``fit`` -> ``assemble``.

On a public/owner mix with Gradient Sports it runs the common-public-held-out PAIRED data-effect comparison
over THREE candidates (public / sc_extended / full) with NESTED HPO -- each candidate re-tuned per outer
fold with that fold's public games excluded (spec 4.1, reviewer M4) -- then selects the shipped corpus via
the registered fixed sequence (scripts/_paired.py). Computes FAIL-CLOSED acceptance gates (spec S3, N3) and
writes a pickle-free artifact ONLY if the gates pass. Quality numbers in metrics.json are CV/protocol
estimates, not the shipped all-data fit (N7).

Requires: silly-kicks[train,xgboost].
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `from scripts._X` resolves

from scripts._stages import Param, Pipeline, Stage, main
from scripts._xshot_stages import STAGES as _REUSABLE  # ADR-129: the reusable extract stage

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]


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
    fold_map=None,
) -> dict:
    """Run ruthless HPO once for one candidate; return the frozen best-params dict.

    ``fold_map`` (ADR-128 ruling 4) evaluates each trial's CV folds in parallel (an InnerPool); the
    objective combines them in sorted fold order, so the study is numerically unchanged.

    ``negative_subsample`` thins negatives in TRAIN folds only (never eval) inside the objective.
    ``objective_inputs`` (D21) carries the run's shared identity parts (driver/args/match_ids); this
    seam COMPLETES them with its per-fold ``tag`` (the trainer's per-site extra) and ``prov`` into the
    store's ``objective_id``, so each fold's study keys on its own tag and a dirty tree never resumes.

    ``study_shard_dir`` (opt-in) caches the frozen params to ``<dir>/<tag>.study.json``: a shard written
    under THIS call's ``objective_id`` and ``n_trials`` is loaded and returned (resume), else HPO runs and
    the shard is (re)written -- the store's own D21 resume rule, so a shard from another run is never
    served stale and a dirty tree never resumes from one (a dirty-tree parallel run therefore recomputes
    its studies in the reduce). The study is deterministic (seeded TPE + a tag-keyed sqlite store), so a
    cached result is byte-identical to an in-process one -- this is what lets the ~15 nested studies run
    as independent parallel workers (5c) and be assembled with no loss of quality (spec 6). JSON
    round-trips finite floats exactly.
    """
    from scripts._input_contract import declare_inputs
    from scripts._provenance import objective_id, store_path_for
    from scripts._study_shared import read_study_shard, write_study_shard
    from silly_kicks.tracking._xshot_occurrence_objective import XShotOccurrenceObjective

    # The id depends on the objective CLASS, never its data, so it is known before the shard check.
    oid = objective_id(XShotOccurrenceObjective, declare_inputs(**objective_inputs, tag=tag), prov=prov)
    if study_shard_dir is not None:
        cached = read_study_shard(study_shard_dir, tag, objective_id=oid, n_trials=n_trials)
        if cached is not None:
            return cached

    from ruthless import Direction, FloatRange, InProcessBackend, OptunaConfig
    from ruthless.config.common import StoreConfig
    from ruthless.strategies.optuna_ import OptunaStrategy

    obj = XShotOccurrenceObjective(
        fold={tag: [(X, pd.Series(y), groups)]},
        negative_subsample=negative_subsample,
        subsample_seed=seed,
        fold_map=fold_map,
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


def _cv_metrics_plan(y, groups) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """The seeded split `_cv_metrics` walks, single-sourced so the ``cv`` stage's fold units walk the SAME
    folds. ``StratifiedGroupKFold.split`` reads ``X`` only for its length, so a zeros column stands in."""
    from sklearn.model_selection import StratifiedGroupKFold

    n_splits = max(2, min(5, len(np.unique(groups))))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y = np.asarray(y)
    return [(i, tr, te) for i, (tr, te) in enumerate(skf.split(np.zeros((len(y), 1)), y, groups))]


def _cv_metrics_fold(X, y, fold_i, tr, te, params, negative_subsample=None, seed=42):
    """One fold of `_cv_metrics`: ``(log_loss, brier, pr_auc | None)``, or None for a skipped fold."""
    import xgboost as xgb
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

    from silly_kicks.tracking._xshot_occurrence import _pinned_params, subsample_negatives

    y = np.asarray(y)
    if len(np.unique(y[tr])) < 2:
        return None
    Xtr, ytr = X.iloc[tr], y[tr]
    if negative_subsample:  # TRAIN fold only; eval fold (te) keeps the true balance
        Xtr, ytr, _ = subsample_negatives(Xtr, ytr, ytr, fraction=negative_subsample, seed=seed + fold_i)
        if len(np.unique(ytr)) < 2:
            return None
    p_ = dict(_pinned_params(params))
    p_["base_score"] = float(ytr.mean())
    clf = xgb.XGBClassifier(**p_)
    clf.fit(Xtr.to_numpy(float), ytr)
    p = clf.predict_proba(X.iloc[te].to_numpy(float))[:, 1]
    pr = float(average_precision_score(y[te], p)) if len(np.unique(y[te])) == 2 else None
    return (float(log_loss(y[te], p, labels=[0, 1])), float(brier_score_loss(y[te], p)), pr)


def _cv_metrics_combine(fold_results, positive_rate: float) -> dict:
    """The gate metrics from the folds in FOLD order, on the TRUE balance (``positive_rate`` = y.mean())."""
    kept = [r for r in fold_results if r is not None]
    lls = [r[0] for r in kept]
    brs = [r[1] for r in kept]
    prs = [r[2] for r in kept if r[2] is not None]
    base = float(positive_rate)
    return {
        "pr_auc": float(np.mean(prs)) if prs else float("nan"),
        "brier": float(np.mean(brs)) if brs else float("nan"),
        "log_loss": float(np.mean(lls)) if lls else float("inf"),
        "pr_auc_std": float(np.std(prs)) if prs else float("nan"),
        "positive_rate": base,
        "base_rate_brier": base * (1 - base),
        "n_usable_folds": len(lls),  # P5
    }


def _cv_metrics(X, y, groups, params, *, negative_subsample=None, seed=42) -> dict:
    """Label-stratified, match-grouped CV at FIXED params -> gate metrics on the TRUE balance.

    ``negative_subsample`` thins negatives in the TRAIN fold only; the held-out fold (and hence
    every reported metric + the base-rate baselines) always uses the true, unsubsampled balance
    (PR-S80 M3 -- the fix for the prior pre-split contamination footgun). ADR-128: the composition of
    `_cv_metrics_plan` / `_cv_metrics_fold` / `_cv_metrics_combine` (exact-equality tested).
    """
    folds = [
        _cv_metrics_fold(X, y, i, tr, te, params, negative_subsample, seed) for i, tr, te in _cv_metrics_plan(y, groups)
    ]
    return _cv_metrics_combine(folds, float(np.asarray(y).mean()))


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
    # D21: a fixed clean identity -- this seam is test-only and its store is a fresh tempdir.
    params = _hpo_once(
        X,
        y,
        np.asarray(groups),
        Path(d),
        tag,
        n_trials,
        objective_inputs={"driver": "train_xshot_occurrence._fit_study_for_test"},
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
    params_for=None,
) -> dict:
    """Nested-HPO paired comparison on the common public held-out folds (spec 4.1, reviewer M4).

    ``params_for(tag)`` (ADR-128) serves each ``{name}_f{fold}`` study's frozen params from the ``study``
    stage instead of running ``_hpo_once`` here; the studies are deterministic, so the deltas are identical.

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
            name: params_for(f"{name}_f{fold}")
            if params_for is not None
            else _hpo_once(
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
    """The shipped candidate failed a fail-closed acceptance gate (N3). main() maps this to exit 1."""


def _build_candidate_masks(match_keys, armspec) -> dict:
    """The candidate row masks, built PURELY from the --arms parameter (ADR-123 Amendment B).

    An arm is a set of dataset match-keys supplied by the launcher; the mask is set membership. There is
    NO provider-name and NO public/private logic in-job -- policy lives off-HF in the launcher. ``main``
    builds the masks once and persists them; the fan-out workers + the reduce read ``StudyInputs.arms``.
    """
    from scripts.hf_jobs.arms import arm_masks

    return arm_masks(match_keys, armspec)


# ---- ADR-128 stage contract --------------------------------------------------------------------------
#
# extract[match] -> prep (one unit: the study inputs) -> study[tag] (inner_parallel: a trial's CV folds run
# on the InnerPool, ruling 4) -> cv[arm, fold] -> select (the ship decision) -> fit (gates + the final fit)
# -> assemble (one unit, the bundle). Every stage single-sources the legacy computation it replaced.
# A legacy line reference (`:N-M`) points at this file at `cd2ebcb`, the last version before the contract.

#: Decision R1 (spec 5.3), DECIDED by measurement (tests/scripts/test_study_resume_r1.py, 2026-10-18): a
#: study whose store is resumed after an interruption does NOT reproduce the uninterrupted study (ruthless
#: 0.7.0: 3 + 3 resumed trials returned the 3-trial best, not the 6-trial one). So the store lives in a
#: unit-local temp dir and an interrupted study unit restarts from trial 0 (its ledger shows attempt > 1).
STUDY_RESUME = False

_PARAMS = (
    Param("dataset", str, None, required=True, help="the staged flat dataset (shards/<gen>/ + _actions/)"),
    Param(
        "arms_json",
        str,
        None,
        required=True,
        help="JSON {arms: {name: [in-job match key, ...]}, ship_order: [...]} of candidate-arm memberships, "
        "computed OFF-HF by the launcher from the public/owner taxonomy (ADR-123 Amendment B). The job applies "
        "it mechanically (np.isin) with NO public/private concept of its own.",
    ),
    Param("n_trials", int, 50),
    Param("horizon_seconds", float, 1.0),
    Param(
        "negative_subsample",
        float,
        None,
        help="Thin this fraction of negatives in TRAIN folds only (never eval), for wall-clock/memory "
        "control on very large corpora. Default OFF -- the maintainer run uses the full true balance.",
    ),
    Param("seed", int, 42, help="Seed for --negative-subsample (deterministic)."),
    Param(
        "feature_set",
        str,
        "faithful",
        choices=("faithful", "position_only"),
        help="'faithful' (velocity-bearing, 27 feats) or 'position_only' (velocity dropped, 26 feats) "
        "for a model that scores on velocity-less SB360 freeze frames.",
    ),
)

#: What determines an extract shard's CONTENT (the legacy `_extract` token inputs, unchanged): the
#: extractor, the label horizon, the domain filter and the carrier params the features are built against.
#: `feature_set` changes the X columns (27 vs 26), so it MUST key the generation (the 4.77.1 trap).


def _feature_names(feature_set: str) -> list[str]:
    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL, XSHOT_FEATURE_NAMES_POSITION_ONLY

    return list(XSHOT_FEATURE_NAMES_POSITION_ONLY if feature_set == "position_only" else XSHOT_FEATURE_NAMES_FAITHFUL)


def _cv_n_splits(groups) -> int:
    """`_cv_metrics_plan`'s fold count for an arm's rows (the legacy ``max(2, min(5, n_games))``)."""
    return max(2, min(5, len(np.unique(groups))))


def _prep_work(key, ctx):
    """The study inputs (legacy main `:844-906`): X/y/groups/providers/match_ids, the arm masks from the
    ``--arms-json`` parameter, ``run_paired``, the D21 objective inputs -- persisted ONCE as a FileSet."""
    import tempfile

    from scripts._stages import FileSet
    from scripts._study_shared import persist_study_inputs
    from scripts.hf_jobs.arms import match_key, parse_arms

    p = ctx.params
    combined = ctx.upstream("extract")
    X = combined[_feature_names(p["feature_set"])]
    y = combined["_y"].to_numpy(int)
    # game_id dtype is provider-asymmetric (kloppy str vs GS int) -> normalize cross-provider groups to
    # str so np.unique / StratifiedGroupKFold can sort them (the model never uses game_id).
    groups = np.asarray(combined["_group"].to_numpy()).astype(str)
    providers = combined["_provider"].to_numpy()
    match_ids = combined["_match_id"].to_numpy()
    armspec = parse_arms(p["arms_json"])
    match_keys = np.array([match_key(str(pv), str(m)) for pv, m in zip(providers, match_ids, strict=True)])
    arms = _build_candidate_masks(match_keys, armspec)
    is_public = arms.get("public", np.zeros(len(match_keys), dtype=bool))
    run_paired = ("sc_extended" in arms) and ("full" in arms)
    objective_inputs = {
        "driver": "train_xshot_occurrence",
        "args": {k: v for k, v in p.items() if k not in {"dataset", "n_trials"}},
        "match_ids": sorted(map(str, match_ids.tolist())),
    }
    if run_paired:
        _, _, folds = _public_folds(X, y, groups, is_public)
        study_tags = [f"{name}_f{fold}" for fold in range(len(folds)) for name in sorted(arms)] + sorted(arms)
        cv_arms = {name: _cv_n_splits(groups[mask]) for name, mask in sorted(arms.items())}
    else:
        study_tags = ["single"]
        cv_arms = {"single": _cv_n_splits(groups)}
    config = {
        "n_trials": p["n_trials"],
        "negative_subsample": p["negative_subsample"],
        "seed": p["seed"],
        "feature_set": p["feature_set"],
        "horizon_seconds": p["horizon_seconds"],
        "run_paired": run_paired,
        # S5.10 commit-free: a stage reads no ctx.prov; adopt.stamp_provenance stamps the commit at adopt.
        "run_prov": {"commit": None, "dirty": None, "tree_state": None},
        "expect_variant": None,  # ADR-123 Amendment B: --expect-variant removed (pining-path preflight)
        "ship_order": list(armspec.ship_order),
        "objective_inputs": objective_inputs,
        "study_tags": study_tags,
        "cv_arms": cv_arms,
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "inputs"
        persist_study_inputs(
            root,
            X=X,
            y=y,
            groups=groups,
            providers=providers,
            match_ids=match_ids,
            is_public=is_public,
            config=config,
            arms=arms,
        )
        files = {q.relative_to(root).as_posix(): q.read_bytes() for q in sorted(root.rglob("*")) if q.is_file()}
    return FileSet(files)


def _prep_reduce(results, ctx) -> dict:
    """The stage output the downstream unit enumerations read: the persisted config (incl. study_tags,
    cv_arms) -- small JSON, so `--plan` never fetches the design matrix."""
    return json.loads(results["only"].files["study_config.json"])


_INPUTS: dict = {}


def _inputs(ctx):
    """The study inputs, materialised from the prep FileSet ONCE per worker (keyed by the prep token) into a
    worker-local dir that outlives the unit, so InnerPool children can memory-map the same files."""
    import atexit
    import shutil
    import tempfile

    from scripts._stages import materialize_fileset
    from scripts._study_shared import load_study_inputs

    tok = ctx.tokens["prep"]
    if tok not in _INPUTS:
        _INPUTS.clear()
        base = tempfile.mkdtemp(prefix="xshot_inputs_")
        atexit.register(shutil.rmtree, base, ignore_errors=True)
        root = Path(base) / "inputs"
        materialize_fileset(ctx, "prep", "only", root)
        _INPUTS[tok] = (root, load_study_inputs(root))
    return _INPUTS[tok]


def _study_rows(inp, tag: str) -> np.ndarray:
    """The row positions a study ``tag`` tunes on (the legacy `run_one_study` / main row logic):
    ``{arm}_f{k}`` -> the arm's rows minus fold k's public games; an arm name -> the arm's rows;
    ``single`` -> every row."""
    if tag == "single":
        return np.arange(len(inp.y))
    name, sep, fold_s = tag.rpartition("_f")
    if sep and fold_s.isdigit() and name in inp.arms:
        _, _, folds = _public_folds(inp.X, inp.y, inp.groups, inp.is_public)
        return np.flatnonzero(inp.arms[name] & folds[int(fold_s)][1])
    return np.flatnonzero(inp.arms[tag])


#: One entry each, per spawned InnerPool child (a child serves one study): the row slice and the plan.
_FOLD_DATA: dict = {}
_FOLD_PLAN: dict = {}


def _study_fold_task(corpus_dir, rows_path, plan_path, fold_i, params, ns, seed):
    """One CV fold of one HPO trial, in an InnerPool child (module-level: pickled by reference).

    Re-derives exactly the rows `_hpo_once` hands its objective (``X.iloc[rows]``, ``y[rows]``) from the
    memory-mapped study inputs, then evaluates the SAME fold the serial objective would."""
    from scripts._corpus_mmap import load_design_matrix
    from silly_kicks.tracking._xshot_occurrence_objective import eval_fold

    dkey = (str(corpus_dir), str(rows_path))
    if dkey not in _FOLD_DATA:
        X, y, _groups = load_design_matrix(Path(corpus_dir))
        rows = np.load(rows_path)
        _FOLD_DATA.clear()
        _FOLD_DATA[dkey] = (X.iloc[rows].reset_index(drop=True), np.asarray(y)[rows].astype(int))
    if str(plan_path) not in _FOLD_PLAN:
        z = np.load(plan_path)  # positional: arr_{2i} = fold i's train rows, arr_{2i+1} = its test rows
        _FOLD_PLAN.clear()
        _FOLD_PLAN[str(plan_path)] = {i: (z[f"arr_{2 * i}"], z[f"arr_{2 * i + 1}"]) for i in range(len(z.files) // 2)}
    Xr, yr = _FOLD_DATA[dkey]
    tr, te = _FOLD_PLAN[str(plan_path)][fold_i]
    return eval_fold(Xr, yr, fold_i, tr, te, params, negative_subsample=ns, subsample_seed=seed)


def _study_work(tag, ctx) -> dict:
    """ONE nested-HPO study (legacy `_hpo_once` on the tag's rows). At ``--inner-workers`` > 1 each trial's
    CV folds run on the InnerPool (ruling 4: the same seeded TPE, the same numbers)."""
    import tempfile

    from scripts._stages import int_key
    from silly_kicks.tracking._xshot_occurrence_objective import cv_fold_plan

    root, inp = _inputs(ctx)
    cfg = inp.config
    rows = _study_rows(inp, tag)
    X, y, groups = inp.X.iloc[rows], inp.y[rows], inp.groups[rows]
    ns, seed = cfg["negative_subsample"], cfg["seed"]
    # ignore_cleanup_errors: the HPO store's sqlite engine still holds the db open when the unit returns,
    # and Windows refuses to delete an open file (Linux unlinks it). The store is unit-local scratch.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        work = Path(tmp)
        fold_map: Callable[[dict], dict] | None = None
        pool = ctx.pool
        if pool is not None and pool.workers > 1:
            plan = cv_fold_plan(np.asarray(y).astype(int), groups)
            # The tag is in the file names: a child caches by path, so a reused temp-dir name can never
            # serve another study's rows.
            corpus_dir, rows_path, plan_path = root / "corpus", work / f"rows_{tag}.npy", work / f"plan_{tag}.npz"
            np.save(rows_path, rows)
            np.savez(plan_path, *[a for _i, tr, te in plan for a in (tr, te)])

            def _fold_map(params: dict) -> dict:
                items = {int_key(i): (corpus_dir, rows_path, plan_path, i, params, ns, seed) for i, _, _ in plan}
                return pool.map_keyed(_study_fold_task, items)

            fold_map = _fold_map

        store_dir = Path(ctx.store.local_path(ctx.generation("study")) or work) if STUDY_RESUME else work
        params = _hpo_once(
            X,
            y,
            groups,
            store_dir,
            tag,
            cfg["n_trials"],
            objective_inputs=cfg["objective_inputs"],
            prov=cfg["run_prov"],
            negative_subsample=ns,
            seed=seed,
            fold_map=fold_map,
        )
    return {"params": params, "n_trials": cfg["n_trials"]}


def _cv_units(ctx) -> list[str]:
    from scripts._stages import int_key, join_key

    arms = ctx.upstream("prep")["cv_arms"]
    return [join_key((arm, int_key(k))) for arm in sorted(arms) for k in range(int(arms[arm]))]


def _arm_params(ctx, arm: str) -> dict:
    """An arm's FINAL params: its full-rows study (the arm name), or ``single`` on the non-paired path."""
    return ctx.upstream("study")[arm]["params"]


def _arm_rows(inp, arm: str) -> np.ndarray:
    return np.arange(len(inp.y)) if arm == "single" else np.flatnonzero(inp.arms[arm])


def _cv_work(key, ctx) -> dict:
    """One fold of `_cv_metrics` for one arm, at that arm's final params (+ the arm's TRUE base rate, so
    the reduce never re-loads the design matrix)."""
    _root, inp = _inputs(ctx)
    arm, _, fold_s = key.rpartition("__")
    rows = _arm_rows(inp, arm)
    X, y, groups = inp.X.iloc[rows], inp.y[rows], inp.groups[rows]
    fold_i = int(fold_s)
    _i, tr, te = _cv_metrics_plan(y, groups)[fold_i]
    cfg = inp.config
    res = _cv_metrics_fold(X, y, fold_i, tr, te, _arm_params(ctx, arm), cfg["negative_subsample"], cfg["seed"])
    return {"fold": res, "positive_rate": float(np.asarray(y).mean())}


def _cv_reduce(results, ctx) -> dict:
    """``{arm: _cv_metrics dict}``, each arm's folds combined in FOLD order (sorted fold keys)."""
    by_arm: dict[str, list] = {}
    for key in sorted(results):
        arm, _, _fold = key.rpartition("__")
        by_arm.setdefault(arm, []).append(results[key])
    return {
        arm: _cv_metrics_combine([r["fold"] for r in folds], folds[0]["positive_rate"])
        for arm, folds in sorted(by_arm.items())
    }


def _select_work(key, ctx) -> dict:
    """The ship decision (legacy `:551-629`): the paired nested deltas at the cached study params
    (`params_for`), or the BROADEST arm of a non-paired corpus (``ship_order[-1]``)."""
    from scripts._corpus_identity import canonical_providers, check_shipped_variant
    from scripts._paired import fixed_sequence_ship

    _root, inp = _inputs(ctx)
    cfg = inp.config
    studies = ctx.upstream("study")
    cv = ctx.upstream("cv")
    candidates: dict = {}
    if cfg["run_paired"]:
        cand_masks = inp.arms
        paired = _paired_data_effect(
            inp.X,
            inp.y,
            inp.groups,
            inp.is_public,
            inp.match_ids,
            candidates=cand_masks,
            n_trials=cfg["n_trials"],
            out_dir=None,
            objective_inputs=cfg["objective_inputs"],
            prov=cfg["run_prov"],
            negative_subsample=cfg["negative_subsample"],
            seed=cfg["seed"],
            params_for=lambda tag: studies[tag]["params"],
        )
        full_vs_sc = [f - s for f, s in zip(paired["full"]["nested"], paired["sc_extended"]["nested"], strict=True)]
        shipped, why = fixed_sequence_ship(
            sc_extended=paired["sc_extended"]["nested"], full=paired["full"]["nested"], full_vs_sc=full_vs_sc
        )
        print(f"Fixed-sequence verdict: ship {shipped} -- {why}")
        check_shipped_variant(cfg.get("expect_variant"), shipped)
        ship_mask = cand_masks[shipped]
        candidates[shipped] = {
            "params": studies[shipped]["params"],
            "metrics": cv[shipped],
            "providers": canonical_providers(inp.providers[ship_mask]),
        }
        candidates["paired"] = {
            "nested": {n: paired[n]["nested"] for n in cand_masks},
            "shared_params": {n: paired[n]["shared_params"] for n in cand_masks},
            "full_vs_sc": full_vs_sc,
            "shipped": shipped,
            "why": why,
        }
    else:
        # ADR-123 Amendment B: the shipped label is an arm NAME from the --arms parameter. Ship the BROADEST
        # arm present (ship_order is narrow->broad): all-public -> public over every row; an owner-tier
        # 2-tier corpus ships sc_extended, NEVER the narrow public arm (the ADR-038 artifact-label control).
        shipped = cfg["ship_order"][-1]
        why = "single-candidate corpus: the broadest arm ships"
        check_shipped_variant(cfg.get("expect_variant"), shipped)
        candidates[shipped] = {
            "params": studies["single"]["params"],
            "metrics": cv["single"],
            "providers": canonical_providers(inp.providers),
        }
    return {"shipped": shipped, "why": why, "candidates": candidates}


def _fit_work(key, ctx):
    """The fail-closed acceptance gates, then the final fit at the shipped params, save and reload check
    (legacy `:631-658`). A failed gate writes the legacy ``metrics_FAILED.json`` payload beside the unit and
    raises, so the unit FAILS and no artifact exists."""
    import tempfile

    from scripts._stages import FileSet
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel, subsample_negatives

    _root, inp = _inputs(ctx)
    cfg = inp.config
    sel = ctx.upstream("select")
    shipped, candidates = sel["shipped"], sel["candidates"]
    acceptance = _gates(candidates[shipped]["metrics"])
    print(f"Shipped variant: {shipped}; gates: {acceptance}")
    if not all(acceptance.values()):
        failed = {"candidates": candidates, "acceptance": acceptance, "shipped_variant": shipped}
        ctx.store.write_bytes(
            f"{ctx.generation('fit')}/only.metrics_FAILED.json", json.dumps(failed, indent=2).encode("utf-8")
        )
        print("ACCEPTANCE GATES FAILED -- refusing to write the bundled artifact.", file=sys.stderr)
        raise AcceptanceGatesFailedError(shipped)
    ship_mask = inp.arms[shipped]
    X, y = inp.X, inp.y
    ns, seed = cfg["negative_subsample"], cfg["seed"]
    Xfit, yfit, _ = (
        subsample_negatives(X[ship_mask], y[ship_mask], y[ship_mask], fraction=ns, seed=seed)
        if ns
        else (X[ship_mask], y[ship_mask], None)
    )
    model = XShotOccurrenceModel(params=candidates[shipped]["params"], feature_set=cfg["feature_set"])
    model.shipped_variant = shipped
    model.provider_list = candidates[shipped]["providers"]
    model.training_commit = cfg["run_prov"]["commit"]
    model.fit(Xfit, pd.Series(yfit), carrier_params=DEFAULT_CARRIER_PARAMS, horizon_seconds=cfg["horizon_seconds"])
    with tempfile.TemporaryDirectory() as tmp:
        art = Path(tmp) / "xshot_occurrence_v1"
        model.save(art)
        reloaded = XShotOccurrenceModel.load(art)
        np.testing.assert_allclose(
            model.predict_proba(X[ship_mask].head(50)), reloaded.predict_proba(X[ship_mask].head(50)), rtol=0, atol=0
        )
        files = {q.name: q.read_bytes() for q in sorted(art.iterdir()) if q.is_file()}
    return FileSet(files)


def _assemble_work(key, ctx):
    """The legacy metrics block (`:659-678`) + the bundle FileSet."""
    from scripts._corpus_identity import canonical_providers, corpus_identity, reproducibility
    from scripts._stages import FileSet

    _root, inp = _inputs(ctx)
    cfg = inp.config
    sel = ctx.upstream("select")
    shipped, candidates = sel["shipped"], sel["candidates"]
    fit = ctx.upstream("fit")
    run_prov = cfg["run_prov"]
    metrics = {
        "run_commit": run_prov["commit"],
        "run_tree_dirty": run_prov["dirty"],
        "run_tree_state": run_prov["tree_state"],
        "shipped_variant": shipped,
        "n_rows": len(inp.X),
        "n_positive": int(np.asarray(inp.y).sum()),
        "providers": canonical_providers(inp.providers),
        # Corpus IDENTITY (spec section 5): exact ids only for an all-public corpus, else a digest
        # (Hub publishes copy metrics.json). n_rows alone could not tell 17 public matches from 27.
        **corpus_identity(inp.providers.tolist(), inp.match_ids.tolist(), all_public=bool(inp.is_public.all())),
        # ADR-067 M4 caveat, emitted here -- never hand-added at bundling (spec 0.11).
        **reproducibility(shipped, candidates[shipped]["providers"], training_commit=run_prov["commit"]),
        "candidates": candidates,
        "acceptance": _gates(candidates[shipped]["metrics"]),
        "estimates_are_cv_not_shipped_fit": True,
        "artifact_size_bytes": sum(len(b) for b in fit.files.values()),
    }
    files = dict(fit.files)
    files["metrics.json"] = json.dumps(metrics, indent=2).encode("utf-8")  # the legacy json.dump bytes
    print(f"Wrote artifact + metrics ({sorted(files)})")
    from scripts.hf_jobs.identity import commit_free

    return FileSet(commit_free(files))  # S5.10: commit-free stage output


PIPELINE = Pipeline(
    name="xshot_occurrence",
    module="scripts.train_xshot_occurrence",
    params=_PARAMS,
    stages=(
        _REUSABLE["extract"],  # ADR-129: declared in scripts._xshot_stages
        Stage(
            name="prep",
            units=lambda ctx: ["only"],
            work=_prep_work,
            needs=("extract",),
            reduce=_prep_reduce,
            params=("arms_json", "n_trials", "negative_subsample", "seed", "feature_set", "horizon_seconds"),
        ),
        Stage(
            name="study",
            units=lambda ctx: list(ctx.upstream("prep")["study_tags"]),
            work=_study_work,
            needs=("prep",),
            reduce=lambda r, ctx: dict(r),
            inner_parallel=True,
            inner_width=5,
        ),
        Stage(name="cv", units=_cv_units, work=_cv_work, needs=("prep", "study"), reduce=_cv_reduce),
        Stage(
            name="select",
            units=lambda ctx: ["only"],
            work=_select_work,
            needs=("prep", "study", "cv"),
            reduce=lambda r, ctx: r["only"],
        ),
        Stage(
            name="fit",
            units=lambda ctx: ["only"],
            work=_fit_work,
            needs=("prep", "select"),
            reduce=lambda r, ctx: r["only"],
        ),
        Stage(
            name="assemble",
            units=lambda ctx: ["only"],
            work=_assemble_work,
            needs=("prep", "select", "fit"),
            reduce=lambda r, ctx: r["only"],
        ),
    ),
    output="assemble",
)

if __name__ == "__main__":
    raise SystemExit(main(PIPELINE))
