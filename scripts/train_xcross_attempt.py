#!/usr/bin/env python
"""Train the xCrossAttempt (xCross) model (TF-17 weights run, PR-B).

Usage (ADR-128 stage contract; the uniform CLI from ``scripts._stages.main``)::

    python -m scripts.train_xcross_attempt --dataset tc3 --probe-dataset xcross_probe \
        --arms-json '<launcher payload>' --store <DIR> --data-root <DATA> --run-all --out <BUNDLE_DIR>

Pipeline: ``extract[match]`` (an ``ItemStage`` over the staged flat dataset: frames + the ``_actions`` SPADL
actions; ``home_teams.json`` signs score_differential) -> ``probe[match]`` (the held-out TF-19 cohort from
the staged per-game probe dataset) -> ``probe_comparison`` -> ``prep`` -> ``study[tag]`` -> ``cv[arm, fold]``
-> ``select`` -> ``fit`` -> ``assemble`` (the bundle + the GK validations).

On a public/owner mix with Gradient Sports it runs the common-public-held-out PAIRED data-effect comparison
over THREE candidates (public / sc_extended / full) with NESTED HPO -- each candidate re-tuned per outer
fold with that fold's public games excluded (spec 4.1, reviewer M4) -- then selects the shipped corpus via
the registered fixed sequence (scripts/_paired.py). Computes FAIL-CLOSED acceptance gates, and writes a
pickle-free artifact ONLY if the gates pass. Quality numbers in metrics.json are CV/protocol estimates, not
the shipped all-data fit.

Mirror of scripts/train_xshot_occurrence.py. Requires: silly-kicks[train,xgboost].
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `from scripts._X` resolves

from scripts._stages import FileSet, ItemStage, Param, Pipeline, Stage, join_key, main

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]


def _source_key(item):
    """The `for_each` key: a `MatchRef` -> its `.key`; a per-game item or `_PergameRef` ->
    `(provider, match_id)`. Module-level so the key-pin gate can see it (_KEY_EXCEPTIONS)."""
    key = getattr(item, "key", None)
    return key if key is not None else (str(item[0]), str(item[1]))


def _new_probe_cohort() -> dict:
    """One TF-19 probe cohort's capture state (M5): bounded frames/actions copies + provenance."""
    return {"frames": [], "actions": [], "home": None, "matches": [], "match_groups": {}}


#: The four per-row arrays `_extract` returns alongside the feature matrix, carried as COLUMNS so
#: one match is one tidy shard. Underscore-prefixed and collision-checked: a feature named `_y`
#: would be silently overwritten, and the model would train on its own label.
_SIDE_COLS = ("_y", "_group", "_provider", "_match_id")


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
    fold_map=None,
) -> dict:
    """Run ruthless HPO once for one candidate; return the frozen best-params dict.

    ``fold_map`` (ADR-128 ruling 4) evaluates each trial's CV folds in parallel (an InnerPool); the
    objective combines them in sorted fold order, so the study is numerically unchanged.

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
    params_for=None,
) -> dict:
    """Nested-HPO paired comparison on the common public held-out folds (spec 4.1, reviewer M4).

    ``params_for(tag)`` (ADR-128) serves each ``{name}_f{fold}`` study's frozen params from the ``study``
    stage instead of running ``_hpo_once`` here; the studies are deterministic, so the deltas are identical.

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
    """The shipped candidate failed a fail-closed acceptance gate. main() maps this to exit 1."""


def _build_candidate_masks(match_keys, armspec) -> dict:
    """The candidate row masks, built PURELY from the --arms parameter (ADR-123 Amendment B).

    An arm is a set of dataset match-keys supplied by the launcher; the mask is set membership. NO
    provider-name and NO public/private logic in-job. ``main`` builds + persists the masks; the fan-out
    workers + the reduce read ``StudyInputs.arms``.
    """
    from scripts.hf_jobs.arms import arm_masks

    return arm_masks(match_keys, armspec)


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


# ---- ADR-128 stage contract --------------------------------------------------------------------------
#
# extract[match] -> probe[match] (the held-out TF-19 cohort, from the staged probe dataset) ->
# probe_comparison (one unit: the reported-not-gated comparison cohort) -> prep -> study[tag]
# (inner_parallel) -> cv[arm, fold] -> select -> fit -> assemble (the bundle + the TF-19 block).
# Mirror of scripts/train_xshot_occurrence.py; every stage single-sources the legacy computation.

#: Decision R1 (spec 5.3): measured on the xShot twin (tests/scripts/test_study_resume_r1.py) -- the two
#: trainers share `_hpo_once`'s store mechanics, so a resumed study does not reproduce the uninterrupted one
#: here either. The store lives in a unit-local temp dir.
STUDY_RESUME = False

#: The legacy capture budget per probe cohort (M5: bounded frames copies).
PROBE_KEEP = 2


def _csv(text: str) -> tuple[str, ...]:
    return tuple(p for p in str(text).split(",") if p)


_PARAMS = (
    Param("dataset", str, None, required=True, help="the staged flat dataset (shards/<gen>/ + _actions/)"),
    Param(
        "probe_dataset",
        str,
        None,
        required=True,
        help="the staged PER-GAME held-out TF-19 probe dataset ({game}/frames.parquet + actions.parquet); "
        "ADR-123's separate probe pass (decision P17: the in-line capture mode is gone)",
    ),
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
        help="Thin this fraction of negatives in TRAIN folds only (never eval). Default OFF "
        "(crosses have a healthy base rate -- PA-M4 -- so subsampling is usually unnecessary).",
    ),
    Param("seed", int, 42, help="Seed for --negative-subsample (deterministic)."),
    Param(
        "probe_providers",
        str,
        "gradientsports",
        help="Comma list of providers eligible for the TF-19 substitution-probe capture (M5: the GATED cohort).",
    ),
    Param(
        "probe_comparison_providers",
        str,
        "skillcorner",
        help="Comma list captured to the reported-not-gated same-population comparison cohort (M5).",
    ),
    Param(
        "feature_set",
        str,
        "faithful",
        choices=("faithful", "position_only"),
        help="'faithful' (velocity-bearing, 16 feats) or 'position_only' (velocity dropped, 15 feats) "
        "for a model that scores on velocity-less SB360 freeze frames.",
    ),
    Param(
        "ship_variant",
        str,
        None,
        choices=("public", "sc_extended", "full"),
        help="Operator override: force-ship THIS variant regardless of the fixed-sequence bundle-selection "
        "verdict (the paired test still runs and its verdict + deltas are recorded). Requires the "
        "multi-candidate (run_paired) corpus. Default None = normal gate-selected ship.",
    ),
)


def _extract_declare(p) -> dict:
    """What determines a feature shard's CONTENT (the legacy `_extract` token inputs, unchanged). The probe
    provider filters are NOT declared: they select which matches are copied into a cohort, not what a
    feature row contains."""
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS

    return {
        "extractor": "prepare_xcross_training_data",
        # feature_set changes the X columns (16 vs 15) -> MUST key the shard generation (4.77.1).
        "feature_set": p["feature_set"],
        "horizon_seconds": p["horizon_seconds"],
        "wide_area_only": True,
        "carrier_params": dict(DEFAULT_CARRIER_PARAMS),
    }


class _PergameRef(NamedTuple):
    """A staged unit seen as the legacy per-game item. ``(prov, mid)`` is the legacy `_source_key`: the
    shard's ``source_provider`` and the per-game directory name (training: the repackaged
    ``{provider}-{game}``; probe: the staged per-game directory), so unit keys and arm keys stay
    byte-identical to the 4.138.0 runs. ``home_map`` says whether the dataset's home map applies."""

    prov: str
    mid: str
    stem: str
    files: tuple[str, ...]
    dataset_param: str


def _extract_refs(ctx) -> list[_PergameRef]:
    from scripts._driver_sources import staged_flat_refs
    from scripts.hf_jobs.datasets import pergame_name

    refs = staged_flat_refs(ctx.data_root, ctx.params["dataset"])
    return [_PergameRef(r.source_provider, pergame_name(r.provider, r.game), r.stem, r.files, "dataset") for r in refs]


def _probe_refs(ctx) -> list[_PergameRef]:
    """The probe dataset's per-game units (`staged_pergame_refs` skips dot-directories such as an
    ``hf download`` ``.cache/`` -- the 4.138.0 crash)."""
    from scripts._driver_sources import staged_pergame_refs

    refs = staged_pergame_refs(ctx.data_root, ctx.params["probe_dataset"])
    return [_PergameRef(r.source_provider, r.stem, r.stem, r.files, "probe_dataset") for r in refs]


def _home_map(ctx) -> dict | None:
    from scripts._driver_sources import home_teams

    return home_teams(ctx.data_root, ctx.params["dataset"]) or None


def _load_match(ref: _PergameRef, ctx):
    """``(prov, mid, actions, frames, home)`` exactly as the legacy `_iter_matches_from_dir` built it.

    xcross USES home to sign score_differential (PA-H1), so the training dataset's ``home_teams.json``
    overrides the first-team heuristic (keyed by game_id; TF-65 C2). The probe pass never had a map."""
    from scripts._driver_sources import dataset_dir, read_dataset_parquet

    root = dataset_dir(ctx.data_root, ctx.params[ref.dataset_param])
    frames = read_dataset_parquet(root / ref.files[0])
    actions = read_dataset_parquet(root / ref.files[1])
    home = None
    home_map = _home_map(ctx) if ref.dataset_param == "dataset" else None
    if home_map is not None:
        gids = frames["game_id"].dropna()
        if len(gids):
            home = home_map.get(str(gids.iloc[0]))
    if home is None:
        home = frames["team_id"].dropna().iloc[0]
    return ref.prov, ref.mid, actions, frames, home


def _prepare(item, ctx):
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL, prepare_xcross_training_data

    collision = set(_SIDE_COLS) & set(XCROSS_FEATURE_NAMES_FAITHFUL)
    if collision:
        raise ValueError(f"side columns {sorted(collision)} collide with feature names")
    _prov, _mid, actions, frames, home = item
    return prepare_xcross_training_data(
        frames,
        actions,
        home_team_id=home,
        feature_set=ctx.params["feature_set"],
        horizon_seconds=ctx.params["horizon_seconds"],
        wide_area_only=True,
        carrier_params=DEFAULT_CARRIER_PARAMS,  # 4.7.0 values; shared constant (anti-drift)
    )


def _extract_work(item, ctx) -> pd.DataFrame | None:
    """One match -> its tidy feature rows + the four side columns (the legacy `_extract._work`), or None."""
    prov, mid, *_ = item
    X, y, groups = _prepare(item, ctx)
    if not len(X):
        return None  # "ran, produced no usable row"
    return X.assign(
        _y=np.asarray(y, int),
        _group=np.asarray(groups),
        _provider=str(prov),
        _match_id=str(mid),  # per-row match id (the arm key)
    )


def _extract_counters(item, _frame) -> dict:
    """The ADR-126 integrity inputs of one match (crosses by the label's own ``type_id`` set, frames,
    duration): counters only, never in a token or the artifact (ADR-128 plan Task 20)."""
    from scripts._integrity_counters import integrity_counters
    from silly_kicks.tracking._xcross_attempt import _DEFAULT_CROSS_TYPES

    _prov, _mid, actions, frames, _home = item
    return integrity_counters(frames, actions, _DEFAULT_CROSS_TYPES)


def _extract_reduce(results, ctx) -> pd.DataFrame:
    """The legacy combine: NON-EMPTY shards in the legacy iteration order (sorted per-game dir names)."""
    from scripts._stages import combine_frames

    parts = [f for f in results.values() if len(f)]
    if not parts:
        raise RuntimeError("No usable training data.")
    # `_group` mixes kloppy str and Gradient Sports int game ids: stored as the strings the study groups on
    return combine_frames(parts, ids=("_group",))


def _mid_of(key: str) -> str:
    """The per-game dir name of a joined `_source_key` unit key (``<prov>__<mid>``): the legacy order."""
    return key.partition("__")[2]


def _capture_record(prov, mid, groups) -> pd.DataFrame:
    """A cohort capture as ONE row (an ItemStage result): the legacy cohort's per-match provenance (the
    match's ``game_id`` groups, M6). Its frames are reloaded from the immutable staged dataset."""
    return pd.DataFrame(
        {
            "provider": [str(prov)],
            "match_id": [str(mid)],
            "groups": [json.dumps(sorted({str(g) for g in np.asarray(groups).tolist()}))],
        }
    )


def _probe_work(item, ctx) -> pd.DataFrame | None:
    """One probe match: its capture record when its provider is GATED (``--probe-providers``) and the
    extraction yields rows -- the legacy probe pass's capture rule -- else None."""
    prov, mid, *_rest = item
    if prov not in _csv(ctx.params["probe_providers"]):
        return None
    X, _y, groups = _prepare(item, ctx)
    if not len(X):
        return None
    return _capture_record(prov, mid, groups)


def _cohort_files(ctx, records: list[pd.DataFrame], refs: dict, provider_filter) -> FileSet:
    """The legacy `_write_probe_sample` payload for the FIRST `PROBE_KEEP` captures (sorted key order =
    the legacy iteration order), reloading each match's frames + actions from the immutable staged dataset,
    so a resumed stage rebuilds the identical cohort (the legacy guard `res.skipped` is 0 by construction)."""
    import tempfile

    cohort = _new_probe_cohort()
    for rec in records[:PROBE_KEEP]:
        row = rec.iloc[0]
        ref = refs[join_key((row["provider"], row["match_id"]))]
        _prov, _mid, actions, frames, home = _load_match(ref, ctx)
        cohort["frames"].append(frames.copy())
        cohort["actions"].append(actions.copy())
        cohort["home"] = home
        cohort["matches"].append([str(row["provider"]), str(row["match_id"])])
        cohort["match_groups"][str(row["match_id"])] = json.loads(row["groups"])
    with tempfile.TemporaryDirectory() as tmp:
        ps = Path(tmp) / "sample"
        _write_probe_sample(ps, cohort, list(provider_filter))
        files = {q.name: q.read_bytes() for q in sorted(ps.iterdir())} if ps.exists() else {}
    return FileSet(files)


def _probe_reduce(results, ctx) -> FileSet:
    st = ctx.pipeline.stage("probe")
    assert isinstance(st, ItemStage)  # noqa: S101 -- the pipeline declares it so
    records = [results[k] for k in results if len(results[k])]
    return _cohort_files(ctx, records, st.ref_map(ctx), _csv(ctx.params["probe_providers"]))


def _probe_comparison_work(key, ctx) -> FileSet:
    """The reported-not-gated comparison cohort (M5): the first `PROBE_KEEP` TRAINING matches of a
    ``--probe-comparison-providers`` provider whose extraction yielded rows, in the legacy iteration order
    (the legacy main pass captured it in-line). Read from the extract unit results, so nothing re-extracts."""
    from scripts._stages import read_result, sorted_units

    st = ctx.pipeline.stage("extract")
    assert isinstance(st, ItemStage)  # noqa: S101 -- the pipeline declares it so
    gen = ctx.generation("extract")
    providers = _csv(ctx.params["probe_comparison_providers"])
    refs = st.ref_map(ctx)
    records = []
    for k in sorted(sorted_units(ctx, st), key=_mid_of):
        ref = refs[k]
        if ref.prov not in providers or len(records) >= PROBE_KEEP:
            continue
        rows = read_result(ctx.store, f"{gen}/{k}")
        if not isinstance(rows, pd.DataFrame) or not len(rows):
            continue
        records.append(_capture_record(ref.prov, ref.mid, rows["_group"].to_numpy()))
    return _cohort_files(ctx, records, refs, providers)


def _feature_names(feature_set: str) -> list[str]:
    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL, XCROSS_FEATURE_NAMES_POSITION_ONLY

    return list(XCROSS_FEATURE_NAMES_POSITION_ONLY if feature_set == "position_only" else XCROSS_FEATURE_NAMES_FAITHFUL)


def _cv_n_splits(groups) -> int:
    """`_cv_metrics`'s fold count for an arm's rows (`_xcross_eval._cv_score`: ``max(2, min(5, n_games))``)."""
    return max(2, min(5, len(np.unique(np.asarray(groups).astype(str)))))


def _sd_range_probe(X: pd.DataFrame) -> dict:
    """The score_differential range probe (B6): guard the phantom-owngoal signature BEFORE any HPO."""
    sd = X["score_differential"].to_numpy(dtype=float)
    sd_fin = sd[np.isfinite(sd)]
    probe = {
        "coverage": float(np.isfinite(sd).mean()),
        "min": float(sd_fin.min()) if sd_fin.size else float("nan"),
        "max": float(sd_fin.max()) if sd_fin.size else float("nan"),
        "abs_ge_12_count": int((np.abs(sd_fin) >= 12).sum()),
    }
    if probe["abs_ge_12_count"] > 0:
        raise RuntimeError(
            f"score_differential range probe FAILED (impossible |sd|>=12): {probe}. "
            "Rebuild the feature cache on clean 4.13.0 GS events."
        )
    if sd_fin.size and np.abs(sd_fin).max() > 6:
        print(f"WARN score_differential |max|>6 (legit blowout possible): {probe}", file=sys.stderr)
    return probe


def _prep_work(key, ctx):
    """The study inputs (legacy main): X/y/groups/providers/match_ids, the arm masks from ``--arms-json``,
    ``run_paired``, the D21 objective inputs and the B6 range probe -- persisted ONCE as a FileSet."""
    import tempfile

    from scripts._study_shared import persist_study_inputs
    from scripts.hf_jobs.arms import match_key, parse_arms

    p = ctx.params
    if set(_csv(p["probe_providers"])) & set(_csv(p["probe_comparison_providers"])):
        raise ValueError("--probe-providers and --probe-comparison-providers must be disjoint.")
    combined = ctx.upstream("extract")
    X = combined[_feature_names(p["feature_set"])]
    y = combined["_y"].to_numpy(int)
    groups = np.asarray(combined["_group"].to_numpy()).astype(str)
    providers = combined["_provider"].to_numpy()
    match_ids = combined["_match_id"].to_numpy()
    sd_probe = _sd_range_probe(X)
    armspec = parse_arms(p["arms_json"])
    match_keys = np.array([match_key(str(pv), str(m)) for pv, m in zip(providers, match_ids, strict=True)])
    arms = _build_candidate_masks(match_keys, armspec)
    is_public = arms.get("public", np.zeros(len(match_keys), dtype=bool))
    run_paired = ("sc_extended" in arms) and ("full" in arms)
    if p["ship_variant"] is not None and not run_paired:
        raise ValueError(
            "--ship-variant requires the multi-candidate (run_paired) corpus (public + owner "
            "SkillCorner + gradientsports) so the variant masks and the TF-19 probe cohort exist."
        )
    objective_inputs = {
        "driver": "train_xcross_attempt",
        "args": {k: v for k, v in p.items() if k not in {"dataset", "probe_dataset", "n_trials"}},
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
        "run_prov": {k: ctx.prov[k] for k in ("commit", "dirty", "tree_state")},
        "objective_inputs": objective_inputs,
        "ship_variant": p["ship_variant"],
        "expect_variant": None,  # ADR-123 Amendment B: --expect-variant removed (pining-path preflight)
        "ship_order": list(armspec.ship_order),
        "score_differential_range_probe": sd_probe,
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
    """The persisted config (incl. study_tags, cv_arms): small JSON, so `--plan` never fetches the matrix."""
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
        base = tempfile.mkdtemp(prefix="xcross_inputs_")
        atexit.register(shutil.rmtree, base, ignore_errors=True)
        root = Path(base) / "inputs"
        materialize_fileset(ctx, "prep", "only", root)
        _INPUTS[tok] = (root, load_study_inputs(root))
    return _INPUTS[tok]


def _study_rows(inp, tag: str) -> np.ndarray:
    """The row positions a study ``tag`` tunes on (the legacy `run_one_study` / main row logic)."""
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
    """One CV fold of one HPO trial, in an InnerPool child (module-level: pickled by reference)."""
    from scripts._corpus_mmap import load_design_matrix
    from silly_kicks.tracking._xcross_attempt_objective import eval_fold

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
    """ONE nested-HPO study (legacy `_hpo_once` on the tag's rows); at ``--inner-workers`` > 1 each trial's
    CV folds run on the InnerPool (ruling 4: the same seeded TPE, the same numbers)."""
    import tempfile

    from scripts._stages import int_key
    from silly_kicks.tracking._xcross_attempt_objective import cv_fold_plan

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
    from scripts._stages import int_key

    arms = ctx.upstream("prep")["cv_arms"]
    return [join_key((arm, int_key(k))) for arm in sorted(arms) for k in range(int(arms[arm]))]


def _arm_rows(inp, arm: str) -> np.ndarray:
    return np.arange(len(inp.y)) if arm == "single" else np.flatnonzero(inp.arms[arm])


def _cv_work(key, ctx) -> dict:
    """One fold of `_cv_metrics` (`_xcross_eval._cv_score`) for one arm at the arm's final params, plus the
    arm's TRUE base rate (so the reduce never re-loads the design matrix)."""
    from silly_kicks.tracking import _xcross_eval as ev

    _root, inp = _inputs(ctx)
    arm, _, fold_s = key.rpartition("__")
    rows = _arm_rows(inp, arm)
    X, y, groups = inp.X.iloc[rows], inp.y[rows], inp.groups[rows]
    fold_i = int(fold_s)
    _i, tr, te = ev._cv_score_plan(y, groups)[fold_i]
    cfg = inp.config
    params = ctx.upstream("study")[arm]["params"]
    res = ev._cv_score_fold(
        X, y, fold_i, tr, te, params, seed=cfg["seed"], negative_subsample=cfg["negative_subsample"]
    )
    return {"fold": res, "positive_rate": float(np.asarray(y, dtype=int).mean())}


def _cv_reduce(results, ctx) -> dict:
    """``{arm: _cv_metrics dict}``, each arm's folds combined in FOLD order (sorted fold keys)."""
    from silly_kicks.tracking import _xcross_eval as ev

    by_arm: dict[str, list] = {}
    for key in sorted(results):
        arm, _, _fold = key.rpartition("__")
        by_arm.setdefault(arm, []).append(results[key])
    out = {}
    for arm, folds in sorted(by_arm.items()):
        base = folds[0]["positive_rate"]
        out[arm] = {
            **ev._cv_score_combine([r["fold"] for r in folds]),
            "positive_rate": base,
            "base_rate_brier": base * (1 - base),
        }
    return out


def _select_work(key, ctx) -> dict:
    """The ship decision: the paired nested deltas at the cached study params (`params_for`), the operator
    ``--ship-variant`` override, or the BROADEST arm of a non-paired corpus (``ship_order[-1]``)."""
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
        ship_variant = cfg.get("ship_variant")
        if ship_variant is not None:
            why = f"operator --ship-variant override (fixed-sequence gate verdict was: {shipped} -- {why})"
            shipped = ship_variant
            print(f"Ship-variant override: forcing ship {shipped}. {why}")
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
        # ADR-123 Amendment B: ship the BROADEST arm NAME present (ship_order is narrow->broad); an owner-tier
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
    """The fail-closed acceptance gates, the final fit at the shipped params, save and reload check. A
    failed gate writes the legacy ``metrics_FAILED.json`` payload beside the unit and raises."""
    import tempfile

    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
    from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel
    from silly_kicks.tracking._xshot_occurrence import subsample_negatives

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
    model = XCrossAttemptModel(params=candidates[shipped]["params"], feature_set=cfg["feature_set"])
    model.shipped_variant = shipped
    model.provider_list = candidates[shipped]["providers"]
    model.training_commit = cfg["run_prov"]["commit"]
    model.fit(Xfit, pd.Series(yfit), carrier_params=DEFAULT_CARRIER_PARAMS, horizon_seconds=cfg["horizon_seconds"])
    with tempfile.TemporaryDirectory() as tmp:
        art = Path(tmp) / "xcross_attempt_v1"
        model.save(art)
        reloaded = XCrossAttemptModel.load(art)
        np.testing.assert_allclose(
            model.predict_proba(X[ship_mask].head(50)), reloaded.predict_proba(X[ship_mask].head(50)), rtol=0, atol=0
        )
        files = {q.name: q.read_bytes() for q in sorted(art.iterdir()) if q.is_file()}
    return FileSet(files)


def _tf19_block(ctx, inp, model, shipped, candidates) -> dict:
    """The headline GK validations on the SHIPPED candidate (PR-B; legacy assemble `run_probe`): the GK-block
    ablation, the CV-held-out permutation importance and the TF-19 substitution probe on the held-out cohort."""
    import tempfile

    from scripts._stages import materialize_fileset
    from silly_kicks.tracking import _xcross_eval as ev

    cfg = inp.config
    ns, seed = cfg["negative_subsample"], cfg["seed"]
    X, y, groups = inp.X, inp.y, inp.groups
    ship_mask = inp.arms[shipped]
    shipped_params = candidates[shipped]["params"]
    gk_ablation = ev.gk_block_ablation(
        X[ship_mask], y[ship_mask], groups[ship_mask], shipped_params, seed=seed, negative_subsample=ns
    )
    perm_imp = ev.permutation_importance_report(
        X[ship_mask], y[ship_mask], groups[ship_mask], shipped_params, n_repeats=10, seed=seed
    )
    with tempfile.TemporaryDirectory() as tmp:
        ps = materialize_fileset(ctx, "probe", "_output", Path(tmp) / "_probe_sample")
        if not (ps / "frames.parquet").exists():
            raise RuntimeError(
                "the probe stage captured no cohort -> cannot run the TF-19 substitution probe (the headline "
                "deliverable). Check --probe-dataset / --probe-providers. Refusing to ship a spurious "
                "tf19_ready=False."
            )
        pf = pd.read_parquet(ps / "frames.parquet")
        pa = pd.read_parquet(ps / "actions.parquet")
        probe_meta = json.loads((ps / "meta.json").read_text(encoding="utf-8"))
    run_paired = cfg["run_paired"]
    phome = probe_meta["home_team_id"]
    probe_matches = probe_meta.get("probe_matches", [])
    match_groups = probe_meta.get("match_groups", {})
    train_groups = set(groups[ship_mask].tolist())
    in_training = {mid: bool(set(g) & train_groups) for mid, g in match_groups.items()}
    probe_meta["in_training_folds"] = in_training
    shipped_providers = set(candidates[shipped].get("providers", []))
    probe_provs_seen = {prov for prov, _mid in probe_matches}
    admitted = bool(run_paired and (not probe_provs_seen or probe_provs_seen & shipped_providers))
    try:
        gated_matches = _gated_probe_matches(probe_meta, admitted)
    except SystemExit as exc:  # the legacy refusal; a stage unit fails on an Exception, never exits the worker
        raise RuntimeError(str(exc)) from None
    if admitted:
        held_groups = {g for _prov, mid in gated_matches for g in match_groups.get(str(mid), [])}
        pf = pf[pf["game_id"].astype(str).isin(held_groups)]
        pa = pa[pa["game_id"].astype(str).isin(held_groups)]
        if pf.empty:
            raise RuntimeError(
                "Held-out probe matches resolved ZERO frames -- the probe sample is inconsistent with its "
                "meta.json match_groups. Re-run the probe stage."
            )
    elif probe_matches and all(in_training.get(str(m[1]), False) for m in probe_matches):
        print(
            "NOTE: every probe match sits in the shipped training corpus (single-candidate path) -- "
            "the substitution-probe statistic below is NOT held-out.",
            file=sys.stderr,
        )
    probe = ev.gk_substitution_probe(model, pf, actions=pa, home_team_id=phome, n_frames=200, seed=seed)
    if not probe.get("tf19_ready", False):
        print(
            f"NOTE: tf19_ready=False ({probe.get('tf19_reason')}) -- surface ships, but flagged "
            "NOT TF-19-ready (loud, not silent).",
            file=sys.stderr,
        )
    return {
        "gk_block_ablation": gk_ablation,
        "gk_substitution_probe": probe,
        "permutation_importance": perm_imp,
        "score_differential_range_probe": cfg["score_differential_range_probe"],
        "probe_sample_matches": probe_matches,
        "probe_sample_in_training_folds": in_training,
        "probe_gated_on_held_out": bool(admitted),
        "tf19_ready": probe.get("tf19_ready", False),
    }


def _assemble_work(key, ctx):
    """The legacy metrics block + the TF-19 block, and the bundle FileSet."""
    import tempfile

    from scripts._corpus_identity import canonical_providers, corpus_identity, reproducibility
    from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel

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
        # Corpus IDENTITY (spec section 5): exact ids only for an all-public corpus, else a digest.
        **corpus_identity(inp.providers.tolist(), inp.match_ids.tolist(), all_public=bool(inp.is_public.all())),
        # ADR-067 M4 caveat, emitted here -- never hand-added at bundling (spec 0.11).
        **reproducibility(shipped, candidates[shipped]["providers"], training_commit=run_prov["commit"]),
        "candidates": candidates,
        "acceptance": _gates(candidates[shipped]["metrics"]),
        "estimates_are_cv_not_shipped_fit": True,
        "artifact_size_bytes": sum(len(b) for b in fit.files.values()),
    }
    with tempfile.TemporaryDirectory() as tmp:
        art = Path(tmp) / "xcross_attempt_v1"
        art.mkdir()
        for name, data in fit.files.items():
            (art / name).write_bytes(data)
        model = XCrossAttemptModel.load(art)  # the shipped artifact itself: bit-identical predictions
    metrics.update(_tf19_block(ctx, inp, model, shipped, candidates))
    files = dict(fit.files)
    files["metrics.json"] = json.dumps(metrics, indent=2).encode("utf-8")  # the legacy json.dump bytes
    print(f"Wrote artifact + metrics ({sorted(files)})")
    return FileSet(files)


PIPELINE = Pipeline(
    name="xcross_attempt",
    module="scripts.train_xcross_attempt",
    params=_PARAMS,
    stages=(
        ItemStage(
            name="extract",
            refs=_extract_refs,
            key=_source_key,
            load=_load_match,
            work=_extract_work,
            counters=_extract_counters,
            reduce=_extract_reduce,
            reduce_order=_mid_of,
            declare=_extract_declare,
            cross_commit_version="xcross-extract-1",
            uses_dataset=True,
            max_failures=0,  # the legacy raised on any failed match
        ),
        ItemStage(
            name="probe",
            refs=_probe_refs,
            key=_source_key,
            load=_load_match,
            work=_probe_work,
            reduce=_probe_reduce,
            reduce_order=_mid_of,
            params=("probe_providers",),
            declare=_extract_declare,
            cross_commit_version="xcross-probe-1",
            uses_dataset=True,
            dataset_param="probe_dataset",
            max_failures=0,
        ),
        Stage(
            name="probe_comparison",
            units=lambda ctx: ["only"],
            work=_probe_comparison_work,
            needs=("extract",),
            reduce=lambda r, ctx: r["only"],
            params=("probe_comparison_providers",),
        ),
        Stage(
            name="prep",
            units=lambda ctx: ["only"],
            work=_prep_work,
            needs=("extract",),
            reduce=_prep_reduce,
            params=(
                "arms_json",
                "n_trials",
                "negative_subsample",
                "seed",
                "feature_set",
                "horizon_seconds",
                "probe_providers",
                "probe_comparison_providers",
                "ship_variant",
            ),
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
            needs=("prep", "select", "fit", "probe", "probe_comparison"),
            reduce=lambda r, ctx: r["only"],
        ),
    ),
    output="assemble",
)

if __name__ == "__main__":
    raise SystemExit(main(PIPELINE))
