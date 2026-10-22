#!/usr/bin/env python
"""Train the Ghost-GK positioning model (TF-18).

Usage (ADR-128 stage contract; the uniform CLI from ``scripts._stages.main``)::

    python -m scripts.train_ghost_gk --dataset tc3 --variant default \
        --store <DIR> --data-root <DATA> --run-all --out <BUNDLE_DIR>

Pipeline: ``validate`` (provider + visibility + schema fail-fast, before any extraction) ->
``extract[match]`` (an ``ItemStage`` over the staged flat dataset: ``shards/<gen>/`` frames, ``_actions/``
and ``home_teams.json``) -> ``cache`` (the whole-corpus matrix) -> ``cv[fold]`` -> ``fit`` (final fit +
the BLOCKING parity gate) -> ``importance`` (metrics-only; one ``inner_parallel`` unit) -> ``assemble``
(the bundle). The tc3 / ``--data-dir`` / ``--actions-dir`` / ``--home-teams`` modes are gone (decision
P17): a corpus is staged first (``scripts/hf_jobs/datasets.toml``).

Requires: silly-kicks[train] (scikit-learn in [1.9, 2) for a shippable fit).

See docs/superpowers/specs/2026-05-26-tf18-training-hub-publish-design.md.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `from scripts._X` resolves

from scripts._corpus_visibility import validate_corpus_visibility  # noqa: F401 -- re-export: tests
from scripts._ghost_gk_stages import (
    _LABEL_PREFIX,
    _SIDE_COLS,
    _assert_no_column_collision,  # noqa: F401 -- re-export: tests
    _carrier,
    _pack,  # noqa: F401 -- re-export: tests
    _root,
    build_extraction_inputs,  # noqa: F401 -- re-export: tests
    cache_token,  # noqa: F401 -- re-export: tests
    grid_spec_from_args,
    validate_corpus_providers,  # noqa: F401 -- re-export: tests
)
from scripts._ghost_gk_stages import (  # ADR-129: the reusable stages live in their own module
    STAGES as _REUSABLE,
)
from scripts._provenance import resolve_training_platform  # noqa: F401 -- re-export (moved to _provenance)
from scripts._stages import Param, Pipeline, Stage, int_key, main
from scripts.hf_jobs.identity import commit_free


def _unpack(combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Invert :func:`_pack` over the combined table.

    The feature frame is reconstructed by REMOVAL, never by selecting ``GHOST_GK_FEATURE_NAMES``:
    the pre-migration code concatenated whatever ``prepare_ghost_gk_training_data`` returned, so a
    positive selection would silently change what the model is fitted on the day the extractor
    widens -- a different model from the same code, with nothing saying so.

    The three arrays are rebuilt via ``np.array(<list>)`` rather than ``Series.to_numpy()`` so their
    dtypes match the pre-migration ones exactly (``np.array(["a"])`` is a ``<U1`` array, not object);
    ``groups`` and ``provider_labels`` reach ``StratifiedGroupKFold`` and ``keepers`` reaches the
    keeper-domain code, none of which should see a dtype change from a storage decision.
    """
    lab_cols = [c for c in combined.columns if str(c).startswith(_LABEL_PREFIX)]
    if not lab_cols:
        raise ValueError(f"no {_LABEL_PREFIX!r} columns in the combined shards -- labels were lost in packing")
    features = combined.drop(columns=[*lab_cols, *_SIDE_COLS]).reset_index(drop=True)
    labels = combined[lab_cols].rename(columns=lambda c: c[len(_LABEL_PREFIX) :]).reset_index(drop=True)
    groups = np.array(combined["_game_id"].tolist())
    provider_labels = np.array(combined["_provider"].astype(str).tolist())
    keepers = np.array(combined["_keeper"].astype(str).tolist(), dtype=object)
    return features, labels, groups, provider_labels, keepers


def _mean_from_counters(total: float, n: float) -> float:
    """``sum / n``, or NaN on an empty population -- matching what ``np.mean([])`` reports."""
    return float(total) / float(n) if n else float("nan")


def high_sweeper_stratum_mae(preds, labels, *, threshold: float = 30.0) -> float:
    """Euclidean MAE over the HIGH-SWEEPER stratum only (labels with ``gk_x > threshold``).

    TF-60 PR3 sec 7.3: the sweeper re-fit must be validated to predict the >30 m regime, not merely
    lower the overall error. NaN on an empty stratum (nothing to average) -- reported, not gated.
    """
    gx = np.asarray(labels["gk_x"], dtype=float)
    gy = np.asarray(labels["gk_y"], dtype=float)
    mask = gx > threshold
    if not mask.any():
        return float("nan")
    dx = preds[mask, 0] - gx[mask]
    dy = preds[mask, 1] - gy[mask]
    return float(np.mean(np.sqrt(dx * dx + dy * dy)))


def per_provider_high_sweeper_coverage(labels, provider_labels, *, threshold: float = 30.0) -> dict:
    """Per-provider fraction of retained keeper labels above ``threshold`` (goal-relative x).

    TF-60 PR3 sec 6: the high-sweeper stratum leans on the full-tracking providers (sportec/GS); this
    reports each provider's >30 m coverage so a consumer knows the provenance of the high regime.
    """
    gx = np.asarray(labels["gk_x"], dtype=float)
    provs = np.asarray(provider_labels)
    out: dict[str, float] = {}
    for p in sorted({str(x) for x in provs.tolist()}):
        sel = provs == p
        out[p] = float(np.mean(gx[sel] > threshold)) if sel.any() else float("nan")
    return out


# ---- ADR-128 stage contract --------------------------------------------------------------------------
#
# validate (one unit) -> extract[match] -> cache (one unit) -> cv[fold] -> fit (one unit) ->
# importance (one inner_parallel unit) -> assemble (one unit, the bundle).
#
# Variant sharing (evidence E11): extract/cache declare only the extraction inputs, so `default` and
# `full` share them; cv/fit/importance declare the model inputs but NOT `variant`, so those two also share
# CV and the final fit. Only `assemble` (the `variant` label in metrics.json) differs.


def _to_bool(text: str) -> bool:
    return str(text).strip().lower() in ("1", "true", "yes", "on")


_PARAMS = (
    Param(
        "dataset",
        str,
        None,
        required=True,
        help="the staged flat dataset (shards/<gen>/ + _actions/ + home_teams.json)",
    ),
    Param("variant", str, "full", choices=("default", "full", "sweeper", "sweeper_position_only")),
    Param("feature_set", str, "faithful", choices=("faithful", "position_only")),
    Param("grid_x_max", float, None, help="TF-60 PR3 extended label ceiling (e.g. 52.5); None = DEFAULT_GHOST_GRID"),
    Param("subsample_fps", float, 1.0),
    Param("n_estimators", int, 500),
    Param("max_depth", int, 8),
    Param("cv_folds", int, 5),
    Param("carrier_beta", float, None),
    Param("carrier_gamma", float, None),
    Param("carrier_tolerance", float, None),
    Param("subsample_cap", int, None, help="cap total training samples (~36000 for the bundled 'default')"),
    Param("training_platform", str, None, help="recorded in metadata (default: the detected platform)"),
    Param("skip_permutation_importance", flag=True),
    Param("perm_importance_sample", int, 150000),
    Param("keeper_grouped", flag=True),
    Param("expansion_keepers", str, None, help="an .npy of 98-cohort keeper ids (relative = the dataset root)"),
    # Every staged flat dataset carries _actions/; this keys the extraction token exactly as the legacy
    # `bool(actions_by_game)` did.
    Param("with_actions", _to_bool, True),
)


def _model_declare(p) -> dict:
    """What cv/fit/importance numbers depend on beyond the extraction (NOT `variant`, E11)."""
    return {
        "feature_set": p["feature_set"],
        "grid_x_max": grid_spec_from_args(p["grid_x_max"]).x_max,
        "carrier_params": _carrier(p),
        "n_estimators": p["n_estimators"],
        "max_depth": p["max_depth"],
        "cv_folds": p["cv_folds"],
        "subsample_cap": p["subsample_cap"],
        "keeper_grouped": bool(p["keeper_grouped"]),
        "expansion_keepers": p["expansion_keepers"],
    }


def _cache_work(key, ctx) -> pd.DataFrame:
    """The whole-corpus combined table: every non-empty extract shard in sorted key order (the legacy
    `res.keys` order: sorted files, then games). Downstream stages `_unpack` it."""
    from scripts._stages import read_result, sorted_units, unit_states

    st = ctx.pipeline.stage("extract")
    gen = ctx.generation("extract")
    states = unit_states(ctx, st, sorted_units(ctx, st), verify=False)
    done = [k for k, v in states.items() if v == "done"]
    parts = [f for f in (read_result(ctx.store, f"{gen}/{k}", names=states.names) for k in done) if len(f)]
    if not parts:
        raise RuntimeError("no training samples extracted")
    return _combine_packs(parts)


def _combine_packs(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """The packs concatenated, with ``_game_id`` storable: a multi-provider corpus mixes int and str ids,
    and the legacy grouped on ``np.array(_game_id.tolist())``, which made every id a string -- the same
    strings `_stages.storable_id_column` stores, so the CV folds do not move."""
    from scripts._stages import combine_frames

    return combine_frames(parts, ids=("_game_id",))


_CORPUS: dict = {}


def _corpus(ctx):
    """``(features, labels, groups, provider_labels, keepers)`` from the cache, with the seeded subsample
    cap applied (PR-S81) -- loaded ONCE per worker (keyed by the cache token)."""
    tok = ctx.tokens["cache"]
    if tok not in _CORPUS:
        features, labels, groups, provider_labels, keepers = _unpack(ctx.upstream("cache"))
        cap = ctx.params["subsample_cap"]
        if cap is not None and len(features) > cap:
            rng = np.random.default_rng(42)
            keep = rng.choice(len(features), size=cap, replace=False)
            keep.sort()
            features = features.iloc[keep].reset_index(drop=True)
            labels = labels.iloc[keep].reset_index(drop=True)
            groups = groups[keep]
            provider_labels = provider_labels[keep]
            keepers = keepers[keep]
        _CORPUS.clear()
        _CORPUS[tok] = (features, labels, groups, provider_labels, keepers)
    return _CORPUS[tok]


def _folds(ctx):
    """The seeded CV split (either regime, spec 4.3) and the keeper-domain report (or None)."""
    features, _labels, groups, provider_labels, keepers = _corpus(ctx)
    p = ctx.params
    if p["keeper_grouped"]:
        from _ghost_domain import common_keeper_domain, keeper_folds

        if p["expansion_keepers"] is None:
            raise RuntimeError(
                "--keeper-grouped requires --expansion-keepers (the keeper ids present in the 98-match cohort)"
            )
        ek = Path(p["expansion_keepers"])
        ek = ek if ek.is_absolute() else _root(ctx) / ek
        expansion_keepers = set(np.load(ek, allow_pickle=True).astype(str).tolist())
        domain, dreport = common_keeper_domain(keepers, expansion_keepers=expansion_keepers, n_splits=p["cv_folds"])
        if set(keepers[domain].tolist()) & expansion_keepers:
            raise RuntimeError("a domain keeper is in the expansion cohort -- the common-domain exclusion failed")
        return list(keeper_folds(keepers, domain, n_splits=p["cv_folds"])), dreport
    from sklearn.model_selection import StratifiedGroupKFold

    cv = StratifiedGroupKFold(n_splits=p["cv_folds"], shuffle=True, random_state=42)
    return list(cv.split(features, provider_labels, groups)), None


def _cv_work(key, ctx) -> dict:
    from scripts._train_guard import require_training_sklearn
    from silly_kicks.tracking._ghost_gk import GhostGkModel

    require_training_sklearn()
    features, labels, _groups, provider_labels, _keepers = _corpus(ctx)
    folds, _dreport = _folds(ctx)
    train_idx, test_idx = folds[int(key)]
    p = ctx.params
    X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
    y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
    model = GhostGkModel(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        verbose=1,
        feature_set=p["feature_set"],
        grid_spec=grid_spec_from_args(p["grid_x_max"]),
    )
    model.fit(X_train, y_train, carrier_params=_carrier(p))
    preds = model.predict_mean(X_test)
    mae_x = float(np.mean(np.abs(preds[:, 0] - y_test["gk_x"].values)))
    mae_y = float(np.mean(np.abs(preds[:, 1] - y_test["gk_y"].values)))
    mae_euclid = float(
        np.mean(np.sqrt((preds[:, 0] - y_test["gk_x"].values) ** 2 + (preds[:, 1] - y_test["gk_y"].values) ** 2))
    )
    test_provs = provider_labels[test_idx]
    per_prov: dict[str, float] = {}
    for prov in np.unique(test_provs):
        mask = test_provs == prov
        per_prov[str(prov)] = float(
            np.mean(
                np.sqrt(
                    (preds[mask, 0] - y_test["gk_x"].values[mask]) ** 2
                    + (preds[mask, 1] - y_test["gk_y"].values[mask]) ** 2
                )
            )
        )
    return {
        "mae_x": mae_x,
        "mae_y": mae_y,
        "mae_euclidean": mae_euclid,
        "per_provider": per_prov,
        # TF-60 PR3: out-of-sample euclid MAE over the >30 m high-sweeper stratum (NaN if none).
        "high_sweeper_mae": high_sweeper_stratum_mae(
            preds, {"gk_x": y_test["gk_x"].values, "gk_y": y_test["gk_y"].values}
        ),
    }


def _fit_work(key, ctx):
    """Final fit + the BLOCKING parity-on-fresh-fit gate (Option A, ADR-016) + save + round-trip."""
    import tempfile

    from scripts._stages import FileSet
    from scripts._train_guard import require_training_sklearn
    from silly_kicks.tracking._ghost_gk import GhostGkModel

    require_training_sklearn()
    features, labels, groups, provider_labels, _keepers = _corpus(ctx)
    p = ctx.params
    cp = _carrier(p)
    final_model = GhostGkModel(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        verbose=1,
        feature_set=p["feature_set"],
        grid_spec=grid_spec_from_args(p["grid_x_max"]),
    )
    final_model.fit(features, labels, carrier_params=cp)
    _par_n = min(20000, len(features))
    _par_idx = np.random.default_rng(0).choice(len(features), _par_n, replace=False)
    _par_Xv = features.iloc[_par_idx][features.columns].values
    _sk_reg_x, _sk_reg_y = final_model._sk_reg_x, final_model._sk_reg_y
    _tree_nodes, _tree_nodes_y = final_model._tree_nodes, final_model._tree_nodes_y
    if _sk_reg_x is None or _sk_reg_y is None or _tree_nodes is None or _tree_nodes_y is None:
        raise RuntimeError("the fresh fit did not populate the regressors / tree arrays")
    _par_sk = np.column_stack([_sk_reg_x.predict(_par_Xv), _sk_reg_y.predict(_par_Xv)])
    _par_err = float(np.abs(final_model.predict_mean(features.iloc[_par_idx]) - _par_sk).max())
    _par_ncat = sum(int(t["is_categorical"].sum()) for t in _tree_nodes) + sum(
        int(t["is_categorical"].sum()) for t in _tree_nodes_y
    )
    print(f"PARITY GATE: max|predict_mean - sklearn| = {_par_err:.2e} over {_par_n} rows; n_cat = {_par_ncat}")
    if _par_err > 1e-6 or _par_ncat != 0:
        raise RuntimeError(f"BLOCKING parity gate FAILED (err={_par_err:.2e}, ncat={_par_ncat}); refusing to publish.")
    # S5.10: a stage reads no ctx.prov; adopt.stamp_provenance stamps training_commit + platform at adopt.
    final_model.training_platform = p["training_platform"]
    # Aggregate corpus provenance (providers + counts ONLY; spec 2026-07-20 S6).
    final_model.corpus_provenance = {
        "providers": sorted({str(x) for x in provider_labels.tolist()}),
        "n_games": len(set(groups.tolist())),
        "n_rows": len(features),
    }
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp) / "ghost_gk_v1"
        final_model.save(artifact_dir)
        loaded = GhostGkModel.load(artifact_dir)
        for attr in ("_tree_nodes", "_tree_nodes_y"):
            for i, (a, b) in enumerate(zip(getattr(final_model, attr), getattr(loaded, attr), strict=True)):
                np.testing.assert_array_equal(a, b, err_msg=f"{attr}[{i}]")
        if loaded._baseline_x != final_model._baseline_x or loaded._baseline_y != final_model._baseline_y:
            raise RuntimeError("round-trip verification: a baseline drifted")
        if loaded.carrier_params != cp:
            raise RuntimeError(f"carrier_params drift: {loaded.carrier_params} != {cp}")
        files = {q.name: q.read_bytes() for q in sorted(artifact_dir.iterdir()) if q.is_file()}
    return FileSet(files)


def _importance_work(key, ctx) -> dict:
    """Permutation importance (metrics-only: printed, never in metrics.json). ONE inner_parallel unit that
    calls scikit-learn with ``n_jobs = ctx.pool.workers`` (decision P11), so scikit-learn's own seed
    stream is unchanged; the model is the fit stage's saved artifact (bit-identical predictions)."""
    import tempfile

    from scripts._stages import materialize_fileset
    from scripts._train_guard import require_training_sklearn
    from silly_kicks.tracking._ghost_gk import GhostGkModel

    p = ctx.params
    if p["skip_permutation_importance"]:
        print("Skipping permutation importance (--skip-permutation-importance).")
        return {"skipped": True, "top": []}
    require_training_sklearn()
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.inspection import permutation_importance

    features, labels, _groups, _provs, _keepers = _corpus(ctx)
    with tempfile.TemporaryDirectory() as tmp:
        model = GhostGkModel.load(materialize_fileset(ctx, "fit", "_output", Path(tmp)))
    columns = features.columns
    pid_log = os.environ.get("SILLY_KICKS_IMPORTANCE_PID_LOG")  # test hook (decision P11); never set in a job

    class _SklearnWrapper(BaseEstimator, RegressorMixin):  # type: ignore[misc]
        def __init__(self, m: GhostGkModel | None = None) -> None:
            self.m = m

        def fit(self, X: np.ndarray, y: np.ndarray) -> _SklearnWrapper:
            return self  # already fitted

        def predict(self, X: np.ndarray) -> np.ndarray:
            if pid_log:
                with open(pid_log, "a", encoding="ascii") as fh:
                    fh.write(f"{os.getpid()}\n")
            if self.m is None:
                raise RuntimeError("the importance wrapper holds no model")
            return self.m.predict_mean(pd.DataFrame(X, columns=columns))[:, 0]

    cap = p["perm_importance_sample"]
    if cap and cap < len(features):
        idx = np.random.default_rng(42).choice(len(features), cap, replace=False)
        X, y = features.values[idx], labels["gk_x"].values[idx]
    else:
        X, y = features.values, labels["gk_x"].values
    n_jobs = ctx.pool.workers if ctx.pool is not None else 1
    print(f"Running permutation importance (5 repeats, n_jobs={n_jobs}) on {len(X)} eval rows...")
    pi = permutation_importance(
        _SklearnWrapper(m=model),
        X,
        y,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=42,
        n_jobs=n_jobs,
    )
    importances = sorted(
        zip(columns, pi.importances_mean, strict=True),  # type: ignore[reportAttributeAccessIssue]
        key=lambda x: -x[1],
    )
    print("Top 10 features:")
    for name, imp in importances[:10]:
        print(f"  {name}: {imp:.4f}")
    return {"skipped": False, "top": [[str(n), float(v)] for n, v in importances]}


def _assemble_work(key, ctx):
    """The legacy metrics block (fold aggregation, bias block, acceptance) + the bundle FileSet."""
    from scripts._corpus_identity import reproducibility
    from scripts._stages import FileSet, item_counters

    p = ctx.params
    features, labels, groups, provider_labels, _keepers = _corpus(ctx)
    folds_out = ctx.upstream("cv")
    fold_metrics = [folds_out[k] for k in sorted(folds_out)]
    _folds_unused, dreport = _folds(ctx) if p["keeper_grouped"] else (None, None)
    fit = ctx.upstream("fit")
    cp = _carrier(p)
    grid_spec = grid_spec_from_args(p["grid_x_max"])
    mae_x_vals = [m["mae_x"] for m in fold_metrics]
    mae_y_vals = [m["mae_y"] for m in fold_metrics]
    mae_e_vals = [m["mae_euclidean"] for m in fold_metrics]
    all_provs_set: set[str] = set()
    for m in fold_metrics:
        all_provs_set.update(m["per_provider"].keys())
    per_prov_agg: dict[str, float] = {}
    for prov in sorted(all_provs_set):
        vals = [m["per_provider"].get(prov, np.nan) for m in fold_metrics]
        per_prov_agg[prov] = float(np.nanmean(vals))
    _SHIPPED = ("rfcde_weights.npz", "metadata.json", "SHA256SUMS")
    artifact_bytes = sum(len(fit.files[f]) for f in _SHIPPED if f in fit.files)
    metrics = {
        "n_games": len(set(groups.tolist())),
        "n_samples": len(features),
        "n_providers": len(set(provider_labels.tolist())),
        "providers": sorted({str(x) for x in provider_labels.tolist()}),
        "cv_folds": p["cv_folds"],
        "subsample_fps": p["subsample_fps"],
        "variant": p["variant"],
        "carrier_params": cp,
        # ADR-067 M4, emitted by the trainer: a staged corpus carries no public-visibility proof, so it is
        # restricted (fail-closed). Commit-free (S5.10): adopt.stamp_provenance stamps the commit + the note.
        **reproducibility("restricted", provider_labels.tolist()),
        "hyperparameters": {"n_estimators": p["n_estimators"], "max_depth": p["max_depth"]},
        "cv_mae_x_mean": float(np.mean(mae_x_vals)),
        "cv_mae_x_std": float(np.std(mae_x_vals)),
        "cv_mae_y_mean": float(np.mean(mae_y_vals)),
        "cv_mae_y_std": float(np.std(mae_y_vals)),
        "cv_mae_euclidean_mean": float(np.mean(mae_e_vals)),
        "cv_mae_euclidean_std": float(np.std(mae_e_vals)),
        "per_provider_mae_euclidean": per_prov_agg,
        "grid_x_max": grid_spec.x_max,
        "high_sweeper_stratum_mae_mean": float(np.nanmean([m["high_sweeper_mae"] for m in fold_metrics])),
        "per_provider_high_sweeper_coverage": per_provider_high_sweeper_coverage(
            {"gk_x": labels["gk_x"].to_numpy()}, provider_labels
        ),
        "acceptance": {
            "overall_mae_lt_2m": float(np.mean(mae_e_vals)) < 2.0,
            "per_provider_mae_lt_3m": all(v < 3.0 for v in per_prov_agg.values()),
            "cross_fold_std_lt_05m": float(np.std(mae_e_vals)) < 0.5,
            "artifact_size_lt_15mb": artifact_bytes < 15_000_000,
        },
        "artifact_size_bytes": artifact_bytes,
    }
    if dreport is not None:
        metrics["keeper_grouped"] = {
            "n_domain_keepers": dreport.n_domain_keepers,
            "n_excluded_keepers": dreport.n_excluded_keepers,
            "underpowered": dreport.underpowered,
            "per_fold_mae_euclidean": mae_e_vals,
        }
    # spec 4.3 rev 5: the selection-bias limitation from the CORPUS-scoped counters (summed over every
    # extract unit in sorted key order), so a resumed extraction still reports the whole corpus.
    bias, bias_unrecorded = item_counters(ctx, "extract")
    if bias.get("bias_depth_detected_n") and bias.get("bias_depth_undetected_n"):
        metrics["detection_selection_bias"] = {
            "keeper_depth_detected_mean": _mean_from_counters(
                bias["bias_depth_detected_sum"], bias["bias_depth_detected_n"]
            ),
            "keeper_depth_undetected_mean": _mean_from_counters(
                bias["bias_depth_undetected_sum"], bias["bias_depth_undetected_n"]
            ),
            "ball_to_keeper_dist_detected_mean": _mean_from_counters(
                bias["bias_b2k_detected_sum"], bias["bias_b2k_detected_n"]
            ),
            "ball_to_keeper_dist_undetected_mean": _mean_from_counters(
                bias["bias_b2k_undetected_sum"], bias["bias_b2k_undetected_n"]
            ),
            "n_detected": int(bias["bias_depth_detected_n"]),
            "n_undetected": int(bias["bias_depth_undetected_n"]),
            "n_games_counters_unrecorded": int(bias_unrecorded),
            "note": (
                "Detected keeper frames are a SELECTION-BIASED sample (the camera sees the keeper "
                "when the ball is near him), so they over-represent the engaged/advanced keeper and "
                "under-sample the deep sweeper regime GKDV cares about. Depth = goal-relative x. "
                "This is a stated limitation, not a gate -- no rule in this cycle detects it. "
                "Both bias axes are measured here: keeper depth AND ball-to-keeper distance "
                "(goal-relative; detected frames are expected to show a SMALLER ball-to-keeper "
                "distance, which IS the selection mechanism). See spec 4.3 rev 5."
            ),
        }
    files = dict(fit.files)
    files["metrics.json"] = json.dumps(metrics, indent=2).encode("utf-8")  # the legacy json.dump bytes
    print("\n=== Acceptance Criteria ===")
    for k, passed in metrics["acceptance"].items():
        print(f"  {k}: {'PASS' if passed else 'FAIL'}")
    return FileSet(commit_free(files))  # S5.10: commit-free stage output (SHA256SUMS + provenance keys dropped)


PIPELINE = Pipeline(
    name="ghost_gk",
    module="scripts.train_ghost_gk",
    params=_PARAMS,
    stages=(
        _REUSABLE["validate"],  # ADR-129: declared in scripts._ghost_gk_stages
        _REUSABLE["extract"],
        Stage(
            name="cache",
            units=lambda ctx: ["only"],
            work=_cache_work,
            needs=("extract",),
            reduce=lambda r, ctx: r["only"],
        ),
        Stage(
            name="cv",
            units=lambda ctx: [int_key(k) for k in range(int(ctx.params["cv_folds"]))],
            work=_cv_work,
            needs=("cache",),
            reduce=lambda r, ctx: dict(r),
            declare=_model_declare,
        ),
        Stage(
            name="fit",
            units=lambda ctx: ["only"],
            work=_fit_work,
            needs=("cache",),
            reduce=lambda r, ctx: r["only"],
            declare=_model_declare,
            params=("training_platform",),
        ),
        Stage(
            name="importance",
            units=lambda ctx: ["only"],
            work=_importance_work,
            needs=("cache", "fit"),
            reduce=lambda r, ctx: r["only"],
            declare=_model_declare,
            params=("skip_permutation_importance", "perm_importance_sample"),
            inner_parallel=True,
            inner_width=8,
        ),
        Stage(
            name="assemble",
            units=lambda ctx: ["only"],
            work=_assemble_work,
            needs=("extract", "cache", "cv", "fit", "importance"),
            reduce=lambda r, ctx: r["only"],
            declare=_model_declare,
            params=("variant", "subsample_fps", "training_platform"),
        ),
    ),
    output="assemble",
)

if __name__ == "__main__":
    raise SystemExit(main(PIPELINE))
