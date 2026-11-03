"""Trainer: bundle the default XSuccessModel weights (TF-61).

Fits ``silly_kicks.xsuccess.XSuccessModel`` (END-BLIND, calibrated XGBoost) on the PUBLIC StatsBomb open-data
corpus. An ADR-128 pipeline (HFU spec 15.5a): ``refs`` -> ``extract`` (one store-cached unit per match: its real
on-ball rows) -> ``hpo`` (the Optuna study, its folds on the InnerPool) -> ``fit`` (the GroupKFold-by-match
out-of-fold metrics, the pooled fit, the pickle-free booster-JSON + metadata + SHA256SUMS bundle, the card).
``python -m scripts.train_xsuccess --run-all --store <STORE> --all-competitions --hpo-trials 30 --out <DIR>``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from scripts import _open_data_stages as od
from scripts._stages import FileSet, ItemStage, Param, Pipeline, Stage, main
from silly_kicks.xsuccess._features import FEATURE_NAMES

#: The columns one match's on-ball-row shard carries (the fit corpus for that match). END-BLIND:
#: no end_x/end_y (xSuccess never reads the realized end). ``non_action`` rows are dropped here.
_SHARD_COLUMNS = ["game_id", "period_id", "type_id", "result_id", "bodypart_id", "start_x", "start_y", "time_seconds"]
_SHARD_SCHEMA_VERSION = "xsuccess-shard-1"


def onball_rows_for_match(actions: pd.DataFrame, match_id: object) -> pd.DataFrame:
    """The real on-ball rows for one match -- its fit shard (pure). ``non_action`` dropped."""
    non_action = spadlconfig.actiontype_id["non_action"]
    p = actions[actions["type_id"] != non_action]
    out = p[["period_id", "type_id", "result_id", "bodypart_id", "start_x", "start_y", "time_seconds"]].copy()
    out.insert(0, "game_id", str(match_id))
    return out.reset_index(drop=True)


def cross_val_metrics(pooled: pd.DataFrame, *, params: dict | None, family: str = "xgboost") -> dict:
    """GroupKFold-by-match out-of-fold AUC / Brier / calibration-in-the-large + per-type reliability."""
    from silly_kicks.xsuccess import XSuccessModel

    y, splits = _cv_splits(pooled)
    oof = np.full(len(y), np.nan)
    if not splits:
        oof = XSuccessModel().fit(pooled, family=family, params=params).predict_success(pooled)
    else:
        import time

        from scripts._study_shared import print_fold

        t0 = time.monotonic()
        for k, (tr, te) in enumerate(splits, start=1):
            m = XSuccessModel().fit(pooled.iloc[tr], family=family, params=params)
            oof[te] = m.predict_success(pooled.iloc[te])
            print_fold("xsuccess", k, len(splits), t0)
    return _oof_metrics(pooled, y, oof)


def _cv_splits(pooled: pd.DataFrame) -> tuple[np.ndarray, list]:
    """The labels and the GroupKFold-by-match folds (none for fewer than 2 matches)."""
    from sklearn.model_selection import GroupKFold

    y = (pooled["result_id"].to_numpy() == spadlconfig.result_id["success"]).astype(int)
    groups = pooled["game_id"].to_numpy().astype(str)
    n_splits = min(5, len(np.unique(groups)))
    return y, (list(GroupKFold(n_splits=n_splits).split(pooled, y, groups)) if n_splits >= 2 else [])


def _oof_metrics(pooled: pd.DataFrame, y: np.ndarray, oof: np.ndarray) -> dict:
    """Out-of-fold AUC / Brier / calibration-in-the-large + per-type reliability (pure)."""
    from sklearn.metrics import roc_auc_score

    keep = np.isfinite(oof)
    yk, ok = y[keep], oof[keep]
    auc = float(roc_auc_score(yk, ok)) if len(np.unique(yk)) > 1 else float("nan")
    brier = float(np.mean((ok - yk) ** 2)) if len(ok) else float("nan")
    # per-type reliability (predicted vs observed) + calibration-in-the-large
    per_type = {}
    tnames = spadlconfig.actiontypes
    types_k = pooled["type_id"].to_numpy()[keep]
    for t in np.unique(types_k):
        m = types_k == t
        if m.sum() >= 20:
            per_type[tnames[int(t)]] = {"pred": float(ok[m].mean()), "obs": float(yk[m].mean()), "n": int(m.sum())}
    return {
        "auc": auc,
        "brier": brier,
        "base_rate": float(yk.mean()) if len(yk) else float("nan"),
        "calibration_in_the_large": {"sum_pred": float(ok.sum()), "n_success": int(yk.sum())},
        "per_type_reliability": per_type,
        "n_oof": len(ok),
    }


def _hpo_params(pooled: pd.DataFrame, n_trials: int, *, pool=None) -> dict | None:
    """Optional Optuna HPO over XGBoost params via the ruthless CachedObjective (ADR-009). With an InnerPool of
    more than one worker, each trial's CV folds run in parallel (HFU spec 15.7); the result equals the serial one."""
    if n_trials <= 0:
        return None
    import tempfile

    import optuna

    from silly_kicks.xsuccess._features import xsuccess_features
    from silly_kicks.xsuccess._objective import XSuccessObjective

    success = spadlconfig.result_id["success"]
    X = xsuccess_features(pooled)
    y = (pooled["result_id"].to_numpy() == success).astype(int)
    g = pooled["game_id"].to_numpy().astype(str)
    obj = XSuccessObjective(fold={"corpus": [(X, y, g)]})
    inv = obj.prepare()

    def _optuna_objective(trial):
        from ruthless.result import Candidate

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        return obj.evaluate_patch(inv, Candidate(id=str(trial.number), params=params), fold_map=fold_map)["logloss"]

    from scripts._study_shared import TrialPrinter

    printer = TrialPrinter("xsuccess", n_trials, metric="logloss")  # one line per trial (launch-4 finding F21)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # the children map files from here
        fold_map = _fold_map(pool, inv, Path(tmp)) if pool is not None and pool.workers > 1 else None
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(_optuna_objective, n_trials=n_trials, callbacks=[printer.optuna_callback])
    return study.best_params


#: Per InnerPool child: the memory-mapped invariant and the fold plan, loaded once (keyed by path).
_FOLD_DATA: dict = {}


def _fold_task(x_path: str, y_path: str, plan_path: str, fold_i: int, params: dict):
    """One CV fold of one HPO trial in an InnerPool child (module-level: pickled by reference); nothing large is
    pickled: the child maps ``X``/``y`` and the fold plan from the job-local files once."""
    from silly_kicks.xsuccess._objective import _fold_score

    key = (x_path, y_path, plan_path)
    if key not in _FOLD_DATA:
        z = np.load(plan_path)  # positional: arr_{2i} = fold i's train rows, arr_{2i+1} = its test rows
        folds = [(z[f"arr_{2 * i}"], z[f"arr_{2 * i + 1}"]) for i in range(len(z.files) // 2)]
        _FOLD_DATA.clear()
        _FOLD_DATA[key] = (np.load(x_path, mmap_mode="r"), np.load(y_path), folds)
    X, y, folds = _FOLD_DATA[key]
    tr, te = folds[fold_i]
    return _fold_score(X, y, tr, te, params)


def _fold_map(pool, inv, work: Path):
    """Write the invariant and the fold plan once; map each trial's folds on ``pool``, back in fold order."""
    from scripts._stages import int_key, ordered_values
    from silly_kicks.xsuccess._objective import _cv_splits

    splits = _cv_splits(inv.y, inv.groups)
    x_path, y_path, plan_path = work / "X.npy", work / "y.npy", work / "plan.npz"
    np.save(x_path, inv.X)
    np.save(y_path, inv.y)
    np.savez(plan_path, *[a for tr, te in splits for a in (tr, te)])
    paths = (str(x_path), str(y_path), str(plan_path))

    def fold_map(params: dict) -> list:
        items = {int_key(i): (*paths, i, params) for i in range(len(splits))}
        return ordered_values(pool.map_keyed(_fold_task, items))

    return fold_map


def render_model_card(metrics: dict) -> str:
    auc = float(metrics.get("auc", float("nan")))
    brier = float(metrics.get("brier", float("nan")))
    base = float(metrics.get("base_rate", float("nan")))
    commit = metrics.get("training_commit", "?")
    providers = metrics.get("providers", "?")
    n_rows = metrics.get("n_rows", "?")
    n_matches = metrics.get("n_matches", "?")
    digest = metrics.get("corpus_match_ids_sha256")  # the corpus identity verify checks (HFU spec 14.13)
    identity = f"\nCorpus identity: {metrics.get('corpus_n_matches')} matches, sha256 {digest}." if digest else ""
    return f"""# xSuccess model -- `default` variant (TF-61)

**What it is.** An event-only, END-BLIND action-completion probability `P(success | pre-action
context)` over ALL on-ball action types, from Paul/Klemp/Memmert 2025 ("Beyond Outcome Bias").
Calibrated XGBoost; xgboost + isotonic at fit, xgboost + pure-numpy isotonic at serve (no runtime
sklearn). Loaded via `silly_kicks.xsuccess.XSuccessModel.bundled()`. Consumed by
`VAEP.rate_adjusted` to remove outcome bias.

**END-BLIND (correctness, not a caveat).** Features use only start-anchored geometry + type +
bodypart + time -- NEVER the realized action end. For a failed action the SPADL end IS the outcome
(the interception/death point), so an end-using completion model would be a postdiction that
re-injects the very bias VAEP_adjusted removes. Guarded by an end-location invariance test.

**Label construct.** SPADL `result_id == success`. All on-ball action types retained (some are
structurally one-sided, e.g. clearance/bad_touch; the model learns their base rate).

**Training corpus + metrics.** {n_matches} match(es) from `{providers}` ({n_rows} on-ball rows).
GroupKFold-by-match out-of-fold: AUC {auc:.3f}, Brier {brier:.3f} vs base rate {base:.3f}. Per-type
reliability + calibration-in-the-large in `metrics.json`.{identity}

**Missing-value policy.** A non-finite feature yields a NaN probability (never fabricated).

**Provenance.** `metrics.json` records `training_commit` ({commit}) and the tree state (clean for the
bundle). Pickle-free booster-JSON + metadata + SHA256 envelope with a feature contract + chirality
probe; `load()` is fail-closed (ADR-011/016/040/050). Every bundled model carries a card (ADR-088).
Attribution: Paul/Klemp/Memmert 2025; XGBoost (Chen & Guestrin 2016) -- see NOTICE.
"""


_FIT_ROWS: dict = {}


def _fit_task(rows_path: str, plan_path: str, fold_i: int, family: str, params: dict | None):
    """One `fit` inner unit in an InnerPool child (module-level: pickled by reference). ``fold_i >= 0``: the fold's
    model predicts its test rows; ``-1``: the final fit on every row, returned as its saved files and feature set.
    The child reads the job-local rows and fold plan once and caches them per process (HFU spec 16.2)."""
    import tempfile

    from silly_kicks.xsuccess import XSuccessModel

    key = (rows_path, plan_path)
    if key not in _FIT_ROWS:
        z = np.load(plan_path)  # positional: arr_{2i} = fold i's train rows, arr_{2i+1} = its test rows
        folds = [(z[f"arr_{2 * i}"], z[f"arr_{2 * i + 1}"]) for i in range(len(z.files) // 2)]
        _FIT_ROWS.clear()
        _FIT_ROWS[key] = (pd.read_parquet(rows_path), folds)
    pooled, folds = _FIT_ROWS[key]
    if fold_i >= 0:
        tr, te = folds[fold_i]
        return XSuccessModel().fit(pooled.iloc[tr], family=family, params=params).predict_success(pooled.iloc[te])
    model = XSuccessModel().fit(pooled, family=family, params=params)
    with tempfile.TemporaryDirectory() as tmp:
        model.save(Path(tmp))
        files = {f.name: f.read_bytes() for f in sorted(Path(tmp).iterdir()) if f.is_file()}
    return files, model.feature_set


def _fit_map(pool, pooled: pd.DataFrame, *, family: str, params: dict | None, work: Path) -> tuple[dict, dict, str]:
    """The out-of-fold metrics, the final model's files and its feature set, with the folds and the final fit as
    InnerPool units; the final fit (the longest unit) is submitted first; the combination is in key order, so the
    result equals the serial `cross_val_metrics` + fit (HFU spec 16.2)."""
    import sys
    import time

    from scripts._stages import int_key, ordered_values

    y, splits = _cv_splits(pooled)
    work.mkdir(parents=True, exist_ok=True)
    rows_path, plan_path = work / "rows.parquet", work / "plan.npz"
    pooled.to_parquet(rows_path, index=False)
    np.savez(plan_path, *[a for tr, te in splits for a in (tr, te)])
    paths = (str(rows_path), str(plan_path))
    items = {
        "final": (*paths, -1, family, params),
        **{int_key(i): (*paths, i, family, params) for i in range(len(splits))},
    }
    t0, done = time.monotonic(), [0]

    def progress(_key, _value) -> None:
        done[0] += 1
        print(
            f"[progress] xsuccess fit {done[0]}/{len(items)} in {time.monotonic() - t0:.1f} s",
            file=sys.stderr,
            flush=True,
        )

    *preds, (files, feature_set) = ordered_values(pool.map_keyed(_fit_task, items, on_done=progress))
    oof = np.full(len(y), np.nan)
    for (_tr, te), p in zip(splits, preds, strict=True):
        oof[te] = p
    return _oof_metrics(pooled, y, oof), files, feature_set


def _extract_reduce(results, ctx) -> pd.DataFrame:
    """The per-match on-ball rows in sorted key order (the legacy sorted shard glob)."""
    from scripts._stages import combine_frames

    frames = list(results.values())
    return combine_frames(frames, ids=("game_id",)) if frames else pd.DataFrame(columns=_SHARD_COLUMNS)


def _hpo_work(key, ctx) -> dict:
    """The HPO study on the pooled rows (none for ``per_type_logistic`` or 0 trials); the folds on ``ctx.pool``."""
    pooled, n_trials = ctx.upstream("extract"), ctx.params["hpo_trials"]
    best = _hpo_params(pooled, n_trials, pool=ctx.pool) if ctx.params["family"] == "xgboost" else None
    return {"best_params": best}


def _fit_work(key, ctx):
    """The out-of-fold metrics at the best parameters, the pooled fit and the bundle (model, card, metrics)."""
    import tempfile

    from scripts._corpus_identity import open_data_identity
    from silly_kicks.xsuccess import XSuccessModel

    pooled, family = ctx.upstream("extract"), ctx.params["family"]
    if not len(pooled):
        raise SystemExit("no on-ball rows collected from the corpus; nothing to fit")
    best = ctx.upstream("hpo")["best_params"]
    # The folds and the final fit on the InnerPool (HFU spec 16.2); a width-1 (escalated) job runs the serial
    # composition in-process, as the hpo stage does -- the same result either way.
    if ctx.pool is not None and ctx.pool.workers > 1 and _cv_splits(pooled)[1]:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            metrics, files, feature_set = _fit_map(ctx.pool, pooled, family=family, params=best, work=Path(tmp))
    else:
        metrics = cross_val_metrics(pooled, params=best, family=family)
        model = XSuccessModel().fit(pooled, family=family, params=best)
        feature_set = model.feature_set
        with tempfile.TemporaryDirectory() as tmp:
            model.save(Path(tmp))
            files = {f.name: f.read_bytes() for f in sorted(Path(tmp).iterdir()) if f.is_file()}
    # The bundled metadata.json carries the training provenance (spec S5.5, ADR-052/056); it is not in SHA256SUMS.
    meta = json.loads(files["metadata.json"])
    meta |= {"training_commit": ctx.prov["commit"], "run_tree_dirty": ctx.prov["dirty"]}
    files["metadata.json"] = (json.dumps(meta, indent=2) + "\n").encode("utf-8")
    refs = ctx.upstream("refs")
    out = {
        **metrics,
        "n_rows": len(pooled),
        "n_matches": int(pooled["game_id"].nunique()),
        "providers": "statsbomb-open " + ", ".join(f"(comp {c}, season {s})" for c, s in refs["competitions"]),
        "hpo_trials": ctx.params["hpo_trials"],
        **open_data_identity([r["match_id"] for r in refs["refs"]]),  # verify compares it with the lock (14.3)
        "hpo_best_params": best,
        "feature_set": feature_set,
        "training_commit": ctx.prov["commit"],
        "run_tree_dirty": ctx.prov["dirty"],
        "run_tree_state": ctx.prov.get("tree_state"),
        **od.pass_fields(ctx),
    }
    files["MODEL_CARD.md"] = render_model_card(out).encode("utf-8")
    files["metrics.json"] = json.dumps(out, indent=2, default=str).encode("utf-8")
    print(json.dumps(out, indent=2, default=str))
    return FileSet(files)


PIPELINE = Pipeline(
    name="xsuccess",
    module="scripts.train_xsuccess",
    params=(
        *od.PARAMS,
        Param("hpo_trials", int, 0, help="Optuna HPO trials (0 = default params)"),
        Param("family", str, "xgboost", help="xgboost (default) or the per_type_logistic fallback (spec S5.4)"),
    ),
    stages=(
        Stage(
            name="refs",
            units=lambda ctx: ["only"],
            work=od.refs_work,
            reduce=lambda r, ctx: r["only"],
            params=od.REFS_PARAMS,
            uses_dataset=True,
            dataset_files=lambda ctx: (),  # open data from the snapshot: no dataset file
        ),
        ItemStage(
            name="extract",
            refs=od.extract_refs,
            key=lambda ref: ref.key,
            load=od.load_open_match,
            work=lambda lm, ctx: onball_rows_for_match(lm.actions, lm.match_id),
            needs=("refs",),
            reduce=_extract_reduce,
            declare=lambda p: {
                "model": "XSuccessModel",
                "schema": _SHARD_SCHEMA_VERSION,
                "feature_names": list(FEATURE_NAMES),
            },
            cross_commit_version="xsuccess-shard-1",
            uses_dataset=True,
            max_failures=None,  # the legacy for_each recorded failures and fit over the rest
        ),
        Stage(
            name="hpo",
            units=lambda ctx: ["only"],
            work=_hpo_work,
            needs=("extract",),
            reduce=lambda r, ctx: r["only"],
            params=("hpo_trials", "family"),
            inner_parallel=True,  # each trial's 5 CV folds on the InnerPool (HFU spec 15.7)
            inner_width=5,
            timeout_s=14400,  # ~2x the projected 30-trial average (spec 15.7)
        ),
        Stage(
            name="fit",
            units=lambda ctx: ["only"],
            work=_fit_work,
            needs=("refs", "extract", "hpo"),
            reduce=lambda r, ctx: r["only"],
            params=("family",),
            inner_parallel=True,  # the CV folds and the final fit on the InnerPool (HFU spec 16.2)
            inner_width=3,  # measured peak 7.26 GiB per unit (spec 16.7): 3 fit the 24 GiB budget
            timeout_s=14400,  # ~4,000-6,200 s expected (spec 16.7)
        ),
    ),
    output="fit",
)

if __name__ == "__main__":
    raise SystemExit(main(PIPELINE))
