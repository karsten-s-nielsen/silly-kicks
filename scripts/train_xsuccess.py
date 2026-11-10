"""Trainer: bundle the default XSuccessModel weights (TF-61).

Fits ``silly_kicks.xsuccess.XSuccessModel`` (END-BLIND, calibrated XGBoost) on the PUBLIC StatsBomb open-data
corpus. An ADR-128 pipeline (HFU spec 15.5a): ``refs`` -> ``extract`` (one store-cached unit per match: its real
on-ball rows) -> ``hpo`` (the Optuna study, its folds on the InnerPool) -> ``fit`` (36 units, one xgboost fit
each: the final fit and the GroupKFold-by-match folds with their calibration splits; the reduce gives the
out-of-fold metrics, the pickle-free booster-JSON + metadata + SHA256SUMS bundle and the card).
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


#: The unit grid of the `fit` stage (HFU fit-units spec 3.2). One calibrated fit is 6 xgboost fits: the model
#: (`main`) and one per calibration split (`cal0`..`cal4`). The stage runs 6 calibrated fits: the final fit and
#: one per GroupKFold-by-match fold. So one unit = one xgboost fit, and the driver shards the 36 units over jobs.
#: The grid is FIXED, so a `--plan` reads nothing; a unit that the corpus does not have returns `ABSENT`.
_FITS: tuple[str, ...] = ("final", "fold0", "fold1", "fold2", "fold3", "fold4")
_PARTS: tuple[str, ...] = ("main", "cal0", "cal1", "cal2", "cal3", "cal4")
FIT_UNITS: tuple[str, ...] = tuple(sorted(f"{fit}-{part}" for fit in _FITS for part in _PARTS))
#: The result of a unit that does not exist for the input: a fold beyond the fold count, a calibration split
#: beyond the fit's split count, and every calibration unit of a family without calibration.
ABSENT: dict = {"absent": True}
_SEED = 42  # `XSuccessModel.fit`'s default seed, the one the serial fit used


def _npy(values: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    np.save(buf, values)
    return buf.getvalue()


def _from_npy(data: bytes) -> np.ndarray:
    import io

    return np.load(io.BytesIO(data))


def _fit_rows(splits: list, fit: str, n: int) -> np.ndarray | None:
    """The pooled row positions that calibrated fit ``fit`` trains on: every row for ``final``, fold ``i``'s
    train rows for ``fold<i>``; ``None`` when the corpus has no such fold."""
    if fit == "final":
        return np.arange(n)
    i = int(fit.removeprefix("fold"))
    return splits[i][0] if i < len(splits) else None


def fit_unit(pooled: pd.DataFrame, key: str, *, family: str, params: dict | None):
    """One `fit` unit ``<fit>-<part>`` (HFU fit-units spec 3.2). The folds are `_cv_splits(pooled)`: computed on
    ALL pooled rows; the finite-feature filter runs after the fold selection, inside the fit steps.

    - ``<fit>-cal<j>``: calibration split ``j`` of the fit: a one-column frame ``p`` (xgboost's float32).
    - ``fold<i>-main``: the booster of the fold; a one-column frame ``raw``, its prediction on the fold's test
      rows BEFORE the isotonic step (NaN for a row with a non-finite feature).
    - ``final-main``: a `FileSet` with ``model.json`` (the saved model) and ``probe_raw.npy`` (the raw prediction
      on the chirality probe, from the in-memory booster); with no folds also ``in_sample_raw.npy``.
    - a unit that the input does not have: `ABSENT`.

    The ``per_type_logistic`` family has no calibration: its ``main`` units fit the whole model and return final
    predictions; its ``cal`` units are `ABSENT`."""
    import tempfile

    from silly_kicks.xsuccess import XSuccessModel
    from silly_kicks.xsuccess import _model as xmodel
    from silly_kicks.xsuccess._features import xsuccess_features

    fit, _, part = key.partition("-")
    _y, splits = _cv_splits(pooled)
    rows = _fit_rows(splits, fit, len(pooled))
    if rows is None:
        return dict(ABSENT)
    train = pooled.iloc[rows]
    test = None if fit == "final" else pooled.iloc[splits[int(fit.removeprefix("fold"))][1]]

    if family != "xgboost":
        if part != "main":
            return dict(ABSENT)
        model = XSuccessModel().fit(train, family=family, params=params)
        if test is not None:
            return pd.DataFrame({"raw": model.predict_success(test)})
        with tempfile.TemporaryDirectory() as tmp:
            model.save(Path(tmp))
            files = {"model.json": (Path(tmp) / "model.json").read_bytes()}
        files["probe_raw.npy"] = _npy(model.predict_success(xmodel._chirality_probe_actions()))
        if not splits:
            files["in_sample_raw.npy"] = _npy(model.predict_success(pooled))
        return FileSet(files)

    X, y, _tid = xmodel._training_arrays(train)
    p = xmodel._xgb_params(params, _SEED)
    if part != "main":
        cal = xmodel._calibration_splits(X, y, _SEED)
        j = int(part.removeprefix("cal"))
        if j >= len(cal):
            return dict(ABSENT)
        return pd.DataFrame({"p": xmodel._calibration_fold_proba(X, y, p, *cal[j])})
    booster = xmodel._fit_booster(X, y, p)
    if test is not None:
        return pd.DataFrame({"raw": xmodel._raw_predict(booster, xsuccess_features(test))})
    with tempfile.TemporaryDirectory() as tmp:
        booster.save_model(str(Path(tmp) / "model.json"))
        files = {"model.json": (Path(tmp) / "model.json").read_bytes()}
    files["probe_raw.npy"] = _npy(xmodel._raw_predict(booster, xsuccess_features(xmodel._chirality_probe_actions())))
    if not splits:
        files["in_sample_raw.npy"] = _npy(xmodel._raw_predict(booster, xsuccess_features(pooled)))
    return FileSet(files)


def fit_reduce(pooled: pd.DataFrame, results, *, family: str, params: dict | None) -> tuple[dict, dict, str]:
    """The out-of-fold metrics, the model files (``model.json``, ``metadata.json``, ``SHA256SUMS``) and the feature
    set from the 36 unit results. It fits the isotonic regression of each calibrated fit from its ``cal`` units,
    calibrates each fold's raw predictions, and assembles the out-of-fold vector over ALL pooled rows in fold
    order: the composition of the former serial fit, so the bytes are the same. It never loads a booster."""
    import tempfile

    from silly_kicks.xsuccess import _model as xmodel

    xgboost_family = family == "xgboost"
    y, splits = _cv_splits(pooled)

    def thresholds(fit: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not xgboost_family:
            return None, None
        rows = _fit_rows(splits, fit, len(pooled))
        if rows is None:  # the reduce asks only for `final` and the folds that `splits` has
            raise ValueError(f"the corpus has no fit {fit!r}")
        X, yk, _tid = xmodel._training_arrays(pooled.iloc[rows])
        cal = xmodel._calibration_splits(X, yk, _SEED)
        if not cal:
            return None, None
        probas = [results[f"{fit}-cal{j}"]["p"].to_numpy() for j in range(len(cal))]
        return xmodel._isotonic_thresholds(xmodel._calibration_oof(len(yk), cal, probas), yk)

    final = results["final-main"]
    iso_x, iso_y = thresholds("final")
    if splits:
        oof = np.full(len(y), np.nan)
        for i, (_tr, te) in enumerate(splits):
            oof[te] = xmodel._calibrate(results[f"fold{i}-main"]["raw"].to_numpy(), *thresholds(f"fold{i}"))
    else:  # fewer than 2 matches: the final model scored in sample, as the serial fit did
        oof = xmodel._calibrate(_from_npy(final.files["in_sample_raw.npy"]), iso_x, iso_y)
    metrics = _oof_metrics(pooled, y, oof)

    payload = xmodel._metadata_payload(
        feature_names=FEATURE_NAMES,
        feature_set=family,
        params=xmodel._xgb_params(params, _SEED) if xgboost_family else dict(xmodel._DEFAULT_PARAMS),
        iso_x=iso_x,
        iso_y=iso_y,
        probe_prediction=xmodel._calibrate(_from_npy(final.files["probe_raw.npy"]), iso_x, iso_y).tolist(),
    )
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "model.json").write_bytes(final.files["model.json"])
        xmodel._write_metadata_and_sums(Path(tmp), payload)
        files = {f.name: f.read_bytes() for f in sorted(Path(tmp).iterdir()) if f.is_file()}
    return metrics, files, family


def _pooled(ctx) -> pd.DataFrame:
    pooled = ctx.upstream("extract")
    if not len(pooled):
        raise SystemExit("no on-ball rows collected from the corpus; nothing to fit")
    return pooled


def _fit_unit_work(key, ctx):
    """One xgboost fit of the `fit` stage (`fit_unit`) at the study's best parameters."""
    return fit_unit(_pooled(ctx), key, family=ctx.params["family"], params=ctx.upstream("hpo")["best_params"])


def _fit_reduce(results, ctx):
    """The bundle (model, metadata, card, metrics) from the 36 unit results (`fit_reduce`)."""
    from scripts._corpus_identity import open_data_identity

    pooled, family = _pooled(ctx), ctx.params["family"]
    best = ctx.upstream("hpo")["best_params"]
    metrics, files, feature_set = fit_reduce(pooled, results, family=family, params=best)
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
        Stage(  # 36 units: one xgboost fit each (HFU fit-units spec 3.2); the reduce assembles the bundle
            name="fit",
            units=lambda ctx: list(FIT_UNITS),
            work=_fit_unit_work,
            needs=("refs", "extract", "hpo"),
            reduce=_fit_reduce,
            params=("family",),
        ),
    ),
    output="fit",
)

if __name__ == "__main__":
    raise SystemExit(main(PIPELINE))
