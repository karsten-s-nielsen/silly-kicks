"""Trainer: bundle the default XSuccessModel weights (TF-61).

Fits ``silly_kicks.xsuccess.XSuccessModel`` (END-BLIND, calibrated XGBoost) on the PUBLIC StatsBomb
open-data corpus and writes the pickle-free booster-JSON + metadata + SHA256SUMS artifact via
``model.save(...)``, plus an out-of-fold ``metrics.json`` stamping ``training_commit`` (ADR-052 /
ADR-011 discipline, mirroring ``train_pass_completion.py`` and the other weight trainers). Inference
imports no sklearn; xgboost + sklearn are used only during the fit/HPO.

ADR-128 stage contract (HFU R3-10 / D12; it replaced the one-unit adapter): ``refs`` (the corpus + match
references, one unit) -> ``extract[match]`` (an ``ItemStage``: one shard of real on-ball rows per open-data
match, fetched per unit, resume-before-download) -> ``study`` (the optional Optuna TPE search, one unit by
construction; a no-op at ``--hpo-trials 0``, the campaign default) -> ``cv[fold]`` (one unit per GroupKFold
fold: its out-of-fold predictions; the reduce assembles the OOF array in row order and computes the metrics)
-> ``fit`` (the pooled fit + calibration, ``save``, the stamped ``metadata.json``, the card). ``--run-all
--out <DIR>`` writes the flat bundle into ``<DIR>``: the model files, ``MODEL_CARD.md`` and ``metrics.json``
(owner ruling 2026-10-30: the bundle ships ``metrics.json``, as every other campaign-trained bundle does);
``adopt`` copies it into ``silly_kicks/xsuccess/weights/``. The legacy CLI is kept verbatim as the test oracle
(``tests/scripts/_legacy_trainers``).

Usage (StatsBomb credentials UNSET so open-data mode is used):
  python -m scripts.train_xsuccess --store <DIR> --run-all --out <DIR> [--competition-id 43 --season-id 106]
      [--competitions 43:106,55:43] [--hpo-trials N] [--family xgboost|per_type_logistic]
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from scripts._stages import FileSet, ItemStage, Param, Pipeline, Stage, cli_main, int_key

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


def _labels(pooled: pd.DataFrame) -> np.ndarray:
    return (pooled["result_id"].to_numpy() == spadlconfig.result_id["success"]).astype(int)


def cv_folds(pooled: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """The GroupKFold-by-match ``(train, test)`` folds, or None when there are fewer than 2 groups (then the
    out-of-fold predictions are the in-sample predictions of one fit on everything)."""
    from sklearn.model_selection import GroupKFold

    groups = pooled["game_id"].to_numpy().astype(str)
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        return None
    return list(GroupKFold(n_splits=n_splits).split(pooled, _labels(pooled), groups))


def fold_predictions(
    pooled: pd.DataFrame, fold: tuple[np.ndarray, np.ndarray] | None, *, params: dict | None, family: str
) -> np.ndarray:
    """The out-of-fold predictions of ONE fold (its test rows), or of the single in-sample fit (``None``)."""
    from silly_kicks.xsuccess import XSuccessModel

    if fold is None:
        return XSuccessModel().fit(pooled, family=family, params=params).predict_success(pooled)
    tr, te = fold
    return XSuccessModel().fit(pooled.iloc[tr], family=family, params=params).predict_success(pooled.iloc[te])


def cross_val_metrics(pooled: pd.DataFrame, *, params: dict | None, family: str = "xgboost") -> dict:
    """GroupKFold-by-match out-of-fold AUC / Brier / calibration-in-the-large + per-type reliability."""
    folds = cv_folds(pooled)
    if folds is None:
        return metrics_from_oof(pooled, fold_predictions(pooled, None, params=params, family=family))
    oof = np.full(len(pooled), np.nan)
    for tr, te in folds:
        oof[te] = fold_predictions(pooled, (tr, te), params=params, family=family)
    return metrics_from_oof(pooled, oof)


def metrics_from_oof(pooled: pd.DataFrame, oof: np.ndarray) -> dict:
    """The out-of-fold metrics' arithmetic (``cross_val_metrics`` before the HFU R3-10 split, unchanged)."""
    from sklearn.metrics import roc_auc_score

    y = _labels(pooled)
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


def _hpo_params(pooled: pd.DataFrame, n_trials: int) -> dict | None:
    """Optional Optuna HPO over XGBoost params via the ruthless CachedObjective (ADR-009)."""
    if n_trials <= 0:
        return None
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
        return obj.evaluate_patch(inv, Candidate(id=str(trial.number), params=params))["logloss"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(_optuna_objective, n_trials=n_trials)
    return study.best_params


def render_model_card(metrics: dict) -> str:
    auc = float(metrics.get("auc", float("nan")))
    brier = float(metrics.get("brier", float("nan")))
    base = float(metrics.get("base_rate", float("nan")))
    commit = metrics.get("training_commit", "?")
    providers = metrics.get("providers", "?")
    n_rows = metrics.get("n_rows", "?")
    n_matches = metrics.get("n_matches", "?")
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
reliability + calibration-in-the-large in `metrics.json`.

**Missing-value policy.** A non-finite feature yields a NaN probability (never fabricated).

**Provenance.** `metrics.json` records `training_commit` ({commit}) and the tree state (clean for the
bundle). Pickle-free booster-JSON + metadata + SHA256 envelope with a feature contract + chirality
probe; `load()` is fail-closed (ADR-011/016/040/050). Every bundled model carries a card (ADR-088).
Attribution: Paul/Klemp/Memmert 2025; XGBoost (Chen & Guestrin 2016) -- see NOTICE.
"""


_PARAMS = (
    Param("dataset", str, "sb_open", help="declared dataset name (scripts/hf_jobs/datasets.toml)"),
    Param("competition_id", int, 43, help="open-data competition (default 43 = World Cup)"),
    Param("season_id", int, 106, help="open-data season (default 106 = 2022)"),
    Param(
        "competitions",
        str,
        None,
        help="comma-sep comp:season pairs (e.g. '43:106,55:43'); overrides --competition-id/--season-id",
    ),
    Param("max_per_provider", int, None, help="cap the number of matches"),
    Param("hpo_trials", int, 0, help="Optuna HPO trials (0 = default params)"),
    Param(
        "family",
        str,
        "xgboost",
        choices=("xgboost", "per_type_logistic"),
        help="model family: calibrated XGBoost (default) or the per-type-logistic ADR-009 fallback",
    ),
    Param("match_ids_json", str, None, help='JSON {"statsbomb": ["3869685", ...]} pinning WHICH matches'),
    Param("cache_dir", str, None, help="raw open-data events cache root (else $SILLY_KICKS_CORPUS_CACHE_DIR)"),
)

#: Owner-reviewed (HFU plan R3-10, A+B approved 2026-10-30): the study is one unit by construction.
_STUDY_SERIAL_REASON = (
    "Optuna TPE: each trial reads every earlier trial, so trials cannot run in parallel without changing the "
    "search; with hpo_trials=0 (the campaign default) the unit is a no-op"
)


def _pairs(params) -> list[tuple[int, int]]:
    """The legacy corpus selection: ``--competitions`` pairs, else ``(--competition-id, --season-id)``."""
    if params["competitions"]:
        return [tuple(int(x) for x in p.split(":")) for p in params["competitions"].split(",")]  # type: ignore[misc]
    return [(params["competition_id"], params["season_id"])]


def _match_ids(params) -> list[str] | None:
    path = params["match_ids_json"]
    if path is None:
        return None
    return (json.loads(Path(path).read_text(encoding="utf-8")) or {}).get("statsbomb")


def _refs_work(key, ctx) -> dict:
    """The corpus and every match reference, as one JSON unit (the public-only guard runs FIRST)."""
    import scripts._sb_open_data as sbod

    sbod.assert_statsbomb_open_data_mode()
    pairs = _pairs(ctx.params)
    refs = sbod.list_open_data_refs(pairs, match_ids=_match_ids(ctx.params), max_matches=ctx.params["max_per_provider"])
    return {
        "pairs": [list(p) for p in pairs],
        "label": "statsbomb-open " + ", ".join(f"(comp {c}, season {s})" for c, s in pairs),
        "refs": [dataclasses.asdict(r) for r in refs],
    }


def _extract_refs(ctx):
    from scripts._sb_open_data import OpenDataRef

    return [OpenDataRef(**d) for d in ctx.upstream("refs")["refs"]]


def _load_match(ref, ctx):
    """Resume-before-download: the executor calls this only for a pending unit (ADR-052 D14)."""
    from scripts._driver_sources import resolve_cache_dir
    from scripts._sb_open_data import assert_statsbomb_open_data_mode, load_open_data_match

    assert_statsbomb_open_data_mode()  # pins the revision in THIS worker process (decision P6)
    return load_open_data_match(ref, cache_dir=resolve_cache_dir(ctx.params["cache_dir"]))


def _extract_work(item, ctx) -> pd.DataFrame:
    return onball_rows_for_match(item.actions, item.match_id)


def _extract_reduce(results, ctx) -> pd.DataFrame:
    from scripts._stages import combine_frames

    frames = list(results.values())  # sorted key order == the legacy sorted shard glob
    return combine_frames(frames, ids=("game_id",)) if frames else pd.DataFrame(columns=_SHARD_COLUMNS)


_POOLED: dict = {}


def _pooled(ctx) -> pd.DataFrame:
    """The pooled on-ball rows, loaded ONCE per worker; refuses an empty corpus. Keyed by the store AND the
    extract token (one token can name different corpora in different stores)."""
    tok = (str(getattr(ctx.store, "root", getattr(ctx.store, "prefix", ""))), ctx.tokens["extract"])
    if tok not in _POOLED:
        pooled = ctx.upstream("extract")
        if not len(pooled):
            raise RuntimeError("no on-ball rows collected from the corpus; nothing to fit")
        _POOLED.clear()
        _POOLED[tok] = pooled
    return _POOLED[tok]


def _study_work(key, ctx) -> dict:
    """The legacy HPO step: Optuna TPE over XGBoost params (None at ``hpo_trials=0`` or for the fallback family)."""
    p = ctx.params
    best = _hpo_params(_pooled(ctx), p["hpo_trials"]) if p["family"] == "xgboost" else None
    return {"best_params": best}


def _cv_units(ctx) -> list[str]:
    """``max(1, min(5, n_done))`` fold keys, from the extract SUMMARY (never its output; see `_cv_work`)."""
    n = int(ctx.summary("extract")["n_done"])
    return [int_key(k) for k in range(max(1, min(5, n)))]


def _cv_work(key, ctx) -> pd.DataFrame:
    """One fold's out-of-fold predictions as ``(row, pred)``; empty for a key past the real split; with fewer
    than 2 groups key ``0000`` carries the single in-sample fit."""
    pooled = _pooled(ctx)
    params, family = ctx.upstream("study")["best_params"], ctx.params["family"]
    k = int(key)
    folds = cv_folds(pooled)
    if folds is None:
        rows, preds = (
            (np.arange(len(pooled)), fold_predictions(pooled, None, params=params, family=family))
            if k == 0
            else ([], [])
        )
    elif k < len(folds):
        rows, preds = folds[k][1], fold_predictions(pooled, folds[k], params=params, family=family)
    else:
        rows, preds = [], []
    return pd.DataFrame({"row": np.asarray(rows, dtype=np.int64), "pred": np.asarray(preds, dtype=float)})


def _cv_reduce(results, ctx) -> dict:
    pooled = _pooled(ctx)
    if cv_folds(pooled) is None:
        oof = results[int_key(0)]["pred"].to_numpy(dtype=float)
    else:
        oof = np.full(len(pooled), np.nan)
        for part in results.values():
            oof[part["row"].to_numpy()] = part["pred"].to_numpy(dtype=float)
    return metrics_from_oof(pooled, oof)


def _fit_work(key, ctx) -> FileSet:
    """The legacy reduce tail: the final fit, ``metadata.json`` stamped with the commit, ``metrics.json`` (legacy
    key order) and the card, as ONE flat bundle (an artifact is flat: ``adopt.retrieve`` refuses a nested path;
    owner ruling 2026-10-30: the bundle ships ``metrics.json``)."""
    import tempfile

    from silly_kicks.xsuccess import XSuccessModel

    pooled = _pooled(ctx)
    p = ctx.params
    best = ctx.upstream("study")["best_params"]
    model = XSuccessModel().fit(pooled, family=p["family"], params=best)
    prov = ctx.prov
    with tempfile.TemporaryDirectory() as tmp:
        wdir = Path(tmp) / "weights"
        model.save(wdir)
        # Spec S5.5 / ADR-052 / ADR-056: the bundled metadata.json carries training provenance too (it is not in
        # SHA256SUMS, so this does not affect the model.json hash the fail-closed load verifies).
        meta_path = wdir / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["training_commit"] = prov["commit"]
        meta["run_tree_dirty"] = prov["dirty"]
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        files = {q.name: q.read_bytes() for q in sorted(wdir.iterdir()) if q.is_file()}
    summary = ctx.summary("extract")
    # Decision P4 (ADR-128): CORPUS-scoped counters, never pass-scoped ones (the legacy `res.manifest()` keys).
    pass_fields = {
        "generation": ctx.tokens["extract"],
        "n_attempted": summary["n_units"],
        "n_failed": summary["n_failed"],
        "n_counters_unrecorded": 0,
        "n_excluded": summary["n_excluded"],
    }
    out = {
        **ctx.upstream("cv"),
        "n_rows": len(pooled),
        "n_matches": int(pooled["game_id"].nunique()),
        "providers": ctx.upstream("refs")["label"],
        "hpo_trials": p["hpo_trials"],
        "hpo_best_params": best,
        "feature_set": model.feature_set,
        "training_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
        **pass_fields,
    }
    payload = json.dumps(out, indent=2, default=str)
    files["MODEL_CARD.md"] = render_model_card(out).encode("utf-8")
    files["metrics.json"] = payload.encode("utf-8")
    print(payload)
    return FileSet(files)


#: HFU plan R3-8 S3 item 12: the reviewed serial reasons of this pipeline's SERIAL stages, verbatim from the
#: plan's SERIAL annex (owner ruling 2026-10-30, "too small to pay"); `scripts/hf_jobs/serial_reasons.py`
#: holds the same texts and `tests/scripts/test_serial_reasons.py` holds the two equal.
_SERIAL_REASONS = {
    "refs": ("Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, under 1 s of CPU."),
    "extract": (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): 64 units, 20 s of CPU in all; a W=8 job is"
        " 9.8 s, mostly worker start-up."
    ),
    "cv": (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): 5 folds, 36 s of CPU in all; at W=5 the "
        "job is 13.4 s."
    ),
    "fit": ("Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one xgb fit, 9 s of CPU."),
}


PIPELINE = Pipeline(
    name="xsuccess",
    module="scripts.train_xsuccess",
    params=_PARAMS,
    stages=(
        Stage(
            name="refs",
            serial_reason=_SERIAL_REASONS["refs"],  # S3 item 12 (SERIAL annex)
            units=lambda ctx: ["only"],
            work=_refs_work,
            reduce=lambda r, ctx: r["only"],
            params=("competition_id", "season_id", "competitions", "max_per_provider"),
            # the match selection is declared by CONTENT, never by path (the 4.77.1 stale-shard rule)
            declare=lambda p: {"match_ids": _match_ids(p)},
            uses_dataset=True,
            dataset_files=lambda ctx: (),  # open data over HTTP: no dataset file
        ),
        ItemStage(
            name="extract",
            serial_reason=_SERIAL_REASONS["extract"],  # S3 item 12 (SERIAL annex)
            refs=_extract_refs,
            key=lambda ref: ref.key,
            load=_load_match,
            work=_extract_work,
            needs=("refs",),
            reduce=_extract_reduce,
            declare=lambda p: {"model": "XSuccessModel", "schema": _SHARD_SCHEMA_VERSION, "cols": list(_SHARD_COLUMNS)},
            cross_commit_version="xsuccess-shard-1",
            uses_dataset=True,
            max_failures=None,  # legacy for_each recorded failures and fit over the rest
        ),
        Stage(
            name="study",
            units=lambda ctx: ["only"],
            work=_study_work,
            needs=("extract",),
            reduce=lambda r, ctx: r["only"],
            params=("hpo_trials", "family"),
            serial_reason=_STUDY_SERIAL_REASON,
        ),
        Stage(
            name="cv",
            serial_reason=_SERIAL_REASONS["cv"],  # S3 item 12 (SERIAL annex)
            units=_cv_units,
            work=_cv_work,
            needs=("extract", "study"),
            reduce=_cv_reduce,
            params=("family",),
        ),
        Stage(
            name="fit",
            serial_reason=_SERIAL_REASONS["fit"],  # S3 item 12 (SERIAL annex)
            units=lambda ctx: ["only"],
            work=_fit_work,
            needs=("refs", "extract", "study", "cv"),
            reduce=lambda r, ctx: r["only"],
            params=("family", "hpo_trials"),
        ),
    ),
    output="fit",
)

if __name__ == "__main__":
    cli_main(PIPELINE)
