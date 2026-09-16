"""Trainer: bundle the default XSuccessModel weights (TF-61).

Fits ``silly_kicks.xsuccess.XSuccessModel`` (END-BLIND, calibrated XGBoost) on the PUBLIC StatsBomb
open-data corpus and writes the pickle-free booster-JSON + metadata + SHA256SUMS artifact via
``model.save(...)``, plus an out-of-fold ``metrics.json`` stamping ``training_commit`` (ADR-052 /
ADR-011 discipline, mirroring ``train_pass_completion.py`` and the other weight trainers). Inference
imports no sklearn; xgboost + sklearn are used only during the fit/HPO.

The expensive per-match corpus load is sharded with ``for_each`` (ADR-052): one shard per match
holding that match's real on-ball rows (the fit corpus). The optional Optuna HPO (``XSuccessObjective``),
the pooled fit + calibration, the GroupKFold-by-match out-of-fold metrics, and ``save`` happen in the
reduce, off the network.

``--out`` is a run directory OUTSIDE the repo (shards + fitted ``weights/`` + ``metrics.json``); copy
``<out>/weights`` into ``silly_kicks/xsuccess/weights/`` for the bundled-weights commit (Commit 2).

Usage (on the box, scripts/ on sys.path):
  python scripts/train_xsuccess.py --out <DIR> [--competition-id 43 --season-id 106] [--hpo-trials N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig

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
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    from silly_kicks.xsuccess import XSuccessModel

    success = spadlconfig.result_id["success"]
    y = (pooled["result_id"].to_numpy() == success).astype(int)
    groups = pooled["game_id"].to_numpy().astype(str)
    n_splits = min(5, len(np.unique(groups)))
    oof = np.full(len(y), np.nan)
    if n_splits < 2:
        oof = XSuccessModel().fit(pooled, family=family, params=params).predict_success(pooled)
    else:
        for tr, te in GroupKFold(n_splits=n_splits).split(pooled, y, groups):
            m = XSuccessModel().fit(pooled.iloc[tr], family=family, params=params)
            oof[te] = m.predict_success(pooled.iloc[te])

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="run dir OUTSIDE the repo (shards + weights/ + metrics.json)")
    ap.add_argument("--competition-id", type=int, default=43, help="open-data competition (default 43 = World Cup)")
    ap.add_argument("--season-id", type=int, default=106, help="open-data season (default 106 = 2022)")
    ap.add_argument(
        "--competitions",
        default=None,
        help="comma-sep comp:season pairs for the FULL redistributable open-data corpus (spec S13 -- "
        "NOT WC2022-only; more data for rare-type calibration); overrides --competition-id/--season-id. "
        "e.g. '43:106,55:43'. The convertible/public set is enumerated at the Task-12 run + recorded in the manifest.",
    )
    ap.add_argument("--max-per-provider", type=int, default=None, help="cap the number of matches")
    ap.add_argument("--hpo-trials", type=int, default=0, help="Optuna HPO trials (0 = default params)")
    ap.add_argument(
        "--family",
        choices=("xgboost", "per_type_logistic"),
        default="xgboost",
        help="model family: calibrated XGBoost (default) or the per-type-logistic ADR-009 fallback "
        "(select it if XGBoost fails the per-type calibration gate; spec S5.4).",
    )
    ap.add_argument(
        "--match-ids-json", default=None, help='JSON {"statsbomb": ["3869685", ...]} pinning WHICH matches.'
    )
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; artifact marked dirty)")
    args = ap.parse_args()

    # Clean-tree guard FIRST, before any corpus work (ADR-052): bundled weights whose provenance is
    # unknown cannot be reproduced or audited.
    from scripts._provenance import git_provenance, require_clean_tree

    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)

    from scripts._driver import for_each
    from scripts._sb_open_data import load_open_data_matches
    from silly_kicks.xsuccess import XSuccessModel
    from silly_kicks.xsuccess._features import FEATURE_NAMES

    match_ids = json.loads(Path(args.match_ids_json).read_text(encoding="utf-8")) if args.match_ids_json else None
    dest = Path(args.out)
    import itertools

    pairs = (
        [tuple(int(x) for x in p.split(":")) for p in args.competitions.split(",")]
        if args.competitions
        else [(args.competition_id, args.season_id)]
    )
    corpus_label = "statsbomb-open " + ", ".join(f"(comp {c}, season {s})" for c, s in pairs)
    ids = (match_ids or {}).get("statsbomb")
    matches_iter = itertools.chain.from_iterable(
        load_open_data_matches(competition_id=c, season_id=s, match_ids=ids, max_matches=args.max_per_provider)
        for c, s in pairs
    )

    def _work(item):
        _provider, match_id, actions, _frames, _home = item
        return onball_rows_for_match(actions, match_id)

    res = for_each(
        matches_iter,
        key=lambda item: (str(item[0]), str(item[1])),
        work=_work,
        shard_root=dest / "shards",
        token_inputs={
            "model": "XSuccessModel",
            "schema": _SHARD_SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "competitions": [list(p) for p in pairs],
        },
        label="match",
    )

    shard_files = sorted(res.shard_dir.glob("*.parquet"))
    pooled = (
        pd.concat([pd.read_parquet(s) for s in shard_files], ignore_index=True)
        if shard_files
        else pd.DataFrame(columns=_SHARD_COLUMNS)
    )
    if not len(pooled):
        raise SystemExit("no on-ball rows collected from the corpus; nothing to fit")

    best_params = _hpo_params(pooled, args.hpo_trials) if args.family == "xgboost" else None
    metrics = cross_val_metrics(pooled, params=best_params, family=args.family)
    model = XSuccessModel().fit(pooled, family=args.family, params=best_params)
    wdir = dest / "weights"
    model.save(wdir)
    # Spec S5.5 / ADR-052 / ADR-056: the bundled metadata.json carries training provenance too.
    # `save()` writes the model envelope; the trainer stamps the commit it alone knows (metadata.json
    # is not in SHA256SUMS, so this does not affect the model.json hash the fail-closed load verifies).
    _meta_path = wdir / "metadata.json"
    _meta = json.loads(_meta_path.read_text(encoding="utf-8"))
    _meta["training_commit"] = prov["commit"]
    _meta["run_tree_dirty"] = prov["dirty"]
    _meta_path.write_text(json.dumps(_meta, indent=2) + "\n", encoding="utf-8")

    out = {
        **metrics,
        "n_rows": len(pooled),
        "n_matches": int(pooled["game_id"].nunique()),
        "providers": corpus_label,
        "hpo_trials": args.hpo_trials,
        "hpo_best_params": best_params,
        "feature_set": model.feature_set,
        "training_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
        **res.manifest(),
    }
    (dest / "metrics.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    (wdir / "MODEL_CARD.md").write_text(render_model_card(out), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))
    print(f"bundled weights + MODEL_CARD.md -> {wdir}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
