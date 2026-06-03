#!/usr/bin/env python
"""Train the xCrossAttempt (xCross) model (TF-17 weights run, PR-B).

Two match sources:
  --data-dir DIR     parquet dirs DIR/*/{frames,actions}.parquet (smoke / local corpus)
  --providers a,b,c  pining loader (skillcorner,idsse,gradientsports) for the maintainer run

Streams per match, caches features, runs ruthless HPO ONCE per candidate (public / full),
evaluates the common-public-held-out PAIRED data-effect comparison, computes FAIL-CLOSED
acceptance gates, and writes a pickle-free artifact ONLY if the gates pass. Quality numbers in
metrics.json are CV/protocol estimates, not the shipped all-data fit.

Mirror of scripts/train_xshot_occurrence.py. Requires: silly-kicks[train,xgboost]
(+ [kloppy] for --providers).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

_PUBLIC_PROVIDERS = {"skillcorner", "idsse"}


def _iter_matches_from_dir(data_dir: Path):
    for game_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        frames = pd.read_parquet(game_dir / "frames.parquet")
        actions = pd.read_parquet(game_dir / "actions.parquet")
        prov = str(frames["source_provider"].iloc[0]) if "source_provider" in frames.columns else "unknown"
        yield prov, game_dir.name, actions, frames, frames["team_id"].dropna().iloc[0]


def _iter_matches_from_pining(providers, max_per_provider):
    sys.path.insert(0, "scripts")
    from _loader_pining import load_matches

    yield from load_matches(providers=providers, max_per_provider=max_per_provider)


def _extract(source, horizon_seconds):
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL, prepare_xcross_training_data

    parts_x, parts_y, parts_g, parts_p = [], [], [], []
    for prov, mid, actions, frames, home in source:
        X, y, groups = prepare_xcross_training_data(
            frames,
            actions,
            home_team_id=home,
            horizon_seconds=horizon_seconds,
            wide_area_only=True,
            carrier_params=DEFAULT_CARRIER_PARAMS,  # 4.7.0 values; shared constant (anti-drift)
        )
        del frames
        if len(X):
            parts_x.append(X)
            parts_y.append(np.asarray(y, int))
            parts_g.append(np.asarray(groups))
            parts_p.append(np.array([prov] * len(X)))
            print(f"  {prov}/{mid}: {len(X)} rows, {int(np.asarray(y).sum())} positives")
    if not parts_x:
        raise SystemExit("No usable training data.")
    X = pd.concat(parts_x, ignore_index=True)[XCROSS_FEATURE_NAMES_FAITHFUL]
    return X, np.concatenate(parts_y), np.concatenate(parts_g), np.concatenate(parts_p)


def _hpo_once(X, y, groups, out_dir, tag, n_trials, *, negative_subsample=None, seed=42) -> dict:
    """Run ruthless HPO once for one candidate; return the frozen best-params dict."""
    from ruthless import Direction, FloatRange, InProcessBackend, OptunaConfig
    from ruthless.config.common import StoreConfig
    from ruthless.strategies.optuna_ import OptunaStrategy

    from silly_kicks.tracking._xcross_attempt_objective import XCrossAttemptObjective

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
        store=StoreConfig(kind="sqlite", path=str(out_dir / f"study_{tag}.db")),
    )
    result = OptunaStrategy(cfg, seed=42).run(obj, backend=InProcessBackend())
    return dict(result.best.candidate.params)


def _cv_metrics(X, y, groups, params, *, negative_subsample=None, seed=42) -> dict:
    """Label-stratified, match-grouped CV at FIXED params -> gate metrics on the TRUE balance."""
    import xgboost as xgb
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
    from sklearn.model_selection import StratifiedGroupKFold

    from silly_kicks.tracking._xcross_attempt import _pinned_params
    from silly_kicks.tracking._xshot_occurrence import subsample_negatives

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
        "n_usable_folds": len(lls),
    }


def _gates(m: dict) -> dict:
    pr = m["pr_auc"]
    br = m["brier"]
    return {
        "enough_usable_folds": m.get("n_usable_folds", 0) >= 2,
        "pr_auc_gt_base_rate": bool(pr == pr and pr > m["positive_rate"]),  # NaN-safe strict
        "brier_lt_base_rate_brier": bool(br == br and br < m["base_rate_brier"]),
        "log_loss_lt_uniform": m["log_loss"] < float(np.log(2)),
    }


def _paired_data_effect(X, y, groups, is_public, *, shared_params, negative_subsample=None, seed=42) -> dict:
    """Common-public-held-out PAIRED data-effect test at FIXED shared hyperparameters."""
    import xgboost as xgb
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedGroupKFold

    from silly_kicks.tracking._xcross_attempt import _pinned_params
    from silly_kicks.tracking._xshot_occurrence import subsample_negatives

    Xp, yp, gp = X[is_public], y[is_public], groups[is_public]
    n = max(2, min(5, len(np.unique(gp))))
    skf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=42)

    def _fit_score(Xtr, ytr, te_idx):
        if len(np.unique(ytr)) < 2 or len(np.unique(yp[te_idx])) < 2:
            return float("nan")
        if negative_subsample:  # TRAIN only; the public held-out fold (te_idx) is never subsampled
            Xtr, ytr, _ = subsample_negatives(Xtr, ytr, ytr, fraction=negative_subsample, seed=seed)
            if len(np.unique(ytr)) < 2:
                return float("nan")
        p_ = dict(_pinned_params(shared_params))
        p_["base_score"] = float(ytr.mean())
        c = xgb.XGBClassifier(**p_)
        c.fit(Xtr.to_numpy(float), ytr)
        pr = c.predict_proba(Xp.iloc[te_idx].to_numpy(float))[:, 1]
        return average_precision_score(yp[te_idx], pr)

    deltas = []
    for tr, te in skf.split(Xp, yp, gp):
        train_games = set(np.asarray(gp)[tr].tolist())
        full_mask = (~is_public) | np.isin(groups, list(train_games))  # GS + public-train only
        d_pub = _fit_score(Xp.iloc[tr], yp[tr], te)
        d_full = _fit_score(X[full_mask], y[full_mask], te)
        if not (np.isnan(d_pub) or np.isnan(d_full)):
            deltas.append(float(d_full - d_pub))
    K = len(deltas)
    n_pos = sum(1 for d in deltas if d > 0)
    ship_two = K >= 2 and n_pos >= K - 1 and (sum(deltas) / K) > 0.0
    return {
        "deltas": deltas,
        "K": K,
        "n_positive": n_pos,
        "ship_two": bool(ship_two),
        "paired_delta_is_data_effect_shared_params": True,
        "paired_hpo_nested": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir")
    src.add_argument("--providers")
    ap.add_argument("--output-dir", required=True)
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
    args = ap.parse_args()
    ns, seed = args.negative_subsample, args.seed

    out = Path(args.output_dir)
    art = out / "xcross_attempt_v1"
    cache = art / "_feature_cache"

    # --- Phase 1: stream + extract + cache ---
    if (cache / "features.parquet").exists():
        print(f"Loading cached features from {cache}")
        X = pd.read_parquet(cache / "features.parquet")
        y = np.load(cache / "labels.npy")
        groups = np.load(cache / "groups.npy", allow_pickle=True)
        providers = np.load(cache / "providers.npy", allow_pickle=True)
    else:
        if args.providers:
            source = _iter_matches_from_pining(args.providers.split(","), args.max_per_provider)
        else:
            source = _iter_matches_from_dir(Path(args.data_dir))
        t0 = time.time()
        X, y, groups, providers = _extract(source, args.horizon_seconds)
        print(f"Extracted {len(X)} rows ({int(y.sum())} positives) in {time.time() - t0:.0f}s")
        cache.mkdir(parents=True, exist_ok=True)
        X.to_parquet(cache / "features.parquet")
        np.save(cache / "labels.npy", y)
        np.save(cache / "groups.npy", groups)
        np.save(cache / "providers.npy", providers)

    groups = np.asarray(groups).astype(str)
    provset = {str(p) for p in providers.tolist()}
    is_public = np.isin(providers, list(_PUBLIC_PROVIDERS))
    two_candidate = is_public.any() and (~is_public).any() and "gradientsports" in provset

    # --- Phase 2/3: HPO once per candidate; ship decision ---
    candidates: dict = {}
    if two_candidate:
        params_public = _hpo_once(
            X[is_public],
            y[is_public],
            groups[is_public],
            out,
            "public",
            args.n_trials,
            negative_subsample=ns,
            seed=seed,
        )
        params_full = _hpo_once(X, y, groups, out, "full", args.n_trials, negative_subsample=ns, seed=seed)
        candidates["public"] = {
            "params": params_public,
            "metrics": _cv_metrics(
                X[is_public], y[is_public], groups[is_public], params_public, negative_subsample=ns, seed=seed
            ),
            "providers": sorted(set(providers[is_public].tolist())),
        }
        candidates["full"] = {
            "params": params_full,
            "metrics": _cv_metrics(X, y, groups, params_full, negative_subsample=ns, seed=seed),
            "providers": sorted(provset),
        }
        paired = _paired_data_effect(
            X, y, groups, is_public, shared_params=params_public, negative_subsample=ns, seed=seed
        )
        candidates["paired"] = paired
        shipped = "full" if paired["ship_two"] else "public"
        ship_mask = np.ones(len(X), bool) if shipped == "full" else is_public
    else:
        params_all = _hpo_once(X, y, groups, out, "single", args.n_trials, negative_subsample=ns, seed=seed)
        if provset <= _PUBLIC_PROVIDERS:
            shipped = "public"
        elif "gradientsports" in provset:
            shipped = "full"
        else:
            shipped = "default"
        candidates[shipped] = {
            "params": params_all,
            "metrics": _cv_metrics(X, y, groups, params_all, negative_subsample=ns, seed=seed),
            "providers": sorted(provset),
        }
        ship_mask = np.ones(len(X), bool)

    # --- Fail-closed acceptance gates on the SHIPPED candidate ---
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
        sys.exit(1)

    # --- Final fit on ALL the shipped candidate's games + save ---
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
    from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel
    from silly_kicks.tracking._xshot_occurrence import subsample_negatives

    Xfit, yfit, _ = (
        subsample_negatives(X[ship_mask], y[ship_mask], y[ship_mask], fraction=ns, seed=seed)
        if ns
        else (X[ship_mask], y[ship_mask], None)
    )
    model = XCrossAttemptModel(params=candidates[shipped]["params"])
    model.shipped_variant = shipped
    model.provider_list = candidates[shipped]["providers"]
    model.fit(Xfit, pd.Series(yfit), carrier_params=DEFAULT_CARRIER_PARAMS, horizon_seconds=args.horizon_seconds)
    model.save(art)
    reloaded = XCrossAttemptModel.load(art)
    np.testing.assert_allclose(
        model.predict_proba(X[ship_mask].head(50)), reloaded.predict_proba(X[ship_mask].head(50)), rtol=0, atol=0
    )

    metrics = {
        "shipped_variant": shipped,
        "n_rows": len(X),
        "n_positive": int(y.sum()),
        "providers": sorted(provset),
        "candidates": candidates,
        "acceptance": acceptance,
        "estimates_are_cv_not_shipped_fit": True,
        "artifact_size_bytes": sum(f.stat().st_size for f in art.glob("*") if f.is_file()),
    }
    json.dump(metrics, open(art / "metrics.json", "w"), indent=2)
    print(f"Wrote artifact + metrics to {art}")


if __name__ == "__main__":
    main()
