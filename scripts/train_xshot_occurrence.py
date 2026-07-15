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

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

# Feature-cache schema token (Task 11 / spec 3.2). The load-bearing invalidation is the
# schema_version bump inside _cache.cache_is_valid -- a pre-Task-11 cache has no cache_meta.json
# and MISSES. The corpus-mismatch fingerprint is intentionally NOT wired (it would need a live
# manifest fetch on the cache-hit path, before providers/match_ids are even loaded); a constant
# token keeps write + check agreeing while the schema check does the real work.
_CACHE_FINGERPRINT = "schema-v2"


def _iter_matches_from_dir(data_dir: Path):
    for game_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        frames = pd.read_parquet(game_dir / "frames.parquet")
        shots = pd.read_parquet(game_dir / "shots.parquet")
        prov = str(frames["source_provider"].iloc[0]) if "source_provider" in frames.columns else "unknown"
        yield prov, game_dir.name, shots, frames, frames["team_id"].dropna().iloc[0]


def _iter_matches_from_pining(providers, max_per_provider, match_ids=None):
    sys.path.insert(0, "scripts")
    from _loader_pining import load_matches

    yield from load_matches(providers=providers, match_ids=match_ids, max_per_provider=max_per_provider)


def _extract(source, horizon_seconds) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL, prepare_xshot_training_data

    parts_x, parts_y, parts_g, parts_p, parts_m = [], [], [], [], []
    for prov, mid, actions_or_shots, frames, home in source:
        X, y, groups = prepare_xshot_training_data(
            frames,
            actions_or_shots,
            home_team_id=home,
            horizon_seconds=horizon_seconds,
            attacking_third_only=True,
            carrier_params=DEFAULT_CARRIER_PARAMS,  # 4.7.0 values; shared constant (anti-drift)
        )
        del frames
        if len(X):
            parts_x.append(X)
            parts_y.append(np.asarray(y, int))
            parts_g.append(np.asarray(groups))
            parts_p.append(np.array([prov] * len(X)))
            parts_m.append(np.array([str(mid)] * len(X)))  # per-row pining match_id (visibility key)
            print(f"  {prov}/{mid}: {len(X)} rows, {int(np.asarray(y).sum())} positives")
    if not parts_x:
        raise SystemExit("No usable training data.")
    X = pd.concat(parts_x, ignore_index=True)[XSHOT_FEATURE_NAMES_FAITHFUL]
    return (
        X,
        np.concatenate(parts_y),
        np.concatenate(parts_g),
        np.concatenate(parts_p),
        np.concatenate(parts_m),
    )


def _hpo_once(X, y, groups, out_dir, tag, n_trials, *, negative_subsample=None, seed=42) -> dict:
    """Run ruthless HPO once for one candidate; return the frozen best-params dict.

    ``negative_subsample`` thins negatives in TRAIN folds only (never eval) inside the objective.
    """
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
    return dict(result.best.candidate.params)


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
    from sklearn.model_selection import StratifiedGroupKFold

    Xp, yp, gp = X[is_public], y[is_public], groups[is_public]
    k = max(2, min(5, len(np.unique(gp))))
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
    out = {name: {"nested": [], "shared_params": []} for name in candidates}

    for fold, (_tr, te) in enumerate(skf.split(Xp, yp, gp)):
        te_games = set(np.asarray(gp)[te].tolist())
        trainable = ~(is_public & np.isin(groups, list(te_games)))  # drop fold-k public games from ALL arms
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


def main(argv=None) -> None:
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
    args = ap.parse_args(argv)
    ns, seed = args.negative_subsample, args.seed

    out = Path(args.output_dir)
    art = out / "xshot_occurrence_v1"
    cache = art / "_feature_cache"
    sys.path.insert(0, "scripts")
    from _cache import cache_is_valid, write_cache_meta

    # --- Phase 1: stream + extract + cache ---
    # Task 11 cache-schema guard: a pre-Task-11 cache (no cache_meta.json, no match_ids.npy) MISSES,
    # so the DGX-populated caches that predate the visibility taxonomy are never silently reused --
    # a stale cache would re-introduce the retired provider-name arm split (spec 3.2).
    # NOTE: the fingerprint is a constant schema token, NOT a per-corpus hash -- it invalidates
    # pre-Task-11 caches but does NOT detect corpus DRIFT within the same schema. So a cache built
    # from corpus A and reused here under the same --output-dir with a different --match-ids-json
    # would be silently reused with stale rows. USE A FRESH --output-dir PER CORPUS (the DGX runbook
    # does). ADR-038 records this deviation from the spec-registered live fingerprint.
    if cache_is_valid(cache, fingerprint=_CACHE_FINGERPRINT) and (cache / "match_ids.npy").exists():
        print(f"Loading cached features from {cache}")
        X = pd.read_parquet(cache / "features.parquet")
        y = np.load(cache / "labels.npy")
        groups = np.load(cache / "groups.npy", allow_pickle=True)
        providers = np.load(cache / "providers.npy", allow_pickle=True)
        match_ids = np.load(cache / "match_ids.npy", allow_pickle=True)
    else:
        if args.providers:
            allowlist = json.load(open(args.match_ids_json)) if args.match_ids_json else None
            source = _iter_matches_from_pining(args.providers.split(","), args.max_per_provider, allowlist)
        else:
            source = _iter_matches_from_dir(Path(args.data_dir))
        t0 = time.time()
        X, y, groups, providers, match_ids = _extract(source, args.horizon_seconds)
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
        write_cache_meta(cache, fingerprint=_CACHE_FINGERPRINT)

    # game_id dtype is provider-asymmetric (kloppy str vs GS int) -> normalize cross-provider
    # groups to str so np.unique / StratifiedGroupKFold can sort them (the model never uses game_id).
    groups = np.asarray(groups).astype(str)
    provset = {str(p) for p in providers.tolist()}
    sys.path.insert(0, "scripts")
    from _corpus import artifact_label, assert_public_corpus, is_public_row
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

    # --- Phase 2/3: nested-HPO paired test (mix) or single-candidate (else); ship decision ---
    candidates: dict = {}
    if run_paired:
        from _paired import fixed_sequence_ship

        is_sc_private = (providers == "skillcorner") & ~is_public  # owner-tier SkillCorner rows
        cand_masks = {
            "public": is_public,
            "sc_extended": is_public | is_sc_private,
            "full": np.ones(len(X), bool),
        }
        paired = _paired_data_effect(
            X,
            y,
            groups,
            is_public,
            match_ids,
            candidates=cand_masks,
            n_trials=args.n_trials,
            out_dir=out,
            negative_subsample=ns,
            seed=seed,
        )
        full_vs_sc = [f - s for f, s in zip(paired["full"]["nested"], paired["sc_extended"]["nested"], strict=True)]
        shipped, why = fixed_sequence_ship(
            sc_extended=paired["sc_extended"]["nested"],
            full=paired["full"]["nested"],
            full_vs_sc=full_vs_sc,
        )
        print(f"Fixed-sequence verdict: ship {shipped} -- {why}")
        ship_mask = cand_masks[shipped]
        # The paired test DECIDED the corpus; the shipped model is tuned once on ALL of it (the
        # per-fold params never leave the paired comparison -- they were held-out estimates).
        shipped_params = _hpo_once(
            X[ship_mask], y[ship_mask], groups[ship_mask], out, shipped, args.n_trials, negative_subsample=ns, seed=seed
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
        params_all = _hpo_once(X, y, groups, out, "single", args.n_trials, negative_subsample=ns, seed=seed)
        ship_mask = np.ones(len(X), bool)
        # Label from the SHIP MASK's visibility composition (spec 3.2), never the provider name: a
        # corpus with ANY restricted row can NEVER be labelled "public" (is_public[ship_mask].all()).
        ship_provs = set(providers[ship_mask].tolist())
        shipped = artifact_label(providers=ship_provs, all_public=bool(is_public[ship_mask].all()))
        candidates[shipped] = {
            "params": params_all,
            "metrics": _cv_metrics(X, y, groups, params_all, negative_subsample=ns, seed=seed),
            "providers": sorted(provset),
        }

    # --- Fail-closed acceptance gates on the SHIPPED candidate (N3) ---
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
    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel, subsample_negatives

    # The shipped fit has no held-out fold (it is all training data), so subsampling here is
    # safe + consistent with the gate-eval CV's train folds.
    Xfit, yfit, _ = (
        subsample_negatives(X[ship_mask], y[ship_mask], y[ship_mask], fraction=ns, seed=seed)
        if ns
        else (
            X[ship_mask],
            y[ship_mask],
            None,
        )
    )
    model = XShotOccurrenceModel(params=candidates[shipped]["params"])
    model.shipped_variant = shipped
    model.provider_list = candidates[shipped]["providers"]
    model.fit(
        Xfit,
        pd.Series(yfit),
        carrier_params=DEFAULT_CARRIER_PARAMS,
        horizon_seconds=args.horizon_seconds,
    )
    model.save(art)
    reloaded = XShotOccurrenceModel.load(art)
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
        "estimates_are_cv_not_shipped_fit": True,  # N7: quality numbers are CV/protocol estimates
        "artifact_size_bytes": sum(f.stat().st_size for f in art.glob("*") if f.is_file()),
    }
    json.dump(metrics, open(art / "metrics.json", "w"), indent=2)
    print(f"Wrote artifact + metrics to {art}")


if __name__ == "__main__":
    main()
