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


def _new_probe_cohort() -> dict:
    """One TF-19 probe cohort's capture state (M5): bounded frames/actions copies + provenance."""
    return {"frames": [], "actions": [], "home": None, "matches": [], "match_groups": {}}


def _extract(
    source,
    horizon_seconds,
    *,
    probe_keep=2,
    probe_providers=("gradientsports",),
    probe_comparison_providers=("skillcorner",),
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, tuple[dict, dict]]:
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL, prepare_xcross_training_data

    parts_x, parts_y, parts_g, parts_p = [], [], [], []
    # TF-19 substitution-probe samples (M1/M3/M5): the GATED cohort is provider-CONTROLLED
    # (--probe-providers; default the gradientsports gated cohort); a second reported-not-gated
    # comparison cohort (--probe-comparison-providers) persists to _probe_sample_comparison/.
    probe, comparison = _new_probe_cohort(), _new_probe_cohort()
    for prov, mid, actions, frames, home in source:
        X, y, groups = prepare_xcross_training_data(
            frames,
            actions,
            home_team_id=home,
            horizon_seconds=horizon_seconds,
            wide_area_only=True,
            carrier_params=DEFAULT_CARRIER_PARAMS,  # 4.7.0 values; shared constant (anti-drift)
        )
        if len(X):
            parts_x.append(X)
            parts_y.append(np.asarray(y, int))
            parts_g.append(np.asarray(groups))
            parts_p.append(np.array([prov] * len(X)))
            cohort = probe if prov in probe_providers else comparison if prov in probe_comparison_providers else None
            if cohort is not None and len(cohort["frames"]) < probe_keep:  # M3: capture a COPY before del frames
                # N3 (memory): keeps up to `probe_keep` matches' frames+actions resident per cohort for
                # the whole loop (deliberate, bounded -- vs the original's immediate del). Fine at tracking
                # scale on the box; probe_keep caps it. The per-match `del frames` still frees all others.
                cohort["frames"].append(frames.copy())
                cohort["actions"].append(actions.copy())
                cohort["home"] = home
                cohort["matches"].append([prov, str(mid)])
                # groups == game_id per row (prepare_xcross_training_data contract), recorded so the
                # gate can compute per-match training-fold membership (M6) + filter the probe frames.
                cohort["match_groups"][str(mid)] = sorted({str(g) for g in np.asarray(groups).tolist()})
            print(f"  {prov}/{mid}: {len(X)} rows, {int(np.asarray(y).sum())} positives")
        del frames
    if not parts_x:
        raise SystemExit("No usable training data.")
    X = pd.concat(parts_x, ignore_index=True)[XCROSS_FEATURE_NAMES_FAITHFUL]
    return (
        X,
        np.concatenate(parts_y),
        np.concatenate(parts_g),
        np.concatenate(parts_p),
        (probe, comparison),
    )


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
    if result.best is None:
        raise RuntimeError("HPO produced no best candidate")
    return dict(result.best.candidate.params)


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
    ap.add_argument(
        "--probe-providers",
        default="gradientsports",
        help="Comma list of providers eligible for the TF-19 substitution-probe capture (M5: the "
        "GATED cohort, persisted to _probe_sample/). Default: the gated gradientsports cohort.",
    )
    ap.add_argument(
        "--probe-comparison-providers",
        default="skillcorner",
        help="Comma list captured to _probe_sample_comparison/ -- the reported-not-gated "
        "same-population comparison leg (M5).",
    )
    args = ap.parse_args()
    ns, seed = args.negative_subsample, args.seed
    probe_provs = [p for p in args.probe_providers.split(",") if p]
    comparison_provs = [p for p in args.probe_comparison_providers.split(",") if p]
    if set(probe_provs) & set(comparison_provs):
        raise SystemExit("--probe-providers and --probe-comparison-providers must be disjoint.")

    out = Path(args.output_dir)
    art = out / "xcross_attempt_v1"
    cache = art / "_feature_cache"

    # --- Phase 1: stream + extract + cache ---
    # M1: bound on BOTH branches (cache-hit never calls _extract).
    probe_bundle = (_new_probe_cohort(), _new_probe_cohort())
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
        X, y, groups, providers, probe_bundle = _extract(
            source,
            args.horizon_seconds,
            probe_providers=tuple(probe_provs),
            probe_comparison_providers=tuple(comparison_provs),
        )
        print(f"Extracted {len(X)} rows ({int(y.sum())} positives) in {time.time() - t0:.0f}s")
        cache.mkdir(parents=True, exist_ok=True)
        X.to_parquet(cache / "features.parquet")
        np.save(cache / "labels.npy", y)
        np.save(cache / "groups.npy", groups)
        np.save(cache / "providers.npy", providers)
        probe_cohort, comparison_cohort = probe_bundle  # persist the TF-19 probe samples (fresh-extract only)
        _write_probe_sample(cache.parent / "_probe_sample", probe_cohort, probe_provs)
        _write_probe_sample(cache.parent / "_probe_sample_comparison", comparison_cohort, comparison_provs)

    groups = np.asarray(groups).astype(str)
    provset = {str(p) for p in providers.tolist()}
    is_public = np.isin(providers, list(_PUBLIC_PROVIDERS))
    two_candidate = is_public.any() and (~is_public).any() and "gradientsports" in provset

    # --- score_differential range probe (B6): guard the clean-4.13.0-GS rebuild prereq ---
    sd = X["score_differential"].to_numpy(dtype=float)
    sd_fin = sd[np.isfinite(sd)]
    sd_probe = {
        "coverage": float(np.isfinite(sd).mean()),
        "min": float(sd_fin.min()) if sd_fin.size else float("nan"),
        "max": float(sd_fin.max()) if sd_fin.size else float("nan"),
        "abs_ge_12_count": int((np.abs(sd_fin) >= 12).sum()),
    }
    if sd_probe["abs_ge_12_count"] > 0:  # HARD-FAIL: phantom-owngoal signature (the old +-18 bug)
        raise SystemExit(
            f"score_differential range probe FAILED (impossible |sd|>=12): {sd_probe}. "
            "Rebuild the feature cache on clean 4.13.0 GS events."
        )
    if sd_fin.size and np.abs(sd_fin).max() > 6:  # SOFT-WARN: a real rout is legitimate
        print(f"WARN score_differential |max|>6 (legit blowout possible): {sd_probe}", file=sys.stderr)

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

    # --- Headline GK validations on the SHIPPED candidate (PR-B, into metrics.json) ---
    from silly_kicks.tracking import _xcross_eval as ev

    shipped_params = candidates[shipped]["params"]
    gk_ablation = ev.gk_block_ablation(
        X[ship_mask], y[ship_mask], groups[ship_mask], shipped_params, seed=seed, negative_subsample=ns
    )
    perm_imp = ev.permutation_importance_report(
        X[ship_mask], y[ship_mask], groups[ship_mask], shipped_params, n_repeats=10, seed=seed
    )
    # M1: the probe sample is REQUIRED -- refuse to ship a spurious tf19_ready=False on a missing sample.
    ps = cache.parent / "_probe_sample"
    if not (ps / "frames.parquet").exists():
        raise SystemExit(
            "Feature cache present but _probe_sample/ absent -> cannot run the TF-19 substitution "
            "probe (the headline deliverable). Delete the feature cache and re-extract (box Task 9), "
            "or restore the probe sample. Refusing to ship a spurious tf19_ready=False."
        )
    pf = pd.read_parquet(ps / "frames.parquet")
    pa = pd.read_parquet(ps / "actions.parquet")
    probe_meta = json.load(open(ps / "meta.json"))
    phome = probe_meta["home_team_id"]

    # --- M6: held-out gated statistic. Record each probe match's training-fold membership
    # (its game ids vs the SHIPPED candidate's training games -- the final fit uses ALL of
    # them) into meta.json, then gate: when the paired test admitted the probe provider to
    # training, the probe runs on held-out matches ONLY (fail-loud when none exist).
    # Pre-plan meta.json files lack the probe fields -> .get defaults, never a KeyError.
    probe_matches = probe_meta.get("probe_matches", [])
    match_groups = probe_meta.get("match_groups", {})
    train_groups = set(groups[ship_mask].tolist())  # groups already astype(str) above
    in_training = {mid: bool(set(g) & train_groups) for mid, g in match_groups.items()}
    probe_meta["in_training_folds"] = in_training
    json.dump(probe_meta, open(ps / "meta.json", "w"), indent=2)
    shipped_providers = set(candidates[shipped].get("providers", []))
    probe_provs_seen = {prov for prov, _mid in probe_matches}
    # Conservative on missing provenance: a two-candidate run cannot verify a pre-plan sample.
    admitted = bool(two_candidate and (not probe_provs_seen or probe_provs_seen & shipped_providers))
    gated_matches = _gated_probe_matches(probe_meta, admitted)
    if admitted:  # the GATED statistic runs on held-out matches only
        held_groups = {g for _prov, mid in gated_matches for g in match_groups.get(str(mid), [])}
        pf = pf[pf["game_id"].astype(str).isin(held_groups)]
        pa = pa[pa["game_id"].astype(str).isin(held_groups)]
        if pf.empty:
            raise SystemExit(
                "Held-out probe matches resolved ZERO frames -- the probe sample is inconsistent "
                "with meta.json match_groups. Delete the feature cache + probe sample and re-extract."
            )
    elif probe_matches and all(in_training.get(str(m[1]), False) for m in probe_matches):
        print(
            "NOTE: every probe match sits in the shipped training corpus (single-candidate path) -- "
            "the substitution-probe statistic below is NOT held-out.",
            file=sys.stderr,
        )
    probe = ev.gk_substitution_probe(model, pf, actions=pa, home_team_id=phome, n_frames=200, seed=seed)

    metrics.update(
        {
            "gk_block_ablation": gk_ablation,
            "gk_substitution_probe": probe,
            "permutation_importance": perm_imp,
            "score_differential_range_probe": sd_probe,
            "probe_sample_matches": probe_matches,
            "probe_sample_in_training_folds": in_training,
            "probe_gated_on_held_out": bool(admitted),
            "tf19_ready": probe.get("tf19_ready", False),
        }
    )
    if not probe.get("tf19_ready", False):
        print(
            f"NOTE: tf19_ready=False ({probe.get('tf19_reason')}) -- surface ships, but flagged "
            "NOT TF-19-ready (loud, not silent).",
            file=sys.stderr,
        )

    json.dump(metrics, open(art / "metrics.json", "w"), indent=2)
    print(f"Wrote artifact + metrics to {art}")


if __name__ == "__main__":
    main()
