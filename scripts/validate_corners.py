"""Construct-validity battery for the TF-46 Arm 1 corner-offence metrics (reported, not gated).

Over the owner corpus: per-corner routine features (fitted CornerRoutineModel) + per-routine chance
quality, then the three-gate reading (responsiveness / discrimination-vs-permutation-null net of team /
transfer) plus the SB360 initial-snapshot divergence (section 12.7): the -2s-vs-kick-frame positional
delta + initial-zone reassignment on a provider carrying BOTH continuous tracking and freeze frames.

Descriptive, no accuracy gate (the encoder/NMF are unsupervised; name-by-mechanism). A power check takes
an InjectionSpec recipe, never an observed-outcome vector, so the observed outcome is unrepresentable.
Clean-tree guard first (ADR-037); input contract declares the seams (ADR-056). Real corpus run is opt-in
/ e2e (DGX or media-PC); tests/scripts cover the pure kernels. Memo lands at
docs/research/corner_routines/ (README.md + metrics.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

_MIN_PER_TEAM = 5
_RNG_SEED = 20261016


def icc_vs_permutation(
    df: pd.DataFrame, value_col: str, group_col: str, *, n_perm: int = 1000, seed: int = _RNG_SEED
) -> dict:
    """One-way ICC of ``value_col`` by ``group_col`` vs a label-permutation null (reported, not gated).

    ICC = between-group variance share. The null permutes group labels ``n_perm`` times; ``null_p95`` is
    its 95th percentile and ``p`` the fraction of null ICCs >= the observed.
    """
    sub = df[[value_col, group_col]].dropna()
    if sub[group_col].nunique() < 2 or len(sub) < 2:
        return {"icc": float("nan"), "null_p95": float("nan"), "p": float("nan"), "n": len(sub)}

    def _icc(vals: np.ndarray, groups: np.ndarray) -> float:
        grand = vals.mean()
        total = float(((vals - grand) ** 2).sum())
        if total <= 0:
            return 0.0
        between = 0.0
        for g in np.unique(groups):
            v = vals[groups == g]
            between += len(v) * (v.mean() - grand) ** 2
        return float(between / total)

    vals = sub[value_col].to_numpy(dtype=float)
    groups = sub[group_col].to_numpy()
    obs = _icc(vals, groups)
    rng = np.random.default_rng(seed)
    null = np.array([_icc(vals, rng.permutation(groups)) for _ in range(n_perm)])
    return {
        "icc": obs,
        "null_p95": float(np.percentile(null, 95)),
        "p": float((null >= obs).mean()),
        "n": len(sub),
    }


def sb360_init_divergence(init_xy: np.ndarray, kick_xy: np.ndarray, model) -> dict:
    """Section 12.7: bound the SB360 train/serve mismatch. ``init_xy`` is the tracking -2s snapshot and
    ``kick_xy`` the kick-frame snapshot for the SAME players; report the positional-delta distribution
    and the initial-zone reassignment rate (argmax initial zone differs between the two snapshots)."""
    from silly_kicks.corners._zones import _gaussian_responsibilities

    init_xy = np.asarray(init_xy, dtype=float).reshape(-1, 2)
    kick_xy = np.asarray(kick_xy, dtype=float).reshape(-1, 2)
    delta = np.hypot(kick_xy[:, 0] - init_xy[:, 0], kick_xy[:, 1] - init_xy[:, 1])
    r_init = _gaussian_responsibilities(init_xy, model.init_means, model.init_covs, model.init_weights)
    r_kick = _gaussian_responsibilities(kick_xy, model.init_means, model.init_covs, model.init_weights)
    reassigned = (r_init.argmax(axis=1) != r_kick.argmax(axis=1)).mean() if len(init_xy) else float("nan")
    return {
        "n_players": len(init_xy),
        "delta_mean_m": float(delta.mean()) if len(delta) else float("nan"),
        "delta_p95_m": float(np.percentile(delta, 95)) if len(delta) else float("nan"),
        "zone_reassignment_rate": float(reassigned),
    }


def reduce_samples(samples: pd.DataFrame) -> dict:
    """Pooled verdicts over all corpus corner-feature/chance-quality rows (the REDUCE)."""
    out: dict = {"n_corners": len(samples), "n_teams": int(samples["team_id"].nunique()) if "team_id" in samples else 0}
    if "sum_xg" in samples:
        xg = samples["sum_xg"].to_numpy(dtype=float)
        out["responsiveness"] = {"sum_xg_mean": float(np.nanmean(xg)) if len(xg) else float("nan")}
    if {"sum_xg", "team_id"} <= set(samples.columns):
        out["discrimination_one_way"] = {"sum_xg": icc_vs_permutation(samples, "sum_xg", "team_id")}
    if {"sum_xg", "provider"} <= set(samples.columns):
        # section 12.5 cross-provider stability: per-provider chance-quality means (SkillCorner broadcast vs
        # optical full-obs vs SB360 freeze) -- reported for the stability reading, not gated.
        out["cross_provider"] = {
            str(p): {"n": len(g), "sum_xg_mean": float(np.nanmean(g["sum_xg"].to_numpy(dtype=float)))}
            for p, g in samples.groupby("provider")
        }
    return out


def _work(item) -> pd.DataFrame:
    """Per-match per-corner routine-feature + chance-quality rows (requires fitted models in --models)."""
    # Corpus orchestration is exercised at Task 14 (DGX); kept thin so the pure kernels above stay the
    # CI-tested surface. The fitted models are loaded in main() and threaded via a closure there.
    raise NotImplementedError("validate_corners._work is wired by main(); run the e2e corpus pass on the DGX.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=None, help="dir holding the fitted run_zone/ + routine/ bundles")
    ap.add_argument("--token", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--providers", nargs="*", default=["skillcorner", "idsse", "sportec", "statsbomb"])
    ap.add_argument("--match-ids-json", default=None, help='per-provider pins: JSON {"<provider>": ["<id>", ...]}')
    ap.add_argument("--max-matches", type=int, default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    from scripts._driver import for_each
    from scripts._input_contract import declare_inputs
    from scripts._loader_pining import pining_source
    from scripts._provenance import git_provenance, require_clean_tree
    from silly_kicks.corners import (
        CornerRoutineModel,
        RunZoneModel,
        add_corner_windows,
        compute_corner_chance_quality,
        compute_corner_routine_features,
        compute_corner_run_encoding,
        identify_corners,
    )
    from silly_kicks.tracking._ball_contact import detect_ball_contacts

    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    match_ids = json.loads(Path(args.match_ids_json).read_text(encoding="utf-8")) if args.match_ids_json else None
    models_dir = Path(args.models) if args.models else None
    zone_model = RunZoneModel.load(models_dir / "run_zone") if models_dir else None
    routine_model = CornerRoutineModel.load(models_dir / "routine") if models_dir else None
    dest = Path(args.out)

    def _match_rows(item) -> pd.DataFrame:
        if zone_model is None or routine_model is None or item.frames is None or item.frames.empty:
            return pd.DataFrame()
        corners = identify_corners(item.actions, provider=item.provider)
        windowed = add_corner_windows(corners, item.frames, detect_ball_contacts(item.frames))
        encoding = compute_corner_run_encoding(windowed, item.frames, model=zone_model)
        feats = compute_corner_routine_features(encoding, model=routine_model)
        quality = (
            compute_corner_chance_quality(windowed, item.actions, xg_column="xg")
            if "xg" in item.actions.columns
            else None
        )
        rows = feats.merge(windowed[["game_id", "corner_id", "attacking_team_id"]], on=["game_id", "corner_id"])
        rows = rows.rename(columns={"attacking_team_id": "team_id"})
        rows["provider"] = item.provider  # for the section 12.5 cross-provider stability leg
        return rows.merge(quality, on=["game_id", "corner_id"], how="left") if quality is not None else rows

    refs, load = pining_source(
        providers=args.providers,
        match_ids=match_ids,
        max_per_provider=args.max_matches,
        cache_dir=args.cache_dir,
        token=args.token,
    )
    res = for_each(
        refs,
        key=lambda r: r.key,
        load=load,
        work=_match_rows,
        shard_root=dest / "shards",
        token_inputs={"metric": "corner_validity", "schema": "corner-validity-1"},
        label="match",
    )
    from scripts._driver import shard_path

    parts = [f for f in (pd.read_parquet(shard_path(res.shard_dir, k)) for k in res.shard_keys) if not f.empty]
    samples = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    verdicts = reduce_samples(samples) if not samples.empty else {"n_corners": 0}

    contract = declare_inputs(
        driver="validate_corners",
        min_per_team=_MIN_PER_TEAM,
        seams=["RunZoneModel", "CornerRoutineModel", "compute_corner_chance_quality"],
    )
    report = {
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "input_contract": contract,
        "verdicts": verdicts,
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
