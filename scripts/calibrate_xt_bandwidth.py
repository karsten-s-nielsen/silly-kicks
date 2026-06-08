"""SK-xT-3 xT bandwidth/resolution calibration CLI — held-out transition-NLL sweep.

Pure objective lives in silly_kicks.calibration._xt_bandwidth_objective; this script owns I/O
(corpus loaders), the Optuna run, the manifest, and the reported downstream xT-quality cross-check.
Recommends a KDEParams+GridSpec; does NOT change any library default (ADR-009).

Usage:
    python scripts/calibrate_xt_bandwidth.py --source pining \
        --providers skillcorner idsse --n-trials 100 --store xt.db \
        --max-points-per-zone 5000 --report-out xt_report --cross-check
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from ruthless import InProcessBackend, render_json, render_summary_md
from ruthless.strategies.optuna_ import OptunaStrategy

import silly_kicks
from silly_kicks.calibration._spaces import grid_from_str, xt_bandwidth_config
from silly_kicks.calibration._xt_bandwidth_objective import XtBandwidthObjective

_XT_COLS = ["game_id", "start_x", "start_y", "end_x", "end_y", "type_id", "result_id"]


def run_xt_bandwidth(*, actions, n_trials, seed, store_path, max_points_per_zone=None):
    """Run the Optuna study on an already-loaded corpus (the testable seam — no I/O)."""
    objective = XtBandwidthObjective(actions, seed=seed, max_points_per_zone=max_points_per_zone)
    config = xt_bandwidth_config(n_trials=n_trials, store_path=store_path)
    result = OptunaStrategy(config, seed=seed).run(objective, backend=InProcessBackend())
    return result, objective


def _finite_or_none(x):
    """Non-finite (nan/inf) -> None for strict-JSON consumers: xt_holdout_nll_se is nan on a 1-fold
    corpus, and bare NaN is invalid JSON for non-Python readers."""
    return float(x) if x is not None and np.isfinite(x) else None


def build_manifest(*, source, seed, n_trials, max_points_per_zone, match_ids, result, cross_check=None):
    """Auditable manifest (recommends; never applies — ADR-009)."""
    rec = None
    best = result.best
    if best is not None:
        p = best.candidate.params
        grid = grid_from_str(str(p["grid"]))
        rec = {
            "method": "kde_smoothed",
            "bandwidth": float(p["bandwidth"]),
            "adaptive": bool(p["adaptive"]),
            "kernel": "gaussian",
            "grid": {"n_zones_x": grid.n_zones_x, "n_zones_y": grid.n_zones_y},
            "xt_holdout_nll": _finite_or_none(best.metrics.get("xt_holdout_nll")),
            "xt_holdout_nll_se": _finite_or_none(best.metrics.get("xt_holdout_nll_se")),
            "singh_holdout_nll": _finite_or_none(best.metrics.get("singh_holdout_nll")),
        }
    return {
        "stage": "xt_bandwidth",
        "source": source,
        "seed": seed,
        "n_trials": n_trials,
        "max_points_per_zone": max_points_per_zone,
        "match_ids": match_ids,
        "recommendation": rec,
        "recommendation_scope": "optimal for held-out destination likelihood; xT-quality impact unverified",
        "applies_to_library_default": False,
        "bandwidth_dual_meaning": "adaptive=True -> Silverman multiplier; adaptive=False -> raw SPADL metres",
        "downstream_xt_quality_cross_check": cross_check,
        "silly_kicks_version": silly_kicks.__version__,
        "ruthless_version": version("ruthless-efficiency"),
        "generated_date": _dt.date.today().isoformat(),
    }


def _scores_per_game(actions, *, k):
    """vaep ``scores`` labels computed PER GAME, reassembled in row order.

    ``vaep.labels.scores`` uses a raw ``shift(-k)`` with NO ``game_id`` grouping, so on a multi-game
    frame it leaks goal-lookahead across game boundaries. Grouping by ``game_id`` first makes the
    label boundary-safe. ``rate()`` has no lookahead, so only the label needs this; the result stays
    row-aligned to ``actions``.
    """
    from silly_kicks.spadl import add_names
    from silly_kicks.vaep.labels import scores as scores_label

    actions = actions.reset_index(drop=True)
    out = np.empty(len(actions), dtype=float)
    for _gid, g in actions.groupby("game_id", sort=False):
        pos = g.index.to_numpy()  # positions in row order (index is 0..n-1 after reset)
        out[pos] = scores_label(add_names(g.reset_index(drop=True)), nr_actions=k)["scores"].to_numpy()
    return out


def xt_quality_cross_check(actions, recommendation, *, k=10, seed=42):
    """Reported (not gated) downstream xT-quality signal.

    Spearman rho between Delta-rate = rate(end) - rate(start) and "the in-possession team scored
    within K actions", on the held-out CV folds, for the recommended grid vs the Singh grid. A
    single number per grid; if NLL-best does NOT also win rho, that is a finding worth surfacing.
    """
    from scipy.stats import spearmanr

    from silly_kicks.calibration._cv import match_cv_splits
    from silly_kicks.xthreat import ExpectedThreat, KDEParams

    actions = actions.reset_index(drop=True)
    g = recommendation["grid"]
    nx, ny = int(g["n_zones_x"]), int(g["n_zones_y"])
    params = KDEParams(bandwidth=float(recommendation["bandwidth"]), adaptive=bool(recommendation["adaptive"]))
    game_ids = actions["game_id"].astype(str).to_numpy()  # robust to provider-asymmetric dtypes

    def _rho(method, kde_params):
        deltas, labels = [], []
        for tr, te in match_cv_splits(game_ids):
            train, test = actions.iloc[tr], actions.iloc[te]
            xt = (
                ExpectedThreat(l=nx, w=ny, method="kde_smoothed", params=kde_params).fit(train)
                if method == "kde"
                else ExpectedThreat(l=nx, w=ny, method="singh_counts").fit(train)
            )
            if not np.any(xt.xT):
                continue
            d = xt.rate(test)  # NaN on non-move rows; rate has no lookahead -> boundary-safe
            y = _scores_per_game(test, k=k)  # per-game label (no cross-game goal leak)
            mask = np.isfinite(d)
            deltas.append(d[mask])
            labels.append(y[mask])
        if not deltas:
            return float("nan")
        dd, yy = np.concatenate(deltas), np.concatenate(labels)
        if len(dd) < 3 or len(np.unique(yy)) < 2:
            return float("nan")
        # spearmanr returns a SignificanceResult whose `.statistic` is the correlation; the scipy
        # stub types the result as an opaque class that omits the attribute, so ignore the stub gap.
        return float(spearmanr(dd, yy).statistic)  # type: ignore[reportAttributeAccessIssue]

    return {"rho_recommended": _rho("kde", params), "rho_singh": _rho("singh", None), "k": k}


def _ids_from_game_id(df: pd.DataFrame) -> dict[str, list[str]]:
    """Reconstruct ``{provider: [match_id, ...]}`` from the provider-qualified ``game_id`` column."""
    ids: dict[str, list[str]] = {}
    for gid in df["game_id"].unique():
        provider, _, mid = str(gid).partition(":")
        ids.setdefault(provider, []).append(mid)
    return ids


# The canonical SPADL columns the sweep (type/result/coords) + cross-check (add_names + vaep.labels
# scores: game/period/team/type/result) actually use. Provider-specific extras (e.g. GS
# original_event_id / is_synthetic) are dropped — they are heterogeneous across providers (mixed
# int/str object columns) and would break the parquet write (pyarrow ArrowTypeError).
_CORPUS_COLS = [
    "game_id",
    "period_id",
    "time_seconds",
    "team_id",
    "player_id",
    "bodypart_id",
    "type_id",
    "result_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
]


def _canonicalize_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Project to the standard SPADL columns and string-cast the provider-asymmetric id columns so
    the multi-provider corpus is parquet-serializable (heterogeneous object columns — mixed-dtype
    team_id/player_id and provider-only ids — otherwise break pyarrow)."""
    out = df[[c for c in _CORPUS_COLS if c in df.columns]].copy()
    for col in ("game_id", "team_id", "player_id"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def _assemble_corpus(args) -> pd.DataFrame:
    """Load SPADL actions (minimal tracking footprint) into one canonical provider-qualified corpus."""
    if args.source == "pining":
        import scripts._loader_pining as loader

        gen = loader.load_matches(
            providers=args.providers,
            tracking_limit=1,
            max_per_provider=args.max_matches_per_provider,
            cache_dir=getattr(args, "cache_dir", None),  # persistent artifact cache on the run box
        )
        parts = [
            # provider-qualified unique string game_id (mixed-dtype + cross-provider collision guard)
            actions.assign(game_id=f"{provider}:{mid}")
            for provider, mid, actions, _frames, _home in gen
        ]
        return _canonicalize_corpus(pd.concat(parts, ignore_index=True))
    # Databricks is NOT a default source — pining is source #1; this path is reserved (separate row).
    import scripts._loader_databricks as loader

    conn = loader._connect()
    try:
        cur = conn.cursor()
        cols = ", ".join(_XT_COLS)
        df = loader._query_param(cur, f"SELECT {cols} FROM soccer_analytics.bronze.spadl_actions")  # noqa: S608
    finally:
        conn.close()
    return _canonicalize_corpus(df.assign(game_id=df["game_id"].map(lambda g: f"databricks:{g}")))


def _load_corpus(args) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Assemble the corpus, with an optional parquet cache + optional game-subsample for contrasts.

    ``--corpus-cache``: if the parquet exists, load it (skips ALL downloads + parsing — the basis of
    the corpus-size contrast: build the full corpus once, then subsample it cheaply). Otherwise
    assemble from the source and write the parquet. ``--subsample-games``: keep a seeded random
    subset of games (applied AFTER load, so it reuses the cache).
    """
    corpus_cache = getattr(args, "corpus_cache", None)
    if corpus_cache and not Path(corpus_cache).exists():
        # Fail FAST (before the multi-minute corpus load) if we cannot write the parquet cache —
        # pandas only surfaces "Unable to find a usable engine" at to_parquet time, after the load.
        import importlib.util

        if not (importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet")):
            raise ValueError("--corpus-cache requires a parquet engine: pip install pyarrow")
    if corpus_cache and Path(corpus_cache).exists():
        df = pd.read_parquet(corpus_cache)
    else:
        df = _assemble_corpus(args)
        if corpus_cache:
            Path(corpus_cache).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(corpus_cache, index=False)

    n = getattr(args, "subsample_games", None)
    if n:
        games = df["game_id"].unique()
        if n < len(games):
            rng = np.random.default_rng(args.seed)
            keep = set(rng.choice(games, size=n, replace=False))
            df = df[df["game_id"].isin(keep)].reset_index(drop=True)
    return df, _ids_from_game_id(df)


def main() -> None:
    ap = argparse.ArgumentParser(description="SK-xT-3 xT bandwidth/resolution calibration")
    ap.add_argument("--source", choices=["pining", "databricks"], default="pining")
    ap.add_argument("--providers", nargs="+", default=["skillcorner", "idsse"])
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--store", required=True)
    ap.add_argument("--max-points-per-zone", type=int, default=None)
    ap.add_argument("--max-matches-per-provider", type=int, default=None)
    ap.add_argument("--cache-dir", default=None, help="persistent dir for cached pining artifact downloads")
    ap.add_argument("--corpus-cache", default=None, help="parquet cache of the assembled corpus (skip download+parse)")
    ap.add_argument("--subsample-games", type=int, default=None, help="keep a seeded random subset of games (contrast)")
    ap.add_argument("--report-out", default="xt_bandwidth_report")
    ap.add_argument("--cross-check", action="store_true", help="run the reported downstream xT-quality cross-check")
    args = ap.parse_args()

    actions, match_ids = _load_corpus(args)
    result, _obj = run_xt_bandwidth(
        actions=actions,
        n_trials=args.n_trials,
        seed=args.seed,
        store_path=args.store,
        max_points_per_zone=args.max_points_per_zone,
    )
    cross_check = None
    if args.cross_check and result.best is not None:
        p = result.best.candidate.params
        grid = grid_from_str(str(p["grid"]))
        cross_check = xt_quality_cross_check(
            actions,
            {
                "bandwidth": float(p["bandwidth"]),
                "adaptive": bool(p["adaptive"]),
                "grid": {"n_zones_x": grid.n_zones_x, "n_zones_y": grid.n_zones_y},
            },
            seed=args.seed,
        )
    manifest = build_manifest(
        source=args.source,
        seed=args.seed,
        n_trials=args.n_trials,
        max_points_per_zone=args.max_points_per_zone,
        match_ids=match_ids,
        result=result,
        cross_check=cross_check,
    )
    report = {"ruthless": json.loads(render_json(result)), "calibration_manifest": manifest}
    with open(f"{args.report_out}.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(f"{args.report_out}.md", "w", encoding="utf-8") as fh:
        fh.write(render_summary_md(result))
        fh.write("\n\n## Calibration manifest\n\n```json\n")
        fh.write(json.dumps(manifest, indent=2))
        fh.write("\n```\n")
    print(render_summary_md(result))
    print(f"Best: {result.best.metrics if result.best else None}")


if __name__ == "__main__":
    main()
