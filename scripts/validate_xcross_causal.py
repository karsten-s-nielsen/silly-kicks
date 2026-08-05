"""Maintainer driver: xCross causal validation harness (TF-17 PR-C, ADR-015).

analyze() is PURE (opportunity frame -> metrics dict) so the e2e can drive it; run() does only
loader I/O + analyze + artifact write. The GK-vs-placebo finding is REPORTED, never asserted: a
null (or an unsupported claim) is a valid result.

Usage (on the box, scripts/ on sys.path, pining token in env):
  python scripts/validate_xcross_causal.py --out <DIR> \
      [--providers skillcorner,idsse,gradientsports] [--carrier-coverage-min 0.6] [--seed 0] \
      [--max-per-provider N] [--tracking-limit N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from _provenance import git_provenance, require_clean_tree

from silly_kicks.causal import matching as M
from silly_kicks.causal.opportunities import GK_BLOCK, PAPER_CONFOUNDERS

_OVERLAP_MIN = 0.5  # min fraction of treated inside the control PS range to claim common support


def _col_means_ignoring_missing(X: np.ndarray, miss: np.ndarray) -> np.ndarray:
    """Per-column mean over the finite entries; an ALL-missing column -> 0.0 (no nanmean-of-empty
    warning -- a fully-missing confounder carries no information, so 0 is the neutral fill)."""
    masked = np.where(miss, np.nan, X)
    has = np.isfinite(masked).any(axis=0)
    cm = np.zeros(X.shape[1], dtype=float)
    if has.any():
        cm[has] = np.nanmean(masked[:, has], axis=0)
    return cm


def _impute_with_indicator(X_gk: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Missing-indicator method (M2): a ``gk_missing`` column (1 if ANY GK col is NaN) + mean
    imputation. The indicator carries the missingness signal so imputation doesn't fabricate the
    confounder-of-interest. Returns (X_gk_imputed_with_indicator, indicator, nan_fraction)."""
    if X_gk.size == 0:
        return X_gk, np.zeros(len(X_gk)), 0.0
    miss = ~np.isfinite(X_gk)
    indicator = miss.any(axis=1).astype(float)
    nan_fraction = float(miss.any(axis=1).mean())
    col_mean = _col_means_ignoring_missing(X_gk, miss)
    imp = X_gk.copy()
    imp[miss] = np.take(col_mean, np.where(miss)[1])
    return np.hstack([imp, indicator.reshape(-1, 1)]), indicator, nan_fraction


def _mean_impute(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return X
    miss = ~np.isfinite(X)
    cm = _col_means_ignoring_missing(X, miss)
    out = X.copy()
    out[miss] = np.take(cm, np.where(miss)[1])
    return out


def _overlap_fraction(ps: np.ndarray, Z: np.ndarray) -> float:
    """Fraction of treated whose PS lies within [min, max] of the control PS (common support;
    a RANGE check -- no density trimming, stated in the report, R3-L4)."""
    t, c = ps[Z == 1], ps[Z == 0]
    if len(t) == 0 or len(c) == 0:
        return 0.0
    return float(((t >= c.min()) & (t <= c.max())).mean())


def analyze(opp: pd.DataFrame, *, seed: int = 0) -> dict:
    """Pure: opportunity frame -> metrics dict. No I/O."""
    Y, Z = opp["Y"].to_numpy(float), opp["Z"].to_numpy(int)
    # M5 positivity guard -- never silently emit NaN ATT
    if int(Z.sum()) == 0 or int((1 - Z).sum()) == 0:
        return {"status": "no_variation_in_treatment", "n_opportunities": len(opp), "n_treated": int(Z.sum())}

    X_base_raw = opp[PAPER_CONFOUNDERS].to_numpy(float)
    base_nan_frac = float((~np.isfinite(X_base_raw)).any(axis=1).mean()) if X_base_raw.size else 0.0  # R2-M3
    X_base = _mean_impute(X_base_raw)
    X_gk_raw = opp[GK_BLOCK].to_numpy(float)
    X_gk, _ind, gk_nan_frac = _impute_with_indicator(X_gk_raw)

    ps_base, _ = M.fit_propensity(X_base, Z, seed=seed)
    att_base = M.estimate_att(Y, Z, ps_base, X_base)
    X_full = np.hstack([X_base, X_gk])
    ps_full, _ = M.fit_propensity(X_full, Z, seed=seed)
    att_full = M.estimate_att(Y, Z, ps_full, X_full)
    atnt_full = M.estimate_atnt(Y, Z, ps_full, X_full)

    gk_shift = att_full.estimate - att_base.estimate
    # Placebo bands (TF-19/ADR-037): the GATE reads the CLUSTER band -- opportunities within a
    # match are dependent, so the null must reassign the GK block in whole matches (game_id
    # clusters); the legacy row-i.i.d. band is still reported so the frozen 4.18.0 record stays
    # comparable.
    cluster_ids = opp["game_id"].to_numpy()
    placebo_row = M.placebo_shift(X_base, X_gk, Y, Z, n_seeds=200, rng_seed=seed)
    placebo_cluster = M.placebo_shift(X_base, X_gk, Y, Z, n_seeds=200, rng_seed=seed, cluster_ids=cluster_ids)
    clears = abs(gk_shift) > max(placebo_cluster["band_p95"], M.GK_ABLATION_MIN_SHIFT)

    overlap = _overlap_fraction(ps_full, Z)  # M4
    smd_improved = bool(att_full.balance["smd_post"].abs().max() < att_full.balance["smd_pre"].abs().max())
    claim_supported = bool(overlap >= _OVERLAP_MIN and smd_improved)

    return {
        "status": "ok",
        "n_opportunities": len(opp),
        "n_treated": int(Z.sum()),
        "base_rate_Y": float(Y.mean()),
        "att_without_gk": {"estimate": att_base.estimate, "se": att_base.se},
        "att_with_gk": {"estimate": att_full.estimate, "se": att_full.se},
        "atnt_with_gk": {"estimate": atnt_full.estimate, "se": atnt_full.se},
        "gk_ablation_shift": gk_shift,
        "placebo_band_p95": placebo_cluster["band_p95"],  # the GATE band (cluster)
        "placebo_band_p95_cluster": placebo_cluster["band_p95"],
        "placebo_band_p95_row": placebo_row["band_p95"],  # legacy row-i.i.d. (4.18.0-comparable)
        "placebo_n_clusters": int(pd.unique(cluster_ids).size),
        "gk_clears_placebo_band": bool(clears),
        "gk_nan_fraction": gk_nan_frac,
        "base_nan_fraction": base_nan_frac,
        "ps_overlap_fraction": overlap,
        "smd_max_pre": float(att_full.balance["smd_pre"].abs().max()),
        "smd_max_post": float(att_full.balance["smd_post"].abs().max()),
        "causal_claim_supported": claim_supported,
        "seed": seed,
        "estimator": "abadie_imbens_2006_with_replacement_J1",
        "caveat": (  # R3-L4: state honesty about what the numbers do and don't mean
            "state-vs-sender + tracking-only opportunity detection; league/era differ from paper. "
            "Common support = treated-within-control-PS-range (no density trimming). Treated/control "
            "Y-windows are time-shifted (treated anchored at t_cross, control at entry). Z is a "
            "same-team cross within T of entry, clamped to possession continuity."
        ),
    }


def run(
    out: Path,
    providers: list[str],
    carrier_min: float,
    seed: int,
    *,
    max_per_provider=None,
    tracking_limit=None,
    token=None,
) -> dict:
    from _loader_pining import load_matches  # scripts/ on sys.path at runtime (mirrors the trainer)

    from scripts._driver import for_each, shard_path
    from silly_kicks.causal.opportunities import build_opportunities

    meta = _load_model_metadata()

    def _work(item):
        provider, _mid, actions, frames, home = item
        o = build_opportunities(frames, actions, home_team_id=home, model_metadata=meta)
        if o.empty:
            return None  # still writes an EMPTY shard: "ran, produced no opportunity"
        # The per-provider grouping below was a dict key in memory; persisted, it has to be a
        # COLUMN or a resumed pass cannot rebuild it. `build_opportunities` emits no `provider`
        # column of its own (verified), so this neither shadows nor duplicates one.
        return o.assign(provider=str(provider))

    res = for_each(
        load_matches(
            providers=providers, max_per_provider=max_per_provider, tracking_limit=tracking_limit, token=token
        ),
        key=lambda item: (str(item[0]), str(item[1])),
        work=_work,
        shard_root=Path(out) / "shards",
        # What determines an opportunity row: the builder and the model metadata whose confounders
        # it joins, plus the frame cap the spell state machine sees. `--providers` /
        # `--max-per-provider` only choose WHICH matches are walked; `--carrier-min` and `--seed`
        # are consumed by the ANALYSIS below, which re-runs from these shards on every invocation.
        token_inputs={
            "build_opportunities": "v1",
            "model_metadata": meta,
            "tracking_limit": tracking_limit,
        },
        tag="xcross_causal",
        label="match",
    )
    if res.failures:
        raise RuntimeError(f"{len(res.failures)} match(es) failed: {res.failures}. Re-run to retry only them.")

    # Combined from THIS PASS'S keys, not `_driver.reconcile`: no partition surface here (no
    # --match-ids-json, no worker tag), so a whole-generation read would fold in matches from a
    # wider earlier run over the same --out. See `reconcile`'s docstring.
    shards = [pd.read_parquet(shard_path(res.shard_dir, k)) for k in res.keys]
    opportunities = [f for f in shards if len(f)]
    all_opp = pd.concat(opportunities, ignore_index=True) if opportunities else pd.DataFrame(columns=["provider"])

    coverage, eligible = {}, []
    for provider in providers:
        df = all_opp[all_opp["provider"] == str(provider)]
        if df.empty:
            coverage[provider] = {"n_opp": 0, "carrier_coverage": 0.0, "included": False}
            continue
        cov = float(df["carrier_resolved"].mean())
        coverage[provider] = {"n_opp": len(df), "carrier_coverage": cov, "included": cov >= carrier_min}
        if cov >= carrier_min:
            # `provider` is dropped again here: it exists only to survive the shard round-trip, and
            # `analyze` must see exactly the frame the in-memory version handed it. Carrying an
            # extra column into a frozen causal harness is a change nobody asked for.
            eligible.append(df[df["carrier_resolved"]].drop(columns="provider"))

    if not eligible:
        metrics = {"status": "no_eligible_provider", "coverage": coverage}
    else:
        metrics = analyze(pd.concat(eligible, ignore_index=True), seed=seed)
        metrics["coverage"] = coverage
    _write(out, metrics)
    return metrics


def _load_model_metadata(variant: str = "default") -> dict:
    p = Path(__file__).resolve().parents[1] / "silly_kicks" / "tracking" / "_xcross_weights" / variant / "metadata.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _write(out: Path, metrics: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    prov = git_provenance()
    metrics["run_commit"] = prov["commit"]
    metrics["run_tree_dirty"] = prov["dirty"]
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "report.md").write_text(_render(metrics), encoding="utf-8")


def _render(m: dict) -> str:
    if m.get("status") != "ok":
        return f"# xCross causal validation (TF-17 PR-C)\n\nstatus: {m.get('status')}\n"
    return (
        "# xCross causal validation (TF-17 PR-C)\n\n"
        f"- Opportunities: {m['n_opportunities']} ({m['n_treated']} treated; base Y={m['base_rate_Y']:.3f})\n"
        f"- ATT without GK: {m['att_without_gk']['estimate']:.4f} (SE {m['att_without_gk']['se']:.4f})\n"
        f"- ATT with GK:    {m['att_with_gk']['estimate']:.4f} (SE {m['att_with_gk']['se']:.4f})\n"
        f"- GK ablation shift: {m['gk_ablation_shift']:.4f}; placebo band p95 cluster (GATE): "
        f"{m['placebo_band_p95_cluster']:.4f}; row (4.18.0-comparable): {m['placebo_band_p95_row']:.4f}\n"
        f"- **GK clears placebo band: {m['gk_clears_placebo_band']}** (reported, not a gate; cluster band)\n"
        f"- NaN fraction GK/base: {m['gk_nan_fraction']:.3f}/{m['base_nan_fraction']:.3f}; "
        f"PS overlap: {m['ps_overlap_fraction']:.3f}\n"
        f"- SMD max pre/post: {m['smd_max_pre']:.3f} / {m['smd_max_post']:.3f}; "
        f"**claim supported: {m['causal_claim_supported']}**\n"
        f"- Caveat: {m['caveat']}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--providers", default="skillcorner,idsse,gradientsports")
    ap.add_argument("--carrier-coverage-min", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-provider", type=int, default=None)
    ap.add_argument("--tracking-limit", type=int, default=None)
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit a dev run on a dirty tree; the artifact still records run_tree_dirty=true",
    )
    a = ap.parse_args()
    # ADR-037: enforce from the ENTRY POINT, before any corpus work is paid for. run() records
    # the truth; the CLI is what refuses.
    require_clean_tree(git_provenance(), allow_dirty=a.allow_dirty)
    run(
        a.out,
        a.providers.split(","),
        a.carrier_coverage_min,
        a.seed,
        max_per_provider=a.max_per_provider,
        tracking_limit=a.tracking_limit,
    )


if __name__ == "__main__":
    main()
