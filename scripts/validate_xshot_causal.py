"""Maintainer driver: shot-arm (xS) causal validation harness (TF-19 SS3.3, ADR-037).

Thin clone of scripts/validate_xcross_causal.py over ``build_opportunities(config=
shot_arm_config(...))``: the same PS-matching + GK-ablation-vs-placebo structure on the xS
confounder/GK-block columns, with the CLUSTER placebo (match-level exchangeability) as the
gate band. The measured quantity is GK-CONFOUNDER ENTANGLEMENT (the supportive-context input
to ``regate_verdict``), NOT a causal deterrence estimate. analyze() is PURE (opportunity
frame -> metrics dict) so tests can drive it; run() does only loader I/O + analyze + artifact
write. A null (or a refused verdict) is a valid result.

R10: the control-conversion door is narrow in real data (a control converts only via a
success-shot inside the anchor-inclusive window with no in-spell shot) -- below
``SHOT_ARM_MIN_CONTROL_CONVERSIONS`` the entanglement verdict is REFUSED (``degenerate``):
near-zero control conversions re-create P1's control-Y degeneracy as a DATA condition, making
the ATT shift confounder-insensitive by construction. The rate and count are always reported.

The corpus pass writes PER-MATCH SHARDS on completion, so a crash resumes instead of restarting
~81 matches, and N processes can split the corpus with disjoint --match-ids-json slices against a
shared --out. Analysing a single slice would emit a metrics.json indistinguishable from a
full-corpus one, so partitioned workers pass --build-only and the analysis runs once at the end.

Usage (on the box, scripts/ on sys.path, pining token in env):
  python scripts/validate_xshot_causal.py --list-matches > all.json
  # N workers, disjoint slices, shared --out:
  python scripts/validate_xshot_causal.py --out <DIR> --match-ids-json <SLICE.json> --build-only
  # once, after the workers finish -- builds nothing new, analyses every shard:
  python scripts/validate_xshot_causal.py --out <DIR> \
      [--providers skillcorner,idsse,gradientsports] [--carrier-coverage-min 0.6] [--seed 0]

--help is dep-light: args are parsed before any loader / silly_kicks import (house pattern).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._input_contract import declare_inputs

# --- Claim-gate constants (registered; mirror validate_xcross_causal.py) -------------------
_OVERLAP_MIN = 0.5  # min fraction of treated inside the control PS range to claim common support
# R10 (registered): minimum CONTROL conversions for the entanglement verdict to be meaningful.
SHOT_ARM_MIN_CONTROL_CONVERSIONS = 30


def input_contract() -> dict:
    """Declare WHICH SYMBOLS these numbers depend on (Cycle B).

    Both causal imports are function-local by necessity, not style: `SHOT_ARM_CONFOUNDERS` and
    `shot_arm_config` are imported inside `analyze()`, not at module scope, so a module-level
    reference would raise NameError at call time.
    """
    from silly_kicks.causal import SHOT_ARM_CONFOUNDERS, shot_arm_config
    from silly_kicks.tracking import _geometry as _geo

    return declare_inputs(
        driver="validate_xshot_causal",
        covariates={
            "shot_arm": SHOT_ARM_CONFOUNDERS,
            "gk_block": shot_arm_config({}).gk_block,
        },
        geometry_version=_geo.GEOMETRY_VERSION,
        extractors=("silly_kicks.tracking._xshot_occurrence",),
    )


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


def _entanglement_gate(opp: pd.DataFrame) -> dict:
    """Pure R10 gate at the opportunity-frame layer: report the control-Y rate/count and whether
    the entanglement verdict is MEASURABLE (>= SHOT_ARM_MIN_CONTROL_CONVERSIONS control
    conversions). Near-zero control conversions make the ATT shift structurally
    confounder-insensitive (P1's degeneracy as a DATA condition) -> the verdict must be
    refused, never read."""
    controls = opp[opp["Z"] == 0]
    n_conversions = int(controls["Y"].sum())
    return {
        "control_n": len(controls),
        "control_conversions": n_conversions,
        "control_y_rate": float(controls["Y"].mean()) if len(controls) else float("nan"),
        "measurable": bool(n_conversions >= SHOT_ARM_MIN_CONTROL_CONVERSIONS),
    }


def _cluster_key(opp: pd.DataFrame) -> np.ndarray:
    """The match-level cluster key, distinguishing providers that share a ``game_id`` value.

    This arm POOLS providers, and `game_id` is only unique WITHIN one. Two providers using the
    same value -- gradientsports int ``123`` and skillcorner str ``"123"`` -- would land in one
    cluster under a `game_id`-only key, fusing two unrelated matches and corrupting the
    cluster-exchangeable null. The composite is built only when `provider` is present, so a
    single-provider caller (and every existing test) keeps the previous key exactly.
    """
    if "provider" not in opp.columns:
        return opp["game_id"].to_numpy()
    return (opp["provider"].astype(str) + "\x1f" + opp["game_id"].astype(str)).to_numpy()


def analyze(opp: pd.DataFrame, *, seed: int = 0, n_seeds: int = 200) -> dict:
    """Pure: shot-arm opportunity frame -> metrics dict. No I/O. ``n_seeds`` sizes the placebo
    null (200 = the registered owner-run value; tests may shrink it)."""
    from silly_kicks.causal import SHOT_ARM_CONFOUNDERS, shot_arm_config
    from silly_kicks.causal import matching as M

    Y, Z = opp["Y"].to_numpy(float), opp["Z"].to_numpy(int)
    # M5 positivity guard -- never silently emit NaN ATT
    if int(Z.sum()) == 0 or int((1 - Z).sum()) == 0:
        return {"status": "no_variation_in_treatment", "n_opportunities": len(opp), "n_treated": int(Z.sum())}

    confounders = list(SHOT_ARM_CONFOUNDERS)
    gk_block = list(shot_arm_config({}).gk_block)  # ('GK_r', 'GK_theta') -- metadata-independent
    X_base_raw = opp[confounders].to_numpy(float)
    base_nan_frac = float((~np.isfinite(X_base_raw)).any(axis=1).mean()) if X_base_raw.size else 0.0
    X_base = _mean_impute(X_base_raw)
    X_gk_raw = opp[gk_block].to_numpy(float)
    X_gk, _ind, gk_nan_frac = _impute_with_indicator(X_gk_raw)

    ps_base, _ = M.fit_propensity(X_base, Z, seed=seed)
    att_base = M.estimate_att(Y, Z, ps_base, X_base)
    X_full = np.hstack([X_base, X_gk])
    ps_full, _ = M.fit_propensity(X_full, Z, seed=seed)
    att_full = M.estimate_att(Y, Z, ps_full, X_full)
    atnt_full = M.estimate_atnt(Y, Z, ps_full, X_full)

    gk_shift = att_full.estimate - att_base.estimate
    # The GATE reads the CLUSTER band (match-level exchangeability -- opportunities within a
    # match are dependent); the row-i.i.d. band is reported for cross-arm comparability with
    # the frozen 4.18.0 xCross record.
    cluster_ids = _cluster_key(opp)
    placebo_row = M.placebo_shift(X_base, X_gk, Y, Z, n_seeds=n_seeds, rng_seed=seed)
    placebo_cluster = M.placebo_shift(X_base, X_gk, Y, Z, n_seeds=n_seeds, rng_seed=seed, cluster_ids=cluster_ids)
    clears = abs(gk_shift) > max(placebo_cluster["band_p95"], M.GK_ABLATION_MIN_SHIFT)

    overlap = _overlap_fraction(ps_full, Z)  # M4
    smd_improved = bool(att_full.balance["smd_post"].abs().max() < att_full.balance["smd_pre"].abs().max())
    claim_supported = bool(overlap >= _OVERLAP_MIN and smd_improved)

    gate = _entanglement_gate(opp)
    if gate["measurable"]:
        entanglement, refusal_reason = ("clears" if clears else "inside_band"), None
    else:  # R10: the verdict is REFUSED; the numbers are still reported (reported-not-gated)
        entanglement = "degenerate"
        refusal_reason = (
            f"control conversions {gate['control_conversions']} < "
            f"SHOT_ARM_MIN_CONTROL_CONVERSIONS={SHOT_ARM_MIN_CONTROL_CONVERSIONS} -- control-Y is "
            "too degenerate for the ATT shift to measure GK entanglement (R10/P1)"
        )

    return {
        "status": "ok",
        "n_opportunities": len(opp),
        "n_treated": int(Z.sum()),
        "base_rate_Y": float(Y.mean()),
        "control_n": gate["control_n"],
        "control_conversions": gate["control_conversions"],  # R10: reported alongside the verdict
        "control_y_rate": gate["control_y_rate"],
        "att_without_gk": {"estimate": att_base.estimate, "se": att_base.se},
        "att_with_gk": {"estimate": att_full.estimate, "se": att_full.se},
        "atnt_with_gk": {"estimate": atnt_full.estimate, "se": atnt_full.se},
        "gk_ablation_shift": gk_shift,
        "placebo_band_p95": placebo_cluster["band_p95"],  # the GATE band (cluster)
        "placebo_band_p95_cluster": placebo_cluster["band_p95"],
        "placebo_band_p95_row": placebo_row["band_p95"],
        "placebo_n_clusters": int(pd.unique(cluster_ids).size),
        "gk_clears_placebo_band": bool(clears),
        "entanglement": entanglement,  # 'clears' | 'inside_band' | 'degenerate' (regate_verdict input)
        "entanglement_refused": bool(not gate["measurable"]),
        "entanglement_refusal_reason": refusal_reason,
        "gk_nan_fraction": gk_nan_frac,
        "base_nan_fraction": base_nan_frac,
        "ps_overlap_fraction": overlap,
        "smd_max_pre": float(att_full.balance["smd_pre"].abs().max()),
        "smd_max_post": float(att_full.balance["smd_post"].abs().max()),
        "causal_claim_supported": claim_supported,
        "seed": seed,
        "estimator": "abadie_imbens_2006_with_replacement_J1",
        "caveat": (  # honesty about what the numbers do and don't mean (mirrors the cross arm)
            "GK-confounder ENTANGLEMENT, not causal deterrence: the shot arm measures whether the "
            "GK block carries Z/Y-aligned signal beyond the xS positional confounders. "
            "Anchor-INCLUSIVE 6 s success-shot outcome (second re-registration, P1); "
            "treated/control Y-windows are time-shifted (treated anchored at the in-spell shot, "
            "control at entry). The cluster placebo band gates (match-level exchangeability); "
            "the row band is reported for comparability."
        ),
    }


def build_shards(
    out: Path,
    providers: list[str],
    *,
    match_ids=None,
    max_per_provider=None,
    tracking_limit=None,
    token=None,
    provenance=None,
    partition_tag: str = "all",
) -> dict:
    """Per-match opportunity shards, written ON COMPLETION so a crash resumes.

    The corpus walk is the whole cost of this driver (~81 matches at minutes each). Building it
    inline and holding everything in memory until the end is the shape that destroyed an 8.7h
    power run: one raise after the last match and every opportunity is gone. Shards also make the
    pass PARTITIONABLE -- N processes with disjoint `--match-ids-json` slices share one `--out`.

    An existing shard is skipped, and a shard is written even when a match yields NO opportunities:
    absent means "not yet run", present-and-empty means "run, produced nothing". Conflating the two
    would make every resume silently recompute the barren matches forever.
    """
    from _loader_pining import load_matches  # scripts/ on sys.path at runtime (mirrors the trainer)

    from scripts._driver import for_each
    from scripts._partition import providers_for_slice
    from scripts._provenance import git_provenance
    from silly_kicks.causal.opportunities import build_opportunities, shot_arm_config

    prov = provenance or git_provenance()
    # A provider with no ids in THIS slice belongs to another worker. Without this the loader reads
    # an empty/absent slice as "the whole manifest" -- so a run sliced on one provider would have
    # every worker load the others in full, writing the same shard paths concurrently.
    providers = providers_for_slice(providers, match_ids)
    meta = _load_model_metadata()
    cfg = shot_arm_config(meta)
    dest = Path(out)

    def _work(item):
        provider, match_id, actions, frames, home = item
        o = build_opportunities(frames, actions, home_team_id=home, model_metadata=meta, config=cfg)
        o = o.copy()
        # The provider is what `coverage` is keyed on, so it must survive the round trip rather than
        # being re-derived from the filename -- a provider containing "__" would silently mis-split.
        o["provider"] = str(provider)
        o["match_id"] = str(match_id)
        return o

    res = for_each(
        load_matches(
            providers=providers,
            match_ids=match_ids,
            max_per_provider=max_per_provider,
            tracking_limit=tracking_limit,
            token=token,
        ),
        key=lambda item: (str(item[0]), str(item[1])),
        work=_work,
        counters=lambda _item, frame: {"n_matches": 1, "n_opportunities": len(frame)},
        shard_root=dest / "shards",
        # What determines a shard's CONTENT: the opportunity builder, the shot-arm config, and the
        # model metadata the config is derived from. `causal/matching.py` is deliberately NOT
        # declared -- it runs in `_entanglement_analysis`, which re-reads these shards on every
        # invocation, so it consumes the content rather than determining it.
        token_inputs={
            "build_opportunities": "v1",
            "shot_arm_config": "v1",
            "model_metadata": meta,
            # The sibling this file's docstring calls its "thin clone" source
            # (`validate_xcross_causal`) declares this with the rationale "plus the frame cap the
            # spell state machine sees". The clone dropped it. A capped dev smoke would otherwise
            # be skipped-into by the real pass, and the S3.3 entanglement verdict -- cited by the
            # 4.60.0 `joins_with_caveat` decision -- would publish under a clean run_commit.
            "tracking_limit": tracking_limit,
        },
        tag=partition_tag,
        label="match",
    )
    totals = dict(res.counters)

    tag = partition_tag
    (dest / f"manifest_{tag}.json").write_text(
        json.dumps(
            {
                **totals,
                # Records the generation token, which `load_shards` reads back to find THIS
                # generation's directory instead of blindly globbing every generation on disk.
                # `res.manifest()`, not a bare `manifest_fields(...)`: only the method threads
                # `counters_unrecorded`, and a hand-written call silently defaults it to 0 -- so a
                # resumed worker whose sidecars were missing would report a complete corpus.
                **res.manifest(),
                "run_commit": prov["commit"],
                "run_tree_dirty": prov["dirty"],
                "run_tree_state": prov["tree_state"],
                "partition": tag,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return totals


def _generation_dir(out: Path) -> Path:
    """The shard directory of the CURRENT generation, read back from the manifests.

    Since the `_driver` migration shards live in ``shards/<token>/``, where the token is a digest of
    the declared inputs. A bare ``rglob`` would therefore concatenate EVERY generation on disk --
    two values for the same match, with the row count inflated by exactly the number of stale
    generations. That is the defect the directory form exists to make unrepresentable, and it would
    have been reintroduced here at the READ side. `manifest_fields` records the token for exactly
    this lookup.

    Mixed generations across workers are REFUSED rather than merged: a corpus assembled from two
    token generations is not a corpus, and `aggregate_manifests` reports the same condition as
    `generation_consistent`.

    Falls back to the flat layout when no manifest carries a generation, so a PRE-migration shard
    set still loads instead of silently reading as empty.
    """
    root = Path(out) / "shards"
    generations = set()
    for m in sorted(Path(out).glob("manifest_*.json")):
        gen = json.loads(m.read_text(encoding="utf-8")).get("generation")
        if gen:
            generations.add(str(gen))
    if len(generations) > 1:
        raise SystemExit(
            f"shards under {root} span multiple generations {sorted(generations)}: the declared "
            f"inputs changed between workers, so these shards do not describe one corpus. Re-run "
            f"the build, or prune the stale generation."
        )
    return root / generations.pop() if generations else root


def load_shards(out: Path) -> pd.DataFrame:
    """Every shard of the CURRENT generation under ``out``, concatenated. Empty when none exist."""
    shards = sorted(_generation_dir(Path(out)).glob("*.parquet"))
    parts = [pd.read_parquet(s) for s in shards]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def analyze_shards(out: Path, providers: list[str], carrier_min: float, seed: int, *, provenance=None) -> dict:
    """Coverage + the entanglement analysis over the persisted shards.

    Separated from the corpus walk so a failure here costs seconds instead of another ~11h pass,
    and so N partitioned workers can build shards without each producing a partial metrics.json.
    """
    from scripts._partition import aggregate_manifests
    from scripts._provenance import git_provenance

    prov = provenance or git_provenance()
    allopp = load_shards(out)

    coverage, eligible = {}, []
    for provider in providers:
        df = allopp[allopp["provider"] == provider] if len(allopp) else allopp
        if not len(df):
            coverage[provider] = {"n_opp": 0, "carrier_coverage": 0.0, "included": False}
            continue
        cov = float(df["carrier_resolved"].mean())
        coverage[provider] = {"n_opp": len(df), "carrier_coverage": cov, "included": cov >= carrier_min}
        if cov >= carrier_min:
            eligible.append(df[df["carrier_resolved"]])

    if not eligible:
        metrics = {"status": "no_eligible_provider", "coverage": coverage}
    else:
        metrics = analyze(pd.concat(eligible, ignore_index=True), seed=seed)
        metrics["coverage"] = coverage

    # SCOPE, recorded explicitly. With partitioned workers the analysis can legitimately run over a
    # subset, and a metrics.json that does not say how much of the corpus it saw is the same
    # "artifact misdescribes its own scope" defect as a manifest reporting one partition's totals.
    corpus = aggregate_manifests(out, defaults=("n_matches", "n_opportunities"))
    metrics["corpus"] = {
        "n_matches": corpus["n_matches"],
        "n_opportunities": corpus["n_opportunities"],
        "n_partitions": corpus["n_partitions"],
        "n_shards": len(list(_generation_dir(Path(out)).glob("*.parquet"))),
        "commit_consistent": corpus["commit_consistent"],
    }
    metrics["input_contract"] = input_contract()
    metrics["run_commit"] = prov["commit"]
    metrics["run_tree_dirty"] = prov["dirty"] or corpus["run_tree_dirty"]
    _write(Path(out), metrics)
    return metrics


def run(
    out: Path,
    providers: list[str],
    carrier_min: float,
    seed: int,
    *,
    match_ids=None,
    max_per_provider=None,
    tracking_limit=None,
    token=None,
    provenance=None,
    build_only: bool = False,
    partition_tag: str = "all",
) -> dict:
    """Build any missing shards, then analyse them.

    This artifact IS the S3.3 entanglement measurement that corrects F6 -- the finding that a
    registered DEFAULT, rather than a measured value, decided the 4.60.0 verdict. Publishing its
    replacement with no run_commit would be the same failure wearing different clothes.

    ENFORCEMENT of the clean tree lives in main(), not here: `run` is called directly by tests, and
    a work function that refuses to execute on a dirty checkout is untestable without mocking git.
    So this records the truth unconditionally while the CLI is what REFUSES.
    """
    from scripts._provenance import git_provenance

    prov = provenance or git_provenance()
    build_shards(
        out,
        providers,
        match_ids=match_ids,
        max_per_provider=max_per_provider,
        tracking_limit=tracking_limit,
        token=token,
        provenance=prov,
        partition_tag=partition_tag,
    )
    if build_only:
        # A partitioned worker stops here: analysing its own slice would write a metrics.json that
        # looks corpus-wide and is not.
        return {"status": "shards_built", "run_commit": prov["commit"]}
    return analyze_shards(out, providers, carrier_min, seed, provenance=prov)


def _load_model_metadata(variant: str = "default") -> dict:
    """xS bundled metadata: shot types + the carrier params the model was trained against."""
    p = Path(__file__).resolve().parents[1] / "silly_kicks" / "tracking" / "_xshot_weights" / variant / "metadata.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _write(out: Path, metrics: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "report.md").write_text(_render(metrics), encoding="utf-8")


def _render(m: dict) -> str:
    if m.get("status") != "ok":
        return f"# Shot-arm (xS) causal validation (TF-19, ADR-037)\n\nstatus: {m.get('status')}\n"
    verdict = f"REFUSED ({m['entanglement_refusal_reason']})" if m["entanglement_refused"] else m["entanglement"]
    return (
        "# Shot-arm (xS) causal validation (TF-19, ADR-037)\n\n"
        f"- Opportunities: {m['n_opportunities']} ({m['n_treated']} treated; base Y={m['base_rate_Y']:.3f})\n"
        f"- Control Y (R10): {m['control_conversions']}/{m['control_n']} conversions "
        f"(rate {m['control_y_rate']:.4f}; floor {SHOT_ARM_MIN_CONTROL_CONVERSIONS})\n"
        f"- ATT without GK: {m['att_without_gk']['estimate']:.4f} (SE {m['att_without_gk']['se']:.4f})\n"
        f"- ATT with GK:    {m['att_with_gk']['estimate']:.4f} (SE {m['att_with_gk']['se']:.4f})\n"
        f"- GK ablation shift: {m['gk_ablation_shift']:.4f}; placebo band p95 cluster (GATE): "
        f"{m['placebo_band_p95_cluster']:.4f}; row (comparability): {m['placebo_band_p95_row']:.4f}\n"
        f"- **Entanglement verdict: {verdict}** (supportive context, not causal deterrence)\n"
        f"- NaN fraction GK/base: {m['gk_nan_fraction']:.3f}/{m['base_nan_fraction']:.3f}; "
        f"PS overlap: {m['ps_overlap_fraction']:.3f}\n"
        f"- SMD max pre/post: {m['smd_max_pre']:.3f} / {m['smd_max_post']:.3f}; "
        f"**claim supported: {m['causal_claim_supported']}**\n"
        f"- Caveat: {m['caveat']}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None, help="output dir (not needed with --list-matches)")
    ap.add_argument("--providers", default="skillcorner,idsse,gradientsports")
    ap.add_argument("--carrier-coverage-min", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-provider", type=int, default=None)
    ap.add_argument("--tracking-limit", type=int, default=None)
    ap.add_argument(
        "--match-ids-json",
        default=None,
        help=(
            'JSON {"gradientsports": ["10502", ...]} pinning WHICH matches this process handles. '
            "This is how the corpus pass is PARALLELISED: split the id list N ways and launch N "
            "processes, each with its own slice, a SHARED --out and --build-only."
        ),
    )
    ap.add_argument(
        "--build-only",
        action="store_true",
        help="build shards and stop (partitioned workers use this; analysing one slice would "
        "write a metrics.json that looks corpus-wide and is not)",
    )
    ap.add_argument(
        "--list-matches",
        action="store_true",
        help="print the available match ids as JSON and exit (build the parallel split from this)",
    )
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; artifact is marked)")
    a = ap.parse_args()

    if a.list_matches:
        from scripts._partition import list_match_ids

        print(json.dumps(list_match_ids(a.providers.split(",")), indent=2))
        return
    if not a.out:
        raise SystemExit("--out is required unless --list-matches is given")
    # Provenance FIRST -- before the corpus walk `run` starts, so a dirty tree costs seconds
    # instead of hours and never reaches an artifact.
    from scripts._provenance import git_provenance, require_clean_tree

    prov = require_clean_tree(git_provenance(), allow_dirty=a.allow_dirty)
    from scripts._partition import worker_tag

    match_ids = json.loads(Path(a.match_ids_json).read_text(encoding="utf-8")) if a.match_ids_json else None
    m = run(
        a.out,
        a.providers.split(","),
        a.carrier_coverage_min,
        a.seed,
        match_ids=match_ids,
        max_per_provider=a.max_per_provider,
        tracking_limit=a.tracking_limit,
        provenance=prov,
        build_only=a.build_only,
        partition_tag=worker_tag(a.match_ids_json),
    )
    print(json.dumps({k: m.get(k) for k in ("status", "corpus", "entanglement_verdict")}, indent=2, default=str))


if __name__ == "__main__":
    main()
