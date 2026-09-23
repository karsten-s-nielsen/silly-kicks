"""Task 11 / D-S9 -- cross-provider xt_gk comparability gate (owner-run, REPORTED not CI).

Before the lakehouse pools SkillCorner and GS ``xt_gk`` into one column, confirm they are on the
same SCALE on overlapping conditions (matched pass-distance bands). Per G2 the expected outcome
post common-scale p-calibration is **within_tolerance** or **escalate** -- a residual xt_gk offset
is, by elimination, the threat-term difference = genuine football (SC's ~17m goal-kicks), which must
NOT be re-scaled away. ``correctable`` (a per-variant post-composite affine on xt_gk, clamped) is
RARE and requires positive evidence the offset is a measurement artifact (uniform across all bands).

Usage:
    python scripts/_xtgk_comparability.py --gs-provider gradientsports --max-per-provider 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _driver import for_each, shard_path
from _events_admission import events_only_loader
from _loader_pining import pining_source, resolve_cache_dir
from _xt_corpus import fit_xt_from_count_pass, xt_count_pass

from silly_kicks.tracking.features import add_xt_gk

_BANDS = [(0.0, 15.0), (15.0, 30.0), (30.0, 45.0), (45.0, 120.0)]
_OFFSET_TOL = 0.01  # |mean xt_gk offset| within a band considered "on the same scale"
_MIN_N = 30  # per-band minimum sample for the band to be powered


def compare_xtgk_distributions(sc, gs, *, bands=_BANDS, offset_tol=_OFFSET_TOL, min_n=_MIN_N):
    """Pure band-comparison + verdict (D-S9/N4/G2). ``sc``/``gs`` are long DataFrames with ``dist``
    and ``xt_gk`` columns. Returns ``(bands_out, verdict)``:

    * per band: SC/GS mean + n + offset (SC minus GS); a band is **powered** iff both n >= ``min_n``.
    * verdict: ``insufficient_overlap`` (no powered band); ``within_tolerance`` (all powered bands
      within ``offset_tol``); ``escalate_or_correctable_artifact`` (out-of-tol but UNIFORM across
      bands -> candidate measurement artifact, evidence-gated); ``escalate`` (out-of-tol, non-uniform
      -> genuine football, document, do NOT auto-conform).

    No I/O, no side effects -- unit-testable on synthetic arrays."""
    bands_out, powered_offsets = [], []
    for lo, hi in bands:
        s = sc.loc[(sc["dist"] >= lo) & (sc["dist"] < hi), "xt_gk"]
        g = gs.loc[(gs["dist"] >= lo) & (gs["dist"] < hi), "xt_gk"]
        n_sc, n_gs = len(s), len(g)
        powered = n_sc >= min_n and n_gs >= min_n
        offset = float(s.mean() - g.mean()) if powered else float("nan")
        bands_out.append(
            {
                "lo": lo,
                "hi": hi,
                "n_sc": n_sc,
                "n_gs": n_gs,
                "sc_mean": float(s.mean()) if n_sc else None,
                "gs_mean": float(g.mean()) if n_gs else None,
                "offset": offset,
                "powered": powered,
            }
        )
        if powered:
            powered_offsets.append(offset)
    if not powered_offsets:
        verdict = "insufficient_overlap"
    elif all(abs(o) <= offset_tol for o in powered_offsets):
        verdict = "within_tolerance"
    elif (max(powered_offsets) - min(powered_offsets)) <= offset_tol:  # same offset every band
        verdict = "escalate_or_correctable_artifact"
    else:
        verdict = "escalate"
    return bands_out, verdict


#: One score shard per match: (provider, dist, xt_gk, variant). Pinned to the token it travels with.
_SCORE_SHARD_SCHEMA_VERSION = "xtgk-comparability-score-1"


def _score_match(provider, actions, frames, xt) -> pd.DataFrame:
    """One match's (provider, dist, xt_gk, variant) rows for its in-scope scored GK distributions.

    The per-match body the streaming ``_collect`` loop used to inline. It is now the ``work`` of a
    resume-before-load ``for_each`` pass (ADR-052 D14): a match that raises here is a RECORDED failure
    (never fatal to the gate, the old ``try/except``'s intent), and its shard is simply absent.
    ``add_xt_gk`` needs frames, so the scoring pass is a full load.
    """
    out = add_xt_gk(actions, frames, xt)  # type: ignore[reportArgumentType]
    scored = out[out["xt_gk"].notna()].copy()
    dist = np.hypot(
        scored["end_x"].to_numpy(float) - scored["start_x"].to_numpy(float),
        scored["end_y"].to_numpy(float) - scored["start_y"].to_numpy(float),
    )
    return pd.DataFrame(
        {
            "provider": provider,
            "dist": dist,
            "xt_gk": scored["xt_gk"].to_numpy(float),
            "variant": scored.get("xt_gk_completion_variant"),
        }
    )


def main() -> int:
    import tempfile

    ap = argparse.ArgumentParser()
    ap.add_argument("--gs-provider", default="gradientsports", help="a native-completion (gs-variant) provider")
    ap.add_argument("--max-per-provider", type=int, default=6)
    ap.add_argument("--tracking-limit", type=int, default=999999)
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="ADR-068: persist each downloaded pining artifact here and reuse it (else "
        "$SILLY_KICKS_CORPUS_CACHE_DIR). The events-only fit pass and the full-load scoring pass "
        "share the cache, so a match is fetched once, not twice.",
    )
    ap.add_argument(
        "--shard-root",
        default=None,
        help="where the resumable per-match count/score shards live (default: a stable temp dir out "
        "of the repo). Pass a persistent path to resume an interrupted owner run.",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="where comparability_report.json is written (default docs/research/xtgk_comparability).",
    )
    ap.add_argument(
        "--allow-failed",
        action="store_true",
        help="fit the shared xT surface without matches that FAILED the events-only count pass (recorded).",
    )
    ap.add_argument(
        "--allow-unmeasured",
        action="store_true",
        help="admit unmeasured SkillCorner matches into the events-only xT fit (recorded).",
    )
    args = ap.parse_args()

    cache_dir = resolve_cache_dir(args.cache_dir)
    shard_root = Path(args.shard_root) if args.shard_root else Path(tempfile.gettempdir()) / "xtgk_comparability_shards"
    providers = [args.gs_provider, "skillcorner"]
    refs, load = pining_source(
        providers=providers,
        max_per_provider=args.max_per_provider,
        tracking_limit=args.tracking_limit,
        cache_dir=cache_dir,
    )
    if not refs:
        print("no matches loaded", file=sys.stderr)
        return 1

    # One shared, FROZEN xT grid fit on the combined corpus -> both providers scored on the SAME grid
    # (a fair scale comparison; this diagnostic is not a leakage-sensitive model eval). xT is
    # event-only and its per-match zone counts are additive (ADR-102), so the fit is a resumable
    # EVENTS-ONLY count pass reduced by fit_from_counts (byte-identical to a pooled fit) -- no frames
    # loaded, no OOM, no double fetch. The SkillCorner leg routes through the admission gate; its
    # verdict artifact is REQUIRED here (this pass is owner-run after commit 2).
    print("=== fitting a shared frozen xT grid (events-only count pass) ===", flush=True)
    ev_load, admission = events_only_loader(refs, cache_dir=cache_dir, allow_unmeasured=args.allow_unmeasured)
    fit_res = xt_count_pass(
        refs,
        key=lambda r: r.key,
        load_actions=lambda r: ev_load(r).actions,
        shard_root=shard_root / "xt_fit_shards",
        token_inputs={"fit_corpus": sorted(f"{r.provider}__{r.match_id}" for r in refs)},
        l=16,
        w=12,
    )
    xt, xt_prov = fit_xt_from_count_pass(fit_res, l=16, w=12, allow_failed=args.allow_failed, admission=admission)

    # Score BOTH providers in ONE resume-before-load pass; each shard carries its provider, so the
    # combine splits SC/GS by the `provider` column. add_xt_gk needs frames -> full load.
    print("=== scoring both providers (add_xt_gk) ===", flush=True)
    res = for_each(
        refs,
        key=lambda r: r.key,
        load=load,
        work=lambda item: _score_match(item.provider, item.actions, item.frames, xt),
        shard_root=shard_root / "score_shards",
        token_inputs={
            "fit_corpus": sorted(f"{r.provider}__{r.match_id}" for r in refs),
            "counts_digest": xt_prov.counts_digest,
            "tracking_limit": args.tracking_limit,
            "score_schema": _SCORE_SHARD_SCHEMA_VERSION,
        },
        tag="xtgk_comparability",
        label="match",
    )
    if res.failures:
        # A single bad match must not kill the gate (the old try/except's intent): report, don't abort.
        print(f"{len(res.failures)} match(es) failed add_xt_gk (reported, not fatal): {res.failures}", file=sys.stderr)

    # Combined from THIS PASS'S shard keys (no partition surface; `shard_keys` skips excluded markers).
    parts = [f for f in (pd.read_parquet(shard_path(res.shard_dir, k)) for k in res.shard_keys) if len(f)]
    scored = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["provider", "dist", "xt_gk"])
    from silly_kicks.id_compat import ids_match

    sc = scored[ids_match(scored["provider"], "skillcorner")] if len(scored) else scored
    gs = scored[ids_match(scored["provider"], args.gs_provider)] if len(scored) else scored

    print("\n=== per-band SC-vs-GS xt_gk comparison ===", flush=True)
    bands_out, verdict = compare_xtgk_distributions(sc, gs)
    for b in bands_out:
        flag = "" if b["powered"] else "  [UNDER-POWERED]"
        sc_m = b["sc_mean"] if b["sc_mean"] is not None else float("nan")
        gs_m = b["gs_mean"] if b["gs_mean"] is not None else float("nan")
        print(
            f"  dist [{b['lo']:.0f},{b['hi']:.0f}): SC mean={sc_m:.4f} (n={b['n_sc']})  "
            f"GS mean={gs_m:.4f} (n={b['n_gs']})  offset={b['offset']:+.4f}{flag}",
            flush=True,
        )
    print(f"\nVERDICT: {verdict}  (offset tol {_OFFSET_TOL}, min_n {_MIN_N})", flush=True)
    print(
        "  within_tolerance -> pool directly; escalate -> document the difference, do NOT auto-conform "
        "SC to GS (default); correctable affine on xt_gk only with evidence the offset is a measurement "
        "artifact uniform across ALL bands (G2).",
        flush=True,
    )

    report = {
        "verdict": verdict,
        "offset_tol": _OFFSET_TOL,
        "min_n": _MIN_N,
        "gs_provider": args.gs_provider,
        "bands": bands_out,
        "n_sc_total": len(sc),
        "n_gs_total": len(gs),
        # Fit provenance: the shared surface, the admission artifact it consulted, and any unmeasured
        # SkillCorner matches admitted under --allow-unmeasured (spec sections 4.4 / 8).
        "counts_digest": xt_prov.counts_digest,
        "admission_digest": xt_prov.admission_digest,
        "unmeasured_admitted": list(xt_prov.unmeasured_admitted),
        "n_fit_matches": len(xt_prov.fit_keys),
        "n_scored_matches": len(res.shard_keys),
        "n_failed": len(res.failures),
    }
    default_out = Path(__file__).resolve().parent.parent / "docs" / "research" / "xtgk_comparability"
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparability_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir / 'comparability_report.json'}\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
