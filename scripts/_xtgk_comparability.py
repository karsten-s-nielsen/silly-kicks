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

from _loader_pining import load_matches

from silly_kicks.tracking.features import add_xt_gk
from silly_kicks.xthreat import ExpectedThreat

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


def _collect(providers, max_per_provider, tracking_limit, xt):
    """Return a long df of (provider, dist, xt_gk) for in-scope scored GK distributions."""
    rows = []
    for prov, mid, actions, frames, _home in load_matches(
        providers=providers, max_per_provider=max_per_provider, tracking_limit=tracking_limit
    ):
        try:
            out = add_xt_gk(actions, frames, xt)  # type: ignore[reportArgumentType]
        except Exception as exc:  # a single bad match shouldn't kill the gate
            print(f"  {prov}/{mid}: add_xt_gk failed ({type(exc).__name__}: {exc})", flush=True)
            continue
        scored = out[out["xt_gk"].notna()].copy()
        dist = np.hypot(
            scored["end_x"].to_numpy(float) - scored["start_x"].to_numpy(float),
            scored["end_y"].to_numpy(float) - scored["start_y"].to_numpy(float),
        )
        rows.append(
            pd.DataFrame(
                {
                    "provider": prov,
                    "dist": dist,
                    "xt_gk": scored["xt_gk"].to_numpy(float),
                    "variant": scored.get("xt_gk_completion_variant"),
                }
            )
        )
        print(f"  {prov}/{mid}: {len(scored)} scored GK distributions", flush=True)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["provider", "dist", "xt_gk"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gs-provider", default="gradientsports", help="a native-completion (gs-variant) provider")
    ap.add_argument("--max-per-provider", type=int, default=6)
    ap.add_argument("--tracking-limit", type=int, default=999999)
    args = ap.parse_args()

    # One shared, FROZEN xT grid fit on the combined corpus -> both providers scored on the SAME grid
    # (a fair scale comparison; this diagnostic is not a leakage-sensitive model eval).
    print("=== fitting a shared frozen xT grid on the combined corpus ===", flush=True)
    combined = []
    for _prov, _mid, actions, _frames, _home in load_matches(
        providers=[args.gs_provider, "skillcorner"],
        max_per_provider=args.max_per_provider,
        tracking_limit=10,
    ):
        combined.append(actions)
    xt = ExpectedThreat(l=16, w=12)
    xt.fit(pd.concat(combined, ignore_index=True))

    print("=== scoring SkillCorner ===", flush=True)
    sc = _collect(["skillcorner"], args.max_per_provider, args.tracking_limit, xt)
    print(f"=== scoring {args.gs_provider} ===", flush=True)
    gs = _collect([args.gs_provider], args.max_per_provider, args.tracking_limit, xt)

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
    }
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "research" / "xtgk_comparability"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparability_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir / 'comparability_report.json'}\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
