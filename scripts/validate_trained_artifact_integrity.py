"""Trained-artifact integrity gate (ADR-126).

The non-vacuous replacement for the campaign's reproduction-free floor. For a trained model family
it composes TWO layers into a GO/NO-GO verdict:

1. **Reproduction tripwire** (:func:`reproduces`) -- a fresh extraction+fit on the artifact's declared
   corpus must reproduce its shipped ``metrics.json`` headline (``n_rows``/``n_positive`` exact,
   candidate ``pr_auc``/``positive_rate`` within a relative tolerance). A mismatch is a HARD STOP.
2. **Construct-validity on the ABSOLUTE metrics** -- a physically-DERIVED base-rate plausibility band
   (:func:`silly_kicks.tracking._artifact_integrity.base_rate_plausible`), ``pr_auc > positive_rate``
   (lift), ``brier < base_rate_brier`` (calibration), and enough usable folds.

Why both: the committed xShot defect (22% positive rate) PASSED ``pr_auc > base_rate`` -- only the
base-rate band + the reproduction tripwire catch it. Certifying "matches the committed ``pr_auc``"
alone would re-bless a defective baseline.

CLI (``main``) wires the real audit over campaign families with clean-tree provenance (ADR-037); the
full re-extract+re-fit pass is opt-in/``e2e`` (the DGX/HF run), while the reproduction + verdict logic
below is the CI-safe teeth.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from silly_kicks.tracking._artifact_integrity import MODEL_INTEGRITY_TOL, base_rate_plausible

_MIN_USABLE_FOLDS = 5


def _shipped_candidate(metrics: Mapping[str, Any]) -> str:
    """The candidate key to compare on: the shipped variant, else the first candidate."""
    sv = metrics.get("shipped_variant")
    if sv and sv in metrics.get("candidates", {}):
        return str(sv)
    return next(iter(metrics["candidates"]))


def reproduces(shipped: Mapping[str, Any], fresh: Mapping[str, Any], *, rel_tol: float = 0.05) -> bool:
    """True iff ``fresh`` reproduces ``shipped``'s headline within tolerance.

    ``n_rows`` and ``n_positive`` must match exactly (same extraction population + label); the shipped
    candidate's ``pr_auc`` and ``positive_rate`` must agree within ``rel_tol`` (relative).
    """
    if shipped.get("n_rows") != fresh.get("n_rows"):
        return False
    if shipped.get("n_positive") != fresh.get("n_positive"):
        return False
    cand = _shipped_candidate(shipped)
    try:
        sm = shipped["candidates"][cand]["metrics"]
        fm = fresh["candidates"][cand]["metrics"]
    except KeyError:
        return False
    for key in ("pr_auc", "positive_rate"):
        a, b = sm.get(key), fm.get(key)
        if a is None or b is None or a != a or b != b:  # missing or NaN
            return False
        if abs(a - b) > rel_tol * max(abs(a), 1e-9):
            return False
    return True


def verdict(
    family: str,
    shipped: Mapping[str, Any],
    fresh: Mapping[str, Any],
    *,
    rel_tol: float = 0.05,
) -> dict:
    """Compose the layered gate into a flag dict (no I/O; provenance attached by ``main``).

    ``fresh`` additionally carries ``n_events`` (shot/cross count), ``fps`` and ``horizon_s`` from its
    own extraction so the base-rate band is derived, not fixed.
    """
    cand = _shipped_candidate(fresh)
    m = fresh["candidates"][cand]["metrics"]
    pr = m.get("pr_auc")
    base = m.get("positive_rate")
    brier = m.get("brier")
    base_brier = m.get("base_rate_brier")
    tol = MODEL_INTEGRITY_TOL.get(family, 1.5)
    base_ok = base is not None and base_rate_plausible(
        base,
        n_events=fresh["n_events"],
        horizon_s=fresh.get("horizon_s", 1.0),
        fps=fresh["fps"],
        n_rows=fresh["n_rows"],
        tol=tol,
    )
    return {
        "family": family,
        "candidate": cand,
        "reproduced": reproduces(shipped, fresh, rel_tol=rel_tol),
        "base_rate_ok": bool(base_ok),
        "lift_ok": bool(pr is not None and base is not None and pr > base),
        "calibration_ok": bool(brier is not None and base_brier is not None and brier < base_brier),
        "enough_folds": bool(m.get("n_usable_folds", 0) >= _MIN_USABLE_FOLDS),
        "positive_rate": base,
        "pr_auc": pr,
    }


def is_go(v: Mapping[str, Any]) -> bool:
    """GO iff every gate in the verdict passed."""
    return bool(v["reproduced"] and v["base_rate_ok"] and v["lift_ok"] and v["calibration_ok"] and v["enough_folds"])


def _load(p: str | Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    from scripts._input_contract import declare_inputs
    from scripts._provenance import git_provenance, require_clean_tree

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True, help="model family, e.g. xshot_occurrence")
    ap.add_argument("--shipped", required=True, help="path to the SHIPPED metrics.json")
    ap.add_argument("--fresh", required=True, help="path to a FRESH re-fit metrics.json (same corpus)")
    ap.add_argument("--n-events", type=int, required=True, help="shot/cross count in the fresh corpus")
    ap.add_argument("--fps", type=float, required=True, help="frames per second (or n_rows/duration_s)")
    ap.add_argument("--horizon-s", type=float, default=1.0)
    ap.add_argument("--rel-tol", type=float, default=0.05)
    ap.add_argument("--out", required=True, help="verdict metrics.json output path")
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; stamped)")
    args = ap.parse_args(argv)

    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    shipped = _load(args.shipped)
    fresh = dict(_load(args.fresh), n_events=args.n_events, fps=args.fps, horizon_s=args.horizon_s)
    v = verdict(args.family, shipped, fresh, rel_tol=args.rel_tol)
    out = {
        "verdict": v,
        "go": is_go(v),
        "input_contract": declare_inputs(
            driver="validate_trained_artifact_integrity",
            family=args.family,
            rel_tol=args.rel_tol,
        ),
        # Canonical research-artifact provenance (ADR-037): the TOP-LEVEL `run_commit` /
        # `run_tree_dirty` shape that `test_artifact_provenance_output.py` enforces across
        # docs/research. `git_provenance()` returns the raw `commit`/`dirty` keys, which the gate
        # does NOT read -- splatting them (`**prov`) emits an untraceable artifact. Map them here,
        # exactly as the sibling drivers (build_rq_pass_scores, derive_opengoal_range, ...) do.
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov["tree_state"],
        "platform": prov["platform"],
        "machine": prov["machine"],
    }
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"family": args.family, "go": out["go"], "verdict": v}, indent=2))


if __name__ == "__main__":
    main()
