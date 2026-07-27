"""Maintainer driver: derive Layer 3's headroom threshold from `openGoal`'s observed range.

TF-19 sign-off package S6. The derivation duty says: "state `openGoal`'s units and observed range
FIRST (a reader cannot currently tell whether 0.02 is generous or unreachable), then set the
threshold as a stated fraction of that range."

Units are settled in code: `_open_goal_fraction` returns the "Unobstructed share of the goal mouth
from the ball" -- a dimensionless fraction in [0, 1], NaN when the ball is on/behind the goal line.
This script supplies the OBSERVED range and multiplies by the pre-committed
`LAYER3_HEADROOM_RANGE_FRACTION`.

BOUNDARY, enforced below rather than merely described: this measures the MARGINAL distribution of
the SHIPPED feature. It does NOT run the ghost substitution and does NOT compute any delta -- that
is Layer 3's probe, which is PR-3b. The distinction is what keeps "derive the constant" from
silently becoming "run the experiment".

Usage (on the box, scripts/ on sys.path, pining token in env):
  python scripts/derive_opengoal_range.py --out <DIR> [--providers gradientsports] \
      [--max-per-provider N] [--tracking-limit N]

--help is dep-light: args are parsed before any loader / silly_kicks import (house pattern).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def summarise(open_goal_values) -> dict:
    """Pure: the reported distribution + the derived threshold. No I/O, no loader."""
    import numpy as np

    from silly_kicks.gkdv._validate import LAYER3_HEADROOM_RANGE_FRACTION

    v = np.asarray(open_goal_values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no finite openGoal values -- refusing to derive a threshold from nothing")
    lo, hi = float(v.min()), float(v.max())
    observed_range = hi - lo
    return {
        "n": int(v.size),
        "units": "dimensionless open-goal-mouth fraction, constructively in [0, 1]",
        "min": lo,
        "p01": float(np.quantile(v, 0.01)),
        "median": float(np.median(v)),
        "p99": float(np.quantile(v, 0.99)),
        "max": hi,
        "observed_range": observed_range,
        "range_fraction": float(LAYER3_HEADROOM_RANGE_FRACTION),
        "layer3_headroom_threshold": observed_range * float(LAYER3_HEADROOM_RANGE_FRACTION),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output directory for opengoal_distribution.json")
    ap.add_argument("--providers", default="gradientsports")
    ap.add_argument("--max-per-provider", type=int, default=None)
    ap.add_argument("--tracking-limit", type=int, default=None)
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; artifact is marked)")
    args = ap.parse_args()

    # The boundary, enforced: this derivation must never import the ghost-substitution engine.
    from scripts._provenance import git_provenance, require_clean_tree

    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)

    if "silly_kicks.gkdv._engine" in sys.modules:
        raise RuntimeError(
            "derivation must not import the ghost-substitution engine -- measuring the marginal "
            "distribution is NOT running Layer 3's probe (spec S6). A raise, not an assert: "
            "asserts vanish under -O, and this boundary must hold in every run."
        )

    import numpy as np

    from scripts._loader_pining import load_matches
    from silly_kicks.tracking._xshot_occurrence import prepare_xshot_training_data

    values: list[float] = []
    # load_matches yields (provider, match_id, ACTIONS, FRAMES, home_team_id) -- actions FIRST.
    for _provider, _match_id, actions, frames, home_team_id in load_matches(
        providers=args.providers.split(","),
        max_per_provider=args.max_per_provider,
        tracking_limit=args.tracking_limit,
    ):
        # Returns (features, labels, groups); only the FEATURES are read here -- this measures the
        # marginal distribution of a shipped feature, it does not train or probe anything.
        feats, _labels, _groups = prepare_xshot_training_data(
            frames,
            actions,
            home_team_id=home_team_id,  # type: ignore[arg-type]  -- loader yields `object`
        )
        if "openGoal" in feats.columns:
            values.extend(np.asarray(feats["openGoal"], dtype=float).tolist())

    out = summarise(values)
    out["run_commit"] = prov["commit"]
    out["run_tree_dirty"] = prov["dirty"]
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "opengoal_distribution.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
