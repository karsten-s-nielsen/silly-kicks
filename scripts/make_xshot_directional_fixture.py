#!/usr/bin/env python
"""Build the frozen directional feature-vector fixture for the CI liveness tripwire (PR-S80 N2).

Extracts xS features from the committed slim real-provider fixtures and freezes IN-DOMAIN
NEAR-goal (small `r` -> expected-positive) and FAR (large `r` -> expected-negative) FEATURE ROWS.

Freezing at the FEATURE-VECTOR layer (not raw frames) makes the CI quality test schema-robust and
arch-robust (it asserts a ranking, AUC, not exact values). The label is the expected class by goal
distance -- a live model must rank near > far; a dead/constant model cannot. Real player geometry
(openGoal, GK/defender distances) is preserved because the rows come from real frames.

Run once to (re)generate the committed parquet:
    python scripts/make_xshot_directional_fixture.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from silly_kicks.tracking import _xshot_occurrence as xs

REPO = Path(__file__).resolve().parents[1]
SLIM = REPO / "tests" / "datasets" / "tracking" / "action_context_slim"
OUT = REPO / "tests" / "datasets" / "tracking" / "xshot_directional" / "frozen_rows.parquet"

_KEEP = {
    "game_id", "period_id", "frame_id", "time_seconds", "frame_rate", "player_id", "team_id",
    "is_ball", "is_goalkeeper", "x", "y", "z", "speed", "speed_source", "ball_state",
    "team_attacking_direction", "confidence", "visibility", "source_provider",
}  # fmt: skip


def _load(provider: str) -> pd.DataFrame:
    df = pd.read_parquet(SLIM / f"{provider}_slim.parquet")
    frames = df[df["__kind"] == "frame"].drop(columns=["__kind"]).reset_index(drop=True)
    frames = frames[[c for c in frames.columns if c in _KEEP]].copy()
    # PR 5: carry the REAL speed through instead of zeroing velocity.
    #
    # This previously set vx = vy = 0, so `speed = hypot(bvx, bvy)` was CONSTANT ZERO on every
    # frozen row -- the same degeneracy PR-S118 found and fixed in the xCross fixture ("it zeroed
    # the model's #1 feature, so the model could not respond"), which was repaired there and left
    # in place here. A feature pinned to one value cannot contribute to a ranking, and it routes
    # down XGBoost's zero branch rather than the missing branch, so the gate was scoring a
    # counterfactual the model never sees in production.
    #
    # `extract_xshot_features` uses only the MAGNITUDE (`hypot`), never the direction, so putting
    # the observed speed entirely in vx reproduces the real `speed` feature exactly.
    frames["vx"] = frames["speed"].astype(float) if "speed" in frames.columns else 0.0
    frames["vy"] = 0.0
    return frames


def main() -> None:
    rows = []
    for prov in ("sportec", "skillcorner", "metrica"):
        if not (SLIM / f"{prov}_slim.parquet").exists():
            print(f"skip {prov}: not committed")
            continue
        frames = _load(prov)
        outfield = frames[~frames["is_ball"].astype(bool)]
        means = outfield.groupby("team_id")["x"].mean()
        if means.empty:
            continue
        def_team = means.idxmin()  # team defending the x=0 goal (lower mean x)
        for _, g in frames.groupby(["game_id", "period_id", "frame_id"], dropna=False):
            feat = xs.extract_xshot_features(g, gk_team_id=def_team, goal_x=0.0)
            r = feat["r"].iloc[0]
            if np.isnan(r):
                continue
            rec = feat.iloc[0].to_dict()
            rec["r_val"] = float(r)
            rec["provider"] = prov
            rows.append(rec)
    df = pd.DataFrame(rows).dropna(subset=["r_val"])
    # PR 5: constrain to the model's TRAINED DOMAIN before splitting near/far.
    #
    # `prepare_xshot_training_data(attacking_third_only=True)` keeps only frames with the ball
    # within `_ATTACKING_THIRD_M` (35 m) of the attacked goal, so anything beyond that was never
    # seen in training and the model's output there is extrapolation, not prediction. The previous
    # global `nlargest` put the whole FAR class at r ~= 101 m -- roughly 3x outside the domain --
    # so the liveness gate was scoring undefined behaviour. Two statistically equivalent models
    # (pr_auc 0.3514 vs 0.3458, 0.37 SD apart) scored AUC 1.0 and 0.0 on it.
    df = df[df["r_val"] <= xs._ATTACKING_THIRD_M].reset_index(drop=True)
    if len(df) < 40:
        raise SystemExit(f"only {len(df)} in-domain rows (r <= {xs._ATTACKING_THIRD_M}); need >= 40")
    # Quartile split rather than a fixed 8+8: more rows means the gate cannot be flipped by a
    # handful of ties, which is what made the 16-row version unable to separate two equal models.
    lo, hi = df["r_val"].quantile([0.25, 0.75])
    near = df[df["r_val"] <= lo].copy()
    near["label"] = 1
    far = df[df["r_val"] >= hi].copy()
    far["label"] = 0
    frozen = pd.concat([near, far], ignore_index=True)[[*xs.XSHOT_FEATURE_NAMES_FAITHFUL, "label", "provider"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frozen.to_parquet(OUT)
    print(f"Wrote {len(frozen)} rows ({int(frozen['label'].sum())} near/pos) to {OUT}")
    print(
        f"  near r range: {near['r_val'].min():.1f}-{near['r_val'].max():.1f}; "
        f"far r range: {far['r_val'].min():.1f}-{far['r_val'].max():.1f}"
    )


if __name__ == "__main__":
    main()
