#!/usr/bin/env python
"""Build the frozen directional feature-vector fixture for the CI liveness tripwire (PR-S80 N2).

Extracts xS features from the committed slim real-provider fixtures and freezes a handful of
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
    frames["vx"] = 0.0
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
    near = df.nsmallest(8, "r_val").copy()
    near["label"] = 1
    far = df.nlargest(8, "r_val").copy()
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
