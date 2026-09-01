"""TF-60 §9 in-possession keeper depth distribution (context probe, local fixtures).

For each committed real-tracking slice: orient via resolve_defended_goals (ADR-055, never team
identity), resolve possession via infer_ball_carrier, and measure the IN-POSSESSION team's
keeper goal-relative depth (distance from its OWN goal) in the committed-forward domain (ball
past halfway). Reports the fraction above GRID_X_MAX=30 m. See README.md for the caveats
(sportec = trustworthy; skillcorner = FOV-biased; metrica = discarded derived-GK mislabel).

Run from the repo root:  python docs/research/tf60_ghost_gk_in_possession_validity/probe_depth_distribution.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from silly_kicks.id_compat import canonical_id
from silly_kicks.tracking import infer_ball_carrier, resolve_defended_goals

GRID_X_MAX = 30.0
MIN_BALL_ADVANCE_M = 52.5

FIXTURES = {
    "sportec_slim (full-tracking, TRUSTWORTHY)": "tests/datasets/tracking/action_context_slim/sportec_slim.parquet",
    "skillcorner_slim (FOV-biased)": "tests/datasets/tracking/action_context_slim/skillcorner_slim.parquet",
    "metrica_slim (derived-GK; see README)": "tests/datasets/tracking/action_context_slim/metrica_slim.parquet",
}


def _load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "__kind" in df.columns:
        df = df[df["__kind"] == "frame"].copy()
    for c in ("is_ball", "is_goalkeeper"):
        df[c] = df[c].astype("boolean").fillna(False)
    return df.reset_index(drop=True)


def _measure(path: str) -> np.ndarray:
    frames = _load(path)
    gmap = resolve_defended_goals(frames)
    carrier = infer_ball_carrier(frames)[["game_id", "period_id", "frame_id", "ball_carrier_team_id"]]
    poss = {
        (canonical_id(g), canonical_id(p), canonical_id(f)): canonical_id(t)
        for g, p, f, t in carrier.itertuples(index=False)
        if pd.notna(t)
    }
    ball = frames[frames["is_ball"]].drop_duplicates(["game_id", "period_id", "frame_id"])
    ball_x = {
        (canonical_id(r.game_id), canonical_id(r.period_id), canonical_id(r.frame_id)): r.x
        for r in ball.itertuples(index=False)
    }
    gk = frames[frames["is_goalkeeper"]]
    committed = []
    for (g, p, f), grp in gk.groupby(["game_id", "period_id", "frame_id"]):
        key = (canonical_id(g), canonical_id(p), canonical_id(f))
        a = poss.get(key)
        if a is None:
            continue
        arow = grp[grp["team_id"].map(canonical_id) == a]
        if len(arow) != 1:
            continue
        g_a = gmap.get(g, p, a, allow_guess=True)
        bx = ball_x.get(key)
        if g_a is None or bx is None or pd.isna(bx):
            continue
        gr_x = float(arow["x"].iloc[0]) if float(g_a) == 0.0 else 105.0 - float(arow["x"].iloc[0])
        ball_own = float(bx) if float(g_a) == 0.0 else 105.0 - float(bx)
        if ball_own > MIN_BALL_ADVANCE_M:
            committed.append(gr_x)
    return np.asarray(committed, dtype=float)


def main() -> None:
    warnings.filterwarnings("ignore")
    for label, path in FIXTURES.items():
        a = _measure(path)
        if len(a) == 0:
            print(f"{label:<45} no committed-forward in-poss keeper frames")
            continue
        print(
            f"{label:<45} n={len(a):>3} med={np.median(a):>5.1f} "
            f"p95={np.percentile(a, 95):>5.1f} max={a.max():>5.1f} %>30m={100 * np.mean(a > GRID_X_MAX):>5.1f}"
        )


if __name__ == "__main__":
    main()
