"""TF-60 §9 controlled-extrapolation probe (decisive, local, committed fixtures only).

Rigidly translate a clean full-tracking scene upfield by Delta metres (a physically-coherent
progressively-higher line; velocities + relative geometry preserved) and serve the shipped
`default` GhostGkModel. The home keeper defends x=0, so its goal-relative x rises by ~Delta.
The prediction TRACKS the real keeper up to ~30 m then SATURATES at the trained-label ceiling
(GRID_X_MAX=30) -- see README.md.

Run from the repo root:  python docs/research/tf60_ghost_gk_in_possession_validity/probe_extrapolation.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from silly_kicks.id_compat import canonical_id
from silly_kicks.tracking import derive_velocities, resolve_defended_goals, serve_ghost_gk_positions
from silly_kicks.tracking.preprocess import smooth_frames

FIXTURE = "tests/datasets/tracking/action_context_slim/sportec_slim.parquet"


def _load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "__kind" in df.columns:
        df = df[df["__kind"] == "frame"].copy()
    for c in ("is_ball", "is_goalkeeper"):
        df[c] = df[c].astype("boolean").fillna(False)
    return df.reset_index(drop=True)


def main() -> None:
    warnings.filterwarnings("ignore")
    base = _load(FIXTURE)
    gmap0 = resolve_defended_goals(base)
    home = next(t for (g, p, t), e in gmap0.resolved.items() if str(p) == "1" and float(e) == 0.0)
    print(f"home_team_id (defends x=0): {home}\n")
    print(f"{'Delta':>6} | {'actual GK gr_x (mean/max)':>26} | {'PRED ghost gr_x (mean/max)':>28} | out_of_box")
    print("-" * 90)
    for delta in (0, 5, 10, 15, 20, 25):
        f = base.copy()
        f["x"] = np.clip(f["x"].to_numpy(dtype=float) + delta, 0.0, 105.0)
        f = derive_velocities(smooth_frames(f))
        served = serve_ghost_gk_positions(f, home_team_id=home, model=None)
        gk = f[f["is_goalkeeper"] & (f["team_id"].map(canonical_id) == canonical_id(home))]
        actual = gk["x"].to_numpy(dtype=float)
        svh = served[served["gk_team_id"].map(canonical_id) == canonical_id(home)]
        pred = svh["ghost_gr_x"].to_numpy(dtype=float)
        oob = svh["ghost_out_of_box"].astype(bool).mean() * 100
        print(
            f"{delta:>6} | mean={actual.mean():>6.1f} max={actual.max():>5.1f}          | "
            f"mean={pred.mean():>6.1f} max={pred.max():>5.1f}            | {oob:>6.1f}%"
        )


if __name__ == "__main__":
    main()
