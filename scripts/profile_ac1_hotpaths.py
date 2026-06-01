"""Phase-0 local measurement: ghost-GK predict_density scipy vs vectorized + cProfile.

Capture with NO contending processes (the calibrate_* sweep shares this box).
Run: .venv/Scripts/python.exe scripts/profile_ac1_hotpaths.py > _phase0_report.txt 2>&1
Persist _phase0_report.txt BEFORE any cleanup.

  --attribute : cProfile add_ghost_gk + add_elastic_sync on a real provider batch and
                print the callers of pandas _ixs / __getitem__ (Phase-0b attribution).
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time

import numpy as np
import pandas as pd

from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GhostGkModel


def _features(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    X = pd.DataFrame(rng.standard_normal((n, 26)), columns=GHOST_GK_FEATURE_NAMES)
    X["phase"] = rng.integers(0, 3, n).astype(float)
    return X


def _bench() -> None:
    model = GhostGkModel.from_variant("default")
    X = _features(16)  # default model is ~4-5 s/sample; 16 keeps the bench tractable
    # warm
    model.predict_density(X.iloc[:3], kde_backend="scipy")
    model.predict_density(X.iloc[:3], kde_backend="vectorized")
    for backend in ("scipy", "vectorized"):
        t0 = time.perf_counter()
        model.predict_density(X, kde_backend=backend)
        dt = time.perf_counter() - t0
        print(
            f"[bench] {backend:10s}: {dt:.3f}s over {len(X)} samples ({1000 * dt / len(X):.2f} ms/sample)",
            flush=True,
        )
    # cProfile the vectorized path
    pr = cProfile.Profile()
    pr.enable()
    model.predict_density(X, kde_backend="vectorized")
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(20)
    print(s.getvalue(), flush=True)


def _attribute() -> None:
    """Attribute the ~14% pandas-scalar access to specific call sites."""
    from silly_kicks.tracking import derive_team_in_possession, infer_ball_carrier
    from silly_kicks.tracking.features import add_elastic_sync, add_ghost_gk  # type: ignore
    from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames
    from tests.tracking._provider_inputs import load_provider_frames, synthesize_actions

    frames = load_provider_frames("skillcorner")
    if "vx" not in frames.columns:
        frames = derive_velocities(smooth_frames(frames))
    carrier = infer_ball_carrier(frames)
    frames = derive_team_in_possession(frames, carrier)
    actions = synthesize_actions(frames)
    home_team_id = frames["team_id"].dropna().iloc[0]

    pr = cProfile.Profile()
    pr.enable()
    try:
        add_ghost_gk(actions, frames, home_team_id=home_team_id)
    except Exception as exc:
        print(f"[attribute] add_ghost_gk failed ({exc}); continuing", flush=True)
    try:
        add_elastic_sync(actions, frames)
    except Exception as exc:
        print(f"[attribute] add_elastic_sync failed ({exc}); continuing", flush=True)
    pr.disable()
    s = io.StringIO()
    st = pstats.Stats(pr, stream=s)
    st.print_callers("_ixs")
    st.print_callers("__getitem__")
    print(s.getvalue(), flush=True)


def main() -> None:
    if "--attribute" in sys.argv:
        _attribute()
    else:
        _bench()


if __name__ == "__main__":
    main()
