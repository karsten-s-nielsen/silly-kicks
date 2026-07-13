"""Spec §3.1 instrument validation: the probe must PASS a planted mixed-dependence
GK-responsive model and FAIL a GK-blind one — under the actual control construction.
A null from an instrument that has never detected a planted signal is uninterpretable."""

import numpy as np
import pandas as pd

from silly_kicks.tracking import _model_eval as me
from tests.tracking._probe_fixtures import planted_model, probe_frames


def _targets(frames, rng, spread=6.0):
    gk = frames[frames["is_goalkeeper"].astype(bool)]
    t = gk[["game_id", "period_id", "frame_id", "x", "y"]].drop_duplicates(subset=["game_id", "period_id", "frame_id"])
    return t.assign(
        target_x=t["x"] - spread * (0.5 + rng.random(len(t))),
        target_y=t["y"] + rng.normal(scale=2.0, size=len(t)),
        ghost_clamped=False,
        ghost_out_of_box=False,
    ).drop(columns=["x", "y"])


def _run(kind, n_frames=150, seed=7):
    # replicate the fixture frames enough times (distinct game_ids) to clear MIN_BAND_N
    base = probe_frames()
    reps = []
    for i in range(n_frames):
        r = base.copy()
        r["game_id"] = f"m{i % 12}"
        r["frame_id"] = r["frame_id"] + 10 * i
        r["time_seconds"] = r["time_seconds"] + 10.0 * i  # carrier hysteresis: no duplicate clocks per game
        reps.append(r)
    frames = pd.concat(reps, ignore_index=True)
    rng = np.random.default_rng(seed)
    targets = _targets(frames, rng)
    out = me.xs_substitution_probe(planted_model(kind), frames, targets, seed=seed)
    # fixture-validity preconditions (M1): the verdict is meaningless if these fail
    assert out["gated_band_n"] >= me.XS_PROBE_MIN_BAND_N, "fixture too small for the registered rule"
    assert out.get("placebo_p95", 0) != 0 or out["verdict"] == "no_valid_placebo"
    return out


def test_mixed_dependence_planted_model_passes():
    out = _run("mixed")
    assert out["gated_band_zero_fraction"] < 1.0
    assert out["verdict"] == "pass", out


def test_gk_blind_model_is_a_clean_interpretable_fail():
    out = _run("gk_blind")
    assert out["verdict"] == "fail", out
    assert out["gated_band_zero_fraction"] == 1.0  # every GK move is a no-op...
    assert out["placebo_p95"] > 0  # ...but the CONTROLS are live
    # together: a clean GK-insensitivity finding, NOT a degenerate measurement (B1/B2)
