"""TF-19 A+2 Task 3: the Layer-1 responsiveness verdict + paired-vector controls.

The verdict is the shipped idiom `gk_med >= RATIO * max(nd_med, placebo_p95)` with the same
`n_domain`/NaN `arm_unscoreable` short-circuits as Layer 0. `paired_vector_controls` displaces the
nearest defending-team outfielder (+ r random ones) by the SAME per-frame (imp - actual) vector the
keeper moved -- so the placebo band is a genuine displaced-teammate counterfactual, asserted at the
EXACT landing coordinate (a vacuous control would displace nothing / a different vector).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.gkdv._probe import (
    PHYSICS_ARM_PROBE_RATIO,
    layer1_responsiveness_verdict,
    paired_vector_controls,
)


def test_responsive_when_gk_beats_ratio_times_controls():
    assert PHYSICS_ARM_PROBE_RATIO == 2.0
    v = layer1_responsiveness_verdict(gk_med=0.30, nd_med=0.10, placebo_p95=0.12, n_domain=300)
    assert v == "responsive"  # 0.30 >= 2.0 * max(0.10, 0.12)


def test_not_responsive_when_flat():
    v = layer1_responsiveness_verdict(gk_med=0.20, nd_med=0.15, placebo_p95=0.12, n_domain=300)
    assert v == "not_responsive"  # 0.20 < 2.0 * 0.15


def test_thin_domain_is_unscoreable_not_flat():
    v = layer1_responsiveness_verdict(gk_med=0.30, nd_med=0.10, placebo_p95=0.12, n_domain=3)
    assert v == "arm_unscoreable"
    assert v != "not_responsive"


def test_velocity_less_arm_is_unscoreable_not_flat():
    v = layer1_responsiveness_verdict(gk_med=np.nan, nd_med=np.nan, placebo_p95=np.nan, n_domain=300)
    assert v == "arm_unscoreable"
    assert v != "not_responsive"


def _frames_two_defenders():
    # keeper (100) + two defending outfielders (101 near, 102 far) + one opponent (200) + ball.
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1, 1],
            "period_id": [1, 1, 1, 1, 1],
            "frame_id": [10, 10, 10, 10, 10],
            "team_id": [1, 1, 1, 2, None],
            "player_id": [100, 101, 102, 200, None],
            "is_ball": [False, False, False, False, True],
            "is_goalkeeper": [True, False, False, False, False],
            "x": [4.0, 20.0, 60.0, 50.0, 40.0],
            "y": [30.0, 40.0, 34.0, 34.0, 34.0],
        }
    )


def _targets_one_frame():
    # keeper displaced by (imp - actual) = (-4, +4).
    return pd.DataFrame(
        {
            "game_id": [1],
            "period_id": [1],
            "frame_id": [10],
            "defending_team_id": [1],
            "actual_x": [4.0],
            "actual_y": [30.0],
            "imp_x": [0.0],
            "imp_y": [34.0],
        }
    )


def _moved_defenders(control, frames):
    """player_ids of DEFENDING (team-1) outfielders whose (x,y) changed vs `frames`."""
    orig = frames.set_index("player_id")[["x", "y"]]
    d = control[(control["team_id"] == 1) & ~control["is_goalkeeper"].astype(bool)]
    return {
        int(pid)
        for pid in d["player_id"]
        if (d[d["player_id"] == pid]["x"].iloc[0], d[d["player_id"] == pid]["y"].iloc[0])
        != (orig.loc[pid, "x"], orig.loc[pid, "y"])
    }


def test_paired_vector_controls_move_ONE_player_each_by_the_keeper_vector():
    # Parent idiom (`_model_eval.py`): ONE player per control -- the NEAREST defender (nd) alone, and
    # r SINGLE-outfielder placebo replicates -- so nd and each placebo are DISTINCT single-player
    # quantities (otherwise `max(nd_med, placebo_p95)` in the Layer-1 verdict is decorative).
    frames = _frames_two_defenders()
    controls = paired_vector_controls(frames, _targets_one_frame(), r=2, rng=np.random.default_rng(0))
    assert set(controls) == {"nearest", "placebo_0", "placebo_1"}

    near = controls["nearest"]
    assert float(near[near["player_id"] == 101]["x"].iloc[0]) == 16.0  # 20 - 4 (nearest defender moved)
    assert float(near[near["player_id"] == 101]["y"].iloc[0]) == 44.0  # 40 + 4
    assert _moved_defenders(near, frames) == {101}  # ONLY the nearest, not 102
    assert float(near[near["player_id"] == 100]["x"].iloc[0]) == 4.0  # keeper untouched
    assert float(near[near["player_id"] == 200]["x"].iloc[0]) == 50.0  # opponent untouched

    for k in (0, 1):
        assert len(_moved_defenders(controls[f"placebo_{k}"], frames)) == 1  # exactly ONE outfielder

    # PURE -- `frames` never mutated.
    assert float(frames[frames["player_id"] == 101]["x"].iloc[0]) == 20.0
