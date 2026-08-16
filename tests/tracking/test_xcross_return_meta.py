"""Guardrails for the additive `return_meta` on prepare_xcross_training_data (keeper-box cycle).

Mirrors the ghost seam (return_meta row-aligned, filtered in lockstep). The change must be
byte-identical on the training path when False, and the meta's clamped ratio must equal what the
scoped gr_x clamp actually produces (so the descriptive split cannot drift from the feature).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import silly_kicks.tracking._geometry as _geo
import silly_kicks.tracking._xcross_attempt as xc
from silly_kicks.spadl import config as spc

_META_COLS = {
    "game_id",
    "period_id",
    "frame_id",
    "time_seconds",
    "team_in_possession",
    "n_off",
    "n_def",
    "n_off_behind",
    "n_def_behind",
    "box_off_def_ratio_clamped",
    "behind_line_box_gr_x",
}


def _match_with_behind_line_defender():
    players = [
        ("A", "A1", 95.0, 10.0, False),  # carrier / ball anchor (wide)
        ("A", "A2", 100.0, 34.0, False),  # attacker in box
        ("B", "B1d", 100.0, 30.0, False),  # defender in box, in front of the line
        ("B", "B2d", 106.0, 34.0, False),  # defender BEHIND the line (gr_x = -1)
        ("B", "Bgk", 104.0, 34.0, True),
        ("ball", None, 95.0, 10.0, False),
    ]
    rows = []
    for fr, t in enumerate([0.0, 0.4, 0.8, 1.2], start=1):
        for team, pid, x, y, gk in players:
            rows.append(
                dict(
                    game_id="g",
                    period_id=1,
                    frame_id=fr,
                    time_seconds=t,
                    team_id=team,
                    player_id=pid,
                    x=x,
                    y=y,
                    vx=1.0,
                    vy=0.0,
                    is_ball=(pid is None),
                    is_goalkeeper=gk,
                    ball_state="alive",
                )
            )
    frames = pd.DataFrame(rows)
    frames["source_provider"] = "test"
    actions = pd.DataFrame(
        {
            "game_id": ["g"],
            "period_id": [1],
            "team_id": ["A"],
            "time_seconds": [0.9],
            "type_id": [spc.actiontype_id["cross"]],
            "result_id": [spc.result_id["success"]],
        }
    )
    return frames, actions


def test_return_meta_false_is_byte_identical_on_the_training_path():
    """Guardrail #3: return_meta=False must be a no-op on (X, y, groups) -- the re-fit is unaffected."""
    frames, actions = _match_with_behind_line_defender()
    default = xc.prepare_xcross_training_data(frames, actions, home_team_id="A")
    assert len(default) == 3  # still a 3-tuple by default
    X0, y0, g0 = default
    X1, y1, g1, _meta = xc.prepare_xcross_training_data(frames, actions, home_team_id="A", return_meta=True)
    pd.testing.assert_frame_equal(X0, X1)
    assert np.array_equal(y0, y1)
    assert np.array_equal(g0, g1)


def test_return_meta_row_alignment_and_columns():
    """Guardrail #4: meta is row-aligned with X and carries the behind-line box detail."""
    frames, actions = _match_with_behind_line_defender()
    X, _y, _g, meta = xc.prepare_xcross_training_data(frames, actions, home_team_id="A", return_meta=True)
    assert len(meta) == len(X)
    assert _META_COLS <= set(meta.columns)
    # the behind-line defender (gr_x = -1) is captured on every example
    assert all(np.allclose(a, [-1.0]) for a in meta["behind_line_box_gr_x"])


def test_meta_clamped_ratio_equals_the_scoped_clamp():
    """Anti-drift: the meta's clamped ratio must equal box_off_def_ratio under a REAL gr_x>=0 clamp,
    so the descriptive split is provably the same points the clamp removes."""
    frames, actions = _match_with_behind_line_defender()
    _X, _y, _g, meta = xc.prepare_xcross_training_data(frames, actions, home_team_id="A", return_meta=True)

    original = _geo.in_penalty_area_goal_relative_array
    try:
        _geo.in_penalty_area_goal_relative_array = lambda gx, yy: original(gx, yy) & (np.asarray(gx) >= 0.0)
        X_clamped = xc.prepare_xcross_training_data(frames, actions, home_team_id="A")[0]
    finally:
        _geo.in_penalty_area_goal_relative_array = original

    assert np.allclose(
        meta["box_off_def_ratio_clamped"].to_numpy(float),
        X_clamped["box_off_def_ratio"].to_numpy(float),
        equal_nan=True,
    )
