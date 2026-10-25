"""TF-46 dedicated orientation-invariance gate (replaces the tracking.__all__-pinned mirror Gate A/B).

(a) behavioural: a left-corner scene and its mirrored right-corner twin of identical geometry reflect to
    equal canonical geometry; non-vacuity probe: the right row MUST move.
(b) declaration-completeness (P2): every emitted corner geometry column (CORNER_*_COLUMNS matching
    reflection.GEOMETRIC_NAME) is declared in CORNER_REFLECTION_KINDS -- so a future geometry column
    cannot silently escape reflection.
(c) F0 (TF-46 Arm 1 Part 2, spec section 13.3): the run encoding orients each corner's snapshots to the
    attacking team's ACTION frame through the GoalMap (ADR-055), never a frame label. The SAME routine attacking
    x = 105 or x = 0, in either period and in a shoot-out period, encodes to the SAME zones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.corners import (
    CORNER_CHANCE_COLUMNS,
    CORNER_IDENTIFY_COLUMNS,
    CORNER_WINDOW_COLUMNS,
    reflect_corners_to_canonical_side,
)
from silly_kicks.corners._encode import _run_column_names, compute_corner_run_encoding
from silly_kicks.corners._reflect import CORNER_REFLECTION_KINDS
from silly_kicks.corners._zones import RunZoneModel
from silly_kicks.reflection import GEOMETRIC_NAME
from silly_kicks.tracking._gk_resolve import resolve_defended_goals


def test_mirror_twins_reflect_to_equal_geometry():
    # kick_y 60 (left) and 8 (right) are mirror images about the y=34 midline (68-60 == 8).
    df = pd.DataFrame({"corner_side": ["left", "right"], "kick_x": [104.0, 104.0], "kick_y": [60.0, 8.0]})
    out = reflect_corners_to_canonical_side(df)
    assert out.loc[0, "kick_y"] == out.loc[1, "kick_y"]  # mirror-equal after canonicalise


def test_non_vacuity_right_row_moves():
    df = pd.DataFrame({"corner_side": ["right"], "kick_x": [104.0], "kick_y": [8.0]})
    out = reflect_corners_to_canonical_side(df)
    assert out.loc[0, "kick_y"] != 8.0  # it MUST have moved


def test_every_emitted_geometry_column_is_declared():
    emitted = set(CORNER_IDENTIFY_COLUMNS) | set(CORNER_WINDOW_COLUMNS) | set(CORNER_CHANCE_COLUMNS)
    geometry = {c for c in emitted if GEOMETRIC_NAME.match(c)}
    undeclared = geometry - set(CORNER_REFLECTION_KINDS)
    assert not undeclared, f"geometry columns not declared in CORNER_REFLECTION_KINDS: {sorted(undeclared)}"
    # non-vacuity: the enumeration actually found the known geometry columns
    assert {"kick_x", "kick_y"} <= geometry


# ---------------------------------------------------------------------------
# F0: GoalMap-oriented run encoding
# ---------------------------------------------------------------------------

_INIT = [(90.0, 25.0), (90.0, 43.0)]  # the routine in the action frame (attacking +x)
_TGT = [(100.0, 28.0), (100.0, 40.0)]


def _model() -> RunZoneModel:
    rng = np.random.default_rng(0)
    init = np.vstack([rng.normal([90.0, 25.0], 1.0, size=(60, 2)), rng.normal([90.0, 43.0], 1.0, size=(60, 2))])
    target = np.vstack(
        [
            rng.normal([100.0, 28.0], 1.0, size=(40, 2)),
            rng.normal([100.0, 40.0], 1.0, size=(40, 2)),
            rng.normal([86.0, 34.0], 1.0, size=(40, 2)),
        ]
    )
    return RunZoneModel().fit(init, target, init_k=2, target_k=3)


def _mirror(xy):
    return [(105.0 - x, 68.0 - y) for x, y in xy]


def _period_frames(period: int, attacks_right: bool, *, label: str | None = None) -> pd.DataFrame:
    """One period of tracking: team tA runs the routine toward the goal it ATTACKS. Keepers make the GoalMap:
    tB's keeper stands in the goal tA attacks, tA's keeper in the other. ``label`` writes a (possibly WRONG)
    team_attacking_direction label -- the encoding must ignore it (direction comes from the GoalMap only)."""
    init = _INIT if attacks_right else _mirror(_INIT)
    tgt = _TGT if attacks_right else _mirror(_TGT)
    gk_a_x, gk_b_x = (3.0, 102.0) if attacks_right else (102.0, 3.0)
    rows = []
    for fid, t, xy in [(period * 10 + 1, 98.0, init), (period * 10 + 2, 101.5, tgt)]:
        players = [("pa", "tA", False, *xy[0]), ("pb", "tA", False, *xy[1])]
        players += [("ka", "tA", True, gk_a_x, 34.0), ("kb", "tB", True, gk_b_x, 34.0)]
        for pid, team, is_gk, x, y in players:
            rows.append(
                {
                    "game_id": "g",
                    "frame_id": fid,
                    "period_id": period,
                    "time_seconds": t,
                    "team_id": team,
                    "player_id": pid,
                    "x": float(x),
                    "y": float(y),
                    "is_ball": False,
                    "is_goalkeeper": is_gk,
                    "team_attacking_direction": label,
                }
            )
    return pd.DataFrame(rows)


def _corners(periods) -> pd.DataFrame:
    n = len(periods)
    return pd.DataFrame(
        {
            "game_id": ["g"] * n,
            "corner_id": list(range(n)),
            "period_id": list(periods),
            "attacking_team_id": ["tA"] * n,
            "corner_side": pd.Series(["left"] * n, dtype="category"),
            "corner_kind": pd.Series(["direct"] * n, dtype="category"),
            "t_init": [98.0] * n,
            "t_target": [101.5] * n,
        }
    )


def _vectors(frames: pd.DataFrame, periods, model) -> list[np.ndarray]:
    out = compute_corner_run_encoding(_corners(periods), frames, model=model)
    rc = _run_column_names(model)
    return [out[out["corner_id"] == i].iloc[0][rc].to_numpy(dtype=float) for i in range(len(periods))]


@pytest.mark.parametrize(
    ("p1_right", "p2_right"),
    [(True, False), (False, True), (True, True), (False, False)],
    ids=["right-left", "left-right", "right-right", "left-left"],
)
def test_f0_both_directions_both_periods_encode_the_same_zones(p1_right, p2_right):
    model = _model()
    frames = pd.concat([_period_frames(1, p1_right), _period_frames(2, p2_right)], ignore_index=True)
    v1, v2 = _vectors(frames, (1, 2), model)
    assert v1.sum() > 0 and v2.sum() > 0  # non-vacuous: the routine encoded in both periods
    assert np.allclose(v1, v2)


def test_f0_shoot_out_period_is_oriented_too():
    model = _model()
    frames = pd.concat([_period_frames(1, True), _period_frames(5, False)], ignore_index=True)
    v1, v5 = _vectors(frames, (1, 5), model)
    assert v5.sum() > 0 and np.allclose(v1, v5)


def test_f0_direction_comes_from_the_goal_map_not_the_frame_label():
    # A WRONG label ("ltr" while tA attacks x = 0) must not change the encoding: the GoalMap decides.
    model = _model()
    right = _period_frames(1, True)
    wrong_label = _period_frames(2, False, label="ltr")
    v1, v2 = _vectors(pd.concat([right, wrong_label], ignore_index=True), (1, 2), model)
    assert np.allclose(v1, v2)


def test_f0_unoriented_snapshot_would_differ_non_vacuity():
    # Non-vacuity: WITHOUT orientation the mirrored period encodes differently, so the equality above is real.
    model = _model()
    rc = _run_column_names(model)
    raw_right = model.encode_runs(np.array(_INIT), np.array(_TGT))
    raw_left = model.encode_runs(np.array(_mirror(_INIT)), np.array(_mirror(_TGT)))
    assert not np.allclose(raw_right, raw_left) and len(rc) == len(raw_right)


def test_f0_unresolved_direction_withholds_the_encoding():
    # No keepers -> the GoalMap cannot resolve the period -> NaN coords -> the run vector is withheld (NaN),
    # never a guessed orientation.
    model = _model()
    frames = _period_frames(1, True)
    frames = frames[~frames["is_goalkeeper"]].copy()
    goal_map = resolve_defended_goals(frames)
    out = compute_corner_run_encoding(_corners((1,)), frames, model=model, goal_map=goal_map)
    rc = _run_column_names(model)
    vec = out.iloc[0][rc].to_numpy(dtype=float)
    assert np.isnan(vec).all()
