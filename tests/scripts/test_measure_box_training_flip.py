"""B1: the training-feature-delta measurement (basis A) on the ghost box feature.

Verifies the scoped clamp (N1), the seam routing (C4), and the N-1 population coherence between the
seam-derived ``changed_fraction`` and the reconstructed behind-line points.
"""

from __future__ import annotations

import pandas as pd
import pytest

import silly_kicks.tracking._geometry as _geo
from scripts.measure_box_constant_delta import _scoped_gr_x_clamp, measure_training_flip
from tests.tracking.test_ghost_gk import _make_ghost_gk_frames


@pytest.fixture
def behind_line_ghost_frames():
    """Two frames with one away attacker BEHIND the attacked (x=0) goal line, in the y-band, plus
    one attacker IN the box (L3: non-vacuous by construction)."""
    parts = []
    for fid in (1, 2):
        f = _make_ghost_gk_frames(frame_id=fid, timestamp=float(fid))
        f.loc[f["player_id"] == "a10", ["x", "y"]] = [-1.0, 34.0]  # behind the goal line, in band
        f.loc[f["player_id"] == "a11", ["x", "y"]] = [5.0, 34.0]  # in the box, in front of the line
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def _box(frames):
    from silly_kicks.tracking import prepare_ghost_gk_training_data

    return prepare_ghost_gk_training_data(frames, home_team_id=1)[0]["attackers_in_box"].to_numpy()


def test_behind_line_fixture_is_non_vacuous(behind_line_ghost_frames):
    # L3: without a behind-line-in-band attacker the clamp non-vacuity assertion is itself vacuous.
    assert (_box(behind_line_ghost_frames) > 0).any()


def test_scoped_clamp_changes_box_feature_and_reverts(behind_line_ghost_frames):
    base = _box(behind_line_ghost_frames)
    with _scoped_gr_x_clamp():
        clamped = _box(behind_line_ghost_frames)
    assert (clamped <= base).all()
    assert (clamped < base).any()  # the behind-line attacker is removed
    after = _box(behind_line_ghost_frames)
    assert (after == base).all()  # module attribute restored
    assert _geo.in_penalty_area_goal_relative_array.__name__ != "_clamped"


def test_measure_training_flip_ghost_is_coherent_and_material(behind_line_ghost_frames):
    out = measure_training_flip(behind_line_ghost_frames, None, 1, model="ghost")
    # 2 of the 4 (frame x GK) examples change (the two home-GK examples lose the behind-line attacker)
    assert out["changed_fraction"] == pytest.approx(0.5)
    assert out["n_behind_line"] == 2  # a10 behind-line in each of the 2 frames
    # gr_x = -1 is within the 2 m off-pitch margin -> a REAL near-line position, not garbage
    assert out["real_near_line_fraction"] == pytest.approx(1.0)
    assert out["offpitch_fraction"] == 0.0
    assert out["train_behind_line_base_rate"] == pytest.approx(0.5)


def test_measure_training_flip_routes_through_the_library_seam(monkeypatch, behind_line_ghost_frames):
    # C4: R_M is identical by construction only because the measurement calls the trainer's seam.
    import silly_kicks.tracking as T

    real = T.prepare_ghost_gk_training_data
    calls = []

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(T, "prepare_ghost_gk_training_data", spy)
    measure_training_flip(behind_line_ghost_frames, None, 1, model="ghost")
    assert calls, "measurement must route through prepare_ghost_gk_training_data, not re-implement extraction"


@pytest.fixture
def xcross_behind_line_match():
    """Four frames + a cross action, with one BEHIND-the-line defender in the attacked box (gr_x=-1)
    and enough in-box players that removing it MOVES box_off_def_ratio (0.5 -> 1.0)."""
    from silly_kicks.spadl import config as spc

    players = [
        ("A", "A1", 95.0, 10.0, False),  # carrier / ball anchor (wide) -- out of the y-band
        ("A", "A2", 100.0, 34.0, False),  # attacker in box (gr_x=5)
        ("B", "B1d", 100.0, 30.0, False),  # defender in box, in front of the line (gr_x=5)
        ("B", "B2d", 106.0, 34.0, False),  # defender BEHIND the goal line (gr_x=-1), in band
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


def test_measure_training_flip_xcross_is_coherent_and_material(xcross_behind_line_match):
    frames, actions = xcross_behind_line_match
    out = measure_training_flip(frames, actions, "A", model="xcross")
    assert out["changed_fraction"] == pytest.approx(1.0)  # box_off_def_ratio 0.5 -> 1.0 on every row
    assert out["n_behind_line"] == 4  # the behind-line defender across 4 frames
    assert out["real_near_line_fraction"] == pytest.approx(1.0)  # gr_x=-1 within the 2 m margin
    assert out["offpitch_fraction"] == 0.0


def test_measure_training_flip_xcross_needs_actions(xcross_behind_line_match):
    frames, _actions = xcross_behind_line_match
    with pytest.raises(ValueError, match="actions"):
        measure_training_flip(frames, None, "A", model="xcross")
