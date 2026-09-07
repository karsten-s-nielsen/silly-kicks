"""TF-60 Task 3: serve_ghost_gk_positions emits frame-coordinate ghost_x/_y (ADR-089).

The serve converts its goal-relative predictions to frame coordinates using the model's OWN
self-inverse transforms (so no consumer re-derives orientation), keeping ghost_gr_x/_y as audit.
Self-contained (no cross-test-module import): a tiny fit + a two-keeper frame with BOTH ends.
"""

import numpy as np
import pandas as pd

from silly_kicks.tracking import resolve_defended_goals, serve_ghost_gk_positions
from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GhostGkModel, _to_gr_x, _to_gr_y


def _fit_tiny_gk_model() -> GhostGkModel:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((80, 26)), columns=GHOST_GK_FEATURE_NAMES)
    labels = pd.DataFrame({"gk_x": rng.uniform(2, 20, 80), "gk_y": rng.uniform(25, 45, 80)})
    model = GhostGkModel(n_estimators=10)
    model.fit(X, labels)
    return model


def _two_keeper_frames() -> pd.DataFrame:
    base = dict(
        game_id="100",
        period_id=1,
        frame_id=1,
        time_seconds=1.0,
        timestamp=1.0,
        ball_state="alive",
        source_provider="gradientsports",
    )

    def _row(pid, team, x, y, *, is_ball=False, is_gk=False, tad=None):
        return {
            **base,
            "player_id": pid,
            "team_id": team,
            "x": float(x),
            "y": float(y),
            "vx": 0.0,
            "vy": 0.0,
            "speed": 0.0,
            "is_ball": is_ball,
            "is_goalkeeper": is_gk,
            "team_attacking_direction": tad,
        }

    rows = [
        _row("ball", None, 60, 30, is_ball=True),
        # home keeper defends x=0 (flip=False); 4 home outfield
        _row("hgk", 1, 5, 34, is_gk=True, tad="ltr"),
        _row("h1", 1, 20, 25, tad="ltr"),
        _row("h2", 1, 22, 40, tad="ltr"),
        _row("h3", 1, 70, 30, tad="ltr"),
        _row("h4", 1, 78, 45, tad="ltr"),
        # away keeper defends x=105 (flip=True); 4 away outfield
        _row("agk", 2, 100, 34, is_gk=True, tad="rtl"),
        _row("a1", 2, 85, 25, tad="rtl"),
        _row("a2", 2, 83, 40, tad="rtl"),
        _row("a3", 2, 35, 30, tad="rtl"),
        _row("a4", 2, 27, 45, tad="rtl"),
    ]
    return pd.DataFrame(rows)


def test_serve_emits_frame_coords_matching_the_inverse() -> None:
    frames = _two_keeper_frames()
    out = serve_ghost_gk_positions(frames, model=_fit_tiny_gk_model(), home_team_id=1)
    assert {"ghost_x", "ghost_y", "ghost_gr_x", "ghost_gr_y"} <= set(out.columns)
    assert len(out) >= 1
    gm = resolve_defended_goals(frames)
    for _, r in out.iterrows():
        end = gm.get(r["game_id"], r["period_id"], r["gk_team_id"], allow_guess=True)
        flip = end is not None and end > 50.0
        np.testing.assert_allclose(r["ghost_x"], _to_gr_x(float(r["ghost_gr_x"]), flip), atol=1e-12)
        np.testing.assert_allclose(r["ghost_y"], _to_gr_y(float(r["ghost_gr_y"]), flip), atol=1e-12)


def test_serve_frame_coords_mirror_consistent_across_both_ends() -> None:
    # home keeper (team 1, defends x=0, flip=False) keeps gr==frame; away keeper (team 2,
    # defends x=105, flip=True) point-reflects BOTH axes -- the both-axes write-back guard.
    frames = _two_keeper_frames()
    out = serve_ghost_gk_positions(frames, model=_fit_tiny_gk_model(), home_team_id=1).set_index("gk_team_id")
    assert {1, 2} <= set(out.index)

    # `to_dict("index")` -> {gk_team_id: {col: value}} with plainly-typed values (avoids the
    # `.loc[scalar]` Series|DataFrame / Scalar typing ambiguity that pandas-stubs cannot narrow to float).
    rows = out.to_dict("index")
    home, away = rows[1], rows[2]
    np.testing.assert_allclose(float(home["ghost_x"]), float(home["ghost_gr_x"]), atol=1e-12)
    np.testing.assert_allclose(float(home["ghost_y"]), float(home["ghost_gr_y"]), atol=1e-12)

    np.testing.assert_allclose(float(away["ghost_x"]), 105.0 - float(away["ghost_gr_x"]), atol=1e-12)
    np.testing.assert_allclose(float(away["ghost_y"]), 68.0 - float(away["ghost_gr_y"]), atol=1e-12)
