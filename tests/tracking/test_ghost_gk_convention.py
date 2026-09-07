"""TF-60: ghost-GK goal-relative convention is the full 180-degree point reflection (both axes).

The extractor + target must be y-chirality-invariant: a scene and its 180-degree point
reflection (mapping the defended goal from x=0 to x=105) produce the SAME goal-relative
feature row. Under the legacy x-only convention the signed-y features (ball_y, ball_vy,
atk_cy, ball_to_goal_angle) differed between the two ends -- these tests are RED until the
both-axes change lands. See the spec 5.1 / ADR-089.
"""

import numpy as np
import pandas as pd

from silly_kicks.tracking._ghost_gk import _to_gr_x, _to_gr_y, extract_ghost_gk_features


def _one_frame(*, goal_x: float) -> pd.DataFrame:
    """A minimal single frame: ball + one GK + a defender + an attacker, all finite."""
    rows = [
        {
            "is_ball": True,
            "is_goalkeeper": False,
            "team_id": None,
            "player_id": None,
            "x": 60.0,
            "y": 20.0,
            "vx": 1.0,
            "vy": -2.0,
        },
        {
            "is_ball": False,
            "is_goalkeeper": True,
            "team_id": 1,
            "player_id": 10,
            "x": (5.0 if goal_x < 50 else 100.0),
            "y": 30.0,
            "vx": 0.0,
            "vy": 0.0,
        },
        {
            "is_ball": False,
            "is_goalkeeper": False,
            "team_id": 1,
            "player_id": 11,
            "x": (20.0 if goal_x < 50 else 85.0),
            "y": 15.0,
            "vx": 0.0,
            "vy": 0.0,
        },
        {
            "is_ball": False,
            "is_goalkeeper": False,
            "team_id": 2,
            "player_id": 21,
            "x": 55.0,
            "y": 40.0,
            "vx": 0.0,
            "vy": 0.0,
        },
    ]
    df = pd.DataFrame(rows)
    df["game_id"] = 1
    df["period_id"] = 1
    df["frame_id"] = 1
    df["time_seconds"] = 0.0
    return df


def test_selfinverse_transforms() -> None:
    for f in (True, False):
        assert _to_gr_x(_to_gr_x(37.0, f), f) == 37.0
        assert _to_gr_y(_to_gr_y(21.0, f), f) == 21.0


def test_to_gr_y_flips_only_when_flip() -> None:
    assert _to_gr_y(20.0, True) == 48.0  # 68 - 20
    assert _to_gr_y(20.0, False) == 20.0


def test_extractor_is_y_chirality_invariant() -> None:
    # A scene at goal_x=0 and its 180-deg point reflection at goal_x=105 must produce the
    # SAME goal-relative feature row. Under the OLD x-only convention the signed-y features
    # differ -> RED before the both-axes change.
    left = _one_frame(goal_x=0.0)
    right = left.copy()
    right["x"] = 105.0 - right["x"]
    right["y"] = 68.0 - right["y"]
    right["vx"] = -right["vx"]
    right["vy"] = -right["vy"]
    fl = extract_ghost_gk_features(left, gk_team_id=1, goal_x=0.0)
    fr = extract_ghost_gk_features(right, gk_team_id=1, goal_x=105.0)
    np.testing.assert_allclose(fl.to_numpy(dtype=float), fr.to_numpy(dtype=float), atol=1e-9, equal_nan=True)


def test_y_regression_breaks_extractor_chirality(monkeypatch) -> None:
    """Non-vacuity companion to ``test_extractor_is_y_chirality_invariant`` (ADR-089).

    The both-axes ``_to_gr_y`` is load-bearing: reverting it to the identity (a regression to
    the legacy x-only convention) must DESTROY the extractor's y-chirality invariance -- the
    same scene at the two goal ends then produces DIFFERENT goal-relative feature rows. This is
    what makes the chirality gate catch the convention: the fingerprint is a function of these
    features, so a moved feature row moves the fingerprint (empirically confirmed by the
    bundled-weight ``verify_chirality`` mismatch this change triggers -- see the transient-red
    set). Documented as a test rather than a change to the SHARED ``_chirality`` fingerprint,
    which xShot/xCross/ghost-outfield also depend on (ruling: no shared-module blast radius).
    """
    import silly_kicks.tracking._ghost_gk as g

    monkeypatch.setattr(g, "_to_gr_y", lambda y, flip: y)  # x-only (absolute-y) regression
    left = _one_frame(goal_x=0.0)
    right = left.copy()
    right["x"] = 105.0 - right["x"]
    right["y"] = 68.0 - right["y"]
    right["vx"] = -right["vx"]
    right["vy"] = -right["vy"]
    fl = g.extract_ghost_gk_features(left, gk_team_id=1, goal_x=0.0)
    fr = g.extract_ghost_gk_features(right, gk_team_id=1, goal_x=105.0)
    # Under the regression the invariance FAILS (signed-y features differ between the two ends).
    assert not np.allclose(fl.to_numpy(dtype=float), fr.to_numpy(dtype=float), atol=1e-9, equal_nan=True)
