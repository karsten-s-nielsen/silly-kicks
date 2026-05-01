"""Tests for VAEP.compute_features extension with frames= kwarg.

Loop 4 covers: backward-compat regression, frame-aware xfn dispatch, ValueError
on missing frames, no module-import cycle.
"""

from __future__ import annotations

import pandas as pd
import pytest


def _make_game_and_actions():
    """Tiny game (one row) + 5 actions, no tracking."""
    game = pd.Series({"game_id": 1, "home_team_id": 1, "away_team_id": 2})
    actions = pd.DataFrame(
        {
            "game_id": [1] * 5,
            "original_event_id": [None] * 5,
            "action_id": [1, 2, 3, 4, 5],
            "period_id": [1, 1, 1, 1, 1],
            "time_seconds": [5.0, 10.0, 15.0, 20.0, 25.0],
            "team_id": [1, 1, 2, 1, 1],
            "player_id": [11, 12, 21, 11, 12],
            "start_x": [10.0, 30.0, 50.0, 70.0, 90.0],
            "start_y": [34.0, 30.0, 34.0, 38.0, 34.0],
            "end_x": [30.0, 50.0, 30.0, 90.0, 100.0],
            "end_y": [34.0, 30.0, 34.0, 38.0, 34.0],
            "type_id": [0, 0, 0, 0, 11],
            "result_id": [1, 1, 1, 1, 1],
            "bodypart_id": [0, 0, 0, 0, 0],
        }
    )
    return game, actions


def test_compute_features_frames_none_is_regression_equivalent():
    """frames=None must be bit-identical to today (backward compat)."""
    from silly_kicks.vaep.base import VAEP

    v_old = VAEP()
    v_new = VAEP()
    game, actions = _make_game_and_actions()

    X_old = v_old.compute_features(game, actions)
    X_new = v_new.compute_features(game, actions, frames=None)
    pd.testing.assert_frame_equal(X_old, X_new)


def test_compute_features_raises_when_frame_aware_xfn_but_no_frames():
    """Frame-aware xfn in xfns + frames=None should raise ValueError with xfn name."""
    from silly_kicks.vaep.base import VAEP
    from silly_kicks.vaep.feature_framework import frame_aware

    @frame_aware
    def fake_tracking_feat(states, frames):
        return pd.DataFrame({"x": [1.0]})

    v = VAEP(xfns=[fake_tracking_feat])
    game, actions = _make_game_and_actions()
    with pytest.raises(ValueError, match="fake_tracking_feat"):
        v.compute_features(game, actions, frames=None)


def test_compute_features_dispatches_frame_aware_xfn():
    """Frame-aware xfn called with (states, frames) yields its columns in output."""
    from silly_kicks.vaep.base import VAEP
    from silly_kicks.vaep.feature_framework import frame_aware

    @frame_aware
    def fake_tracking_feat(states, frames):
        n = len(states[0])
        return pd.DataFrame({"fake_a0": [42.0] * n}, index=states[0].index)

    v = VAEP(xfns=[fake_tracking_feat])
    game, actions = _make_game_and_actions()
    # frames here is empty but non-None -> the @frame_aware xfn doesn't use it,
    # so the dispatch path is exercised.
    # IMPORTANT: the frames DataFrame must have a `team_attacking_direction`
    # column so the lazy-imported play_left_to_right call can run.
    frames = pd.DataFrame(
        {
            "period_id": [],
            "frame_id": [],
            "time_seconds": [],
            "team_attacking_direction": [],
            "x": [],
            "y": [],
        }
    )
    X = v.compute_features(game, actions, frames=frames)
    assert "fake_a0" in X.columns
    assert (X["fake_a0"] == 42.0).all()


def test_no_module_import_cycle_when_frames_is_none():
    """Importing silly_kicks.vaep.base alone (without tracking) must not fail."""
    import importlib

    importlib.import_module("silly_kicks.vaep.base")


def test_rate_passes_frames_to_compute_features():
    """rate(..., frames=...) routes through compute_features when game_states is None."""
    from sklearn.exceptions import NotFittedError

    from silly_kicks.vaep.base import VAEP
    from silly_kicks.vaep.feature_framework import frame_aware

    @frame_aware
    def fake_tracking_feat(states, frames):
        return pd.DataFrame({"fake_col": [0.5] * len(states[0])}, index=states[0].index)

    v = VAEP(xfns=[fake_tracking_feat])
    game, actions = _make_game_and_actions()
    frames = pd.DataFrame(
        {
            "period_id": [],
            "frame_id": [],
            "time_seconds": [],
            "team_attacking_direction": [],
            "x": [],
            "y": [],
        }
    )

    # rate without fit raises NotFittedError, but it must reach _estimate_probabilities;
    # before that it calls compute_features with frames= when game_states is None.
    # We just verify the call doesn't error on the frames= passthrough.
    with pytest.raises(NotFittedError):
        v.rate(game, actions, game_states=None, frames=frames)


# ---------------------------------------------------------------------------
# PR-S21 — pre_shot_gk_default_xfns dispatch
# ---------------------------------------------------------------------------


def _make_game_actions_with_gk_id_and_frames():
    """Tiny game + 5 actions including a SHOT with defending_gk_player_id populated;
    one frame containing the GK row at known coords."""
    game = pd.Series({"game_id": 1, "home_team_id": 1, "away_team_id": 2})
    from silly_kicks.spadl import config as spadlconfig

    shot_id = spadlconfig.actiontype_id["shot"]
    pass_id = spadlconfig.actiontype_id["pass"]
    actions = pd.DataFrame(
        {
            "game_id": [1] * 5,
            "original_event_id": [None] * 5,
            "action_id": [1, 2, 3, 4, 5],
            "period_id": [1, 1, 1, 1, 1],
            "time_seconds": [5.0, 10.0, 15.0, 20.0, 25.0],
            "team_id": [1, 1, 2, 1, 1],
            "player_id": [11, 12, 21, 11, 12],
            "start_x": [10.0, 30.0, 50.0, 70.0, 90.0],
            "start_y": [34.0, 30.0, 34.0, 38.0, 34.0],
            "end_x": [30.0, 50.0, 30.0, 90.0, 100.0],
            "end_y": [34.0, 30.0, 34.0, 38.0, 34.0],
            "type_id": [pass_id, pass_id, pass_id, pass_id, shot_id],
            "result_id": [1, 1, 1, 1, 1],
            "bodypart_id": [0, 0, 0, 0, 0],
            "defending_gk_player_id": [
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                99.0,
            ],
        }
    )
    # One frame at t=25.0 containing GK player_id=99 (team 2) at (104, 34).
    frames = pd.DataFrame(
        [
            dict(
                game_id=1,
                period_id=1,
                frame_id=2500,
                time_seconds=25.0,
                frame_rate=10.0,
                player_id=99,
                team_id=2,
                is_ball=False,
                is_goalkeeper=True,
                x=104.0,
                y=34.0,
                z=float("nan"),
                speed=0.5,
                speed_source="native",
                ball_state="alive",
                team_attacking_direction="ltr",
                confidence=None,
                visibility=None,
                source_provider="test",
            ),
        ]
    )
    return game, actions, frames


def test_compute_features_dispatches_pre_shot_gk_default_xfns():
    """xfns=pre_shot_gk_default_xfns + frames -> 4 features x nb_states columns emitted."""
    from silly_kicks.tracking.features import pre_shot_gk_default_xfns
    from silly_kicks.vaep.base import VAEP

    game, actions, frames = _make_game_actions_with_gk_id_and_frames()
    v = VAEP(xfns=pre_shot_gk_default_xfns)  # type: ignore[arg-type]  # FrameAwareTransformer wraps FeatureTransfomer
    X = v.compute_features(game, actions, frames=frames)
    # 4 features x nb_states (default 3) = 12 columns
    expected_cols = {
        f"pre_shot_gk_{name}_a{i}" for name in ("x", "y", "distance_to_goal", "distance_to_shot") for i in range(3)
    }
    assert expected_cols.issubset(set(X.columns))


def test_compute_features_pre_shot_gk_xfn_silent_nan_when_defending_gk_player_id_missing():
    """When ``defending_gk_player_id`` is missing, per-Series helpers emit all-NaN.

    The aggregator (``add_pre_shot_gk_position``) raises ValueError directly — that's
    the user-facing contract — but per-Series helpers used via ``lift_to_states`` in
    VAEP must tolerate VAEP's internal column-name introspection (which builds a dummy
    10-row gamestate without the extension column). Documented in the docstrings of
    ``pre_shot_gk_*`` and ``pre_shot_gk_default_xfns``: callers must run
    ``add_pre_shot_gk_context(actions)`` first to populate ``defending_gk_player_id``.
    """
    from silly_kicks.tracking.features import pre_shot_gk_default_xfns
    from silly_kicks.vaep.base import VAEP

    game, actions, frames = _make_game_actions_with_gk_id_and_frames()
    actions = actions.drop(columns=["defending_gk_player_id"])
    v = VAEP(xfns=pre_shot_gk_default_xfns)  # type: ignore[arg-type]  # FrameAwareTransformer wraps FeatureTransfomer
    X = v.compute_features(game, actions, frames=frames)
    # All GK columns silently NaN — model can still fit but feature carries no signal.
    for col in X.columns:
        if col.startswith("pre_shot_gk_"):
            assert X[col].isna().all(), f"{col} has non-NaN values when defending_gk_player_id missing"
