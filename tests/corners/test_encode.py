"""TF-46 compute_corner_run_encoding: per-corner run vector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.corners._encode import _run_column_names, compute_corner_run_encoding
from silly_kicks.corners._zones import RunZoneModel


def _model() -> RunZoneModel:
    rng = np.random.default_rng(0)
    init = np.vstack([rng.normal([90.0, 25.0], 1.0, size=(60, 2)), rng.normal([90.0, 43.0], 1.0, size=(60, 2))])
    target = np.vstack(
        [
            rng.normal([100.0, 28.0], 1.0, size=(40, 2)),
            rng.normal([100.0, 40.0], 1.0, size=(40, 2)),
            rng.normal([86.0, 34.0], 1.0, size=(40, 2)),  # top-of-box / second-ball edge (in-region)
        ]
    )
    return RunZoneModel().fit(init, target, init_k=2, target_k=3)


def _corners():
    return pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "corner_id": [0, 1],
            "period_id": [1, 1],
            "attacking_team_id": ["tA", "tA"],
            "corner_side": pd.Series(["left", "left"], dtype="category"),
            "corner_kind": pd.Series(["direct", "short"], dtype="category"),
            "t_init": [98.0, 198.0],
            "t_target": [101.5, 201.5],
        }
    )


def _frames():
    # two attacking players: init near the two initial clusters, target near the two active zones
    def block(fid, t, xy):
        return pd.DataFrame(
            {
                "frame_id": [fid, fid],
                "period_id": [1, 1],
                "time_seconds": [t, t],
                "team_id": ["tA", "tA"],
                "player_id": ["pa", "pb"],
                "x": [xy[0][0], xy[1][0]],
                "y": [xy[0][1], xy[1][1]],
                "is_ball": [False, False],
            }
        )

    init_frame = block(1, 98.0, [(90.0, 25.0), (90.0, 43.0)])
    target_frame = block(2, 101.5, [(100.0, 28.0), (100.0, 40.0)])
    return pd.concat([init_frame, target_frame], ignore_index=True)


def test_direct_corner_encodes_short_is_nan():
    model = _model()
    out = compute_corner_run_encoding(_corners(), _frames(), model=model)
    run_cols = _run_column_names(model)
    assert list(out.columns) == ["game_id", "corner_id", *run_cols]
    assert len(run_cols) == model.n_initial_zones * model.n_target_zones  # 2*3 = 6
    direct = out[out["corner_id"] == 0].iloc[0]
    short = out[out["corner_id"] == 1].iloc[0]
    assert direct[run_cols].to_numpy(dtype=float).sum() > 0  # direct corner encoded
    assert short[run_cols].isna().all()  # short corner excluded (NaN row kept)


def test_derived_corner_nan_window_yields_nan_run():
    # B13 / spec section 9.4: a derived corner has a NaN window (NaN t_init/t_target) and a NaN side. The
    # encoder must return a NaN run row -- never pick an arbitrary frame via argmin over an all-NaN
    # distance, and never reflect str(NaN)=="nan" as "right". Non-vacuous: with a real window it encodes
    # (test_direct_corner_encodes_short_is_nan).
    model = _model()
    corners = pd.DataFrame(
        {
            "game_id": ["g1"],
            "corner_id": [0],
            "period_id": [1],
            "attacking_team_id": ["tA"],
            "corner_side": pd.Series([None], dtype="category"),  # NaN side (derived corner)
            "corner_kind": pd.Series(["direct"], dtype="category"),
            "t_init": [float("nan")],
            "t_target": [float("nan")],
        }
    )
    out = compute_corner_run_encoding(corners, _frames(), model=model)
    run_cols = _run_column_names(model)
    assert out.iloc[0][run_cols].isna().all()  # NaN run row, no arbitrary frame


def test_nan_corner_side_real_window_yields_nan_run():
    # B18 / spec section 9.4: a NaN start_y yields a NaN corner_side even on a REAL (non-derived) window.
    # The canonical reflection is then undefined; str(NaN) must NOT silently flip as "right". The encoder
    # returns a NaN run row (ADR-042), not a fabricated vector. Non-vacuous: the SAME window with a "left"
    # side encodes (test_direct_corner_encodes_short_is_nan).
    model = _model()
    corners = pd.DataFrame(
        {
            "game_id": ["g1"],
            "corner_id": [0],
            "period_id": [1],
            "attacking_team_id": ["tA"],
            "corner_side": pd.Series([None], dtype="category"),  # NaN side, real window
            "corner_kind": pd.Series(["direct"], dtype="category"),
            "t_init": [98.0],
            "t_target": [101.5],
        }
    )
    out = compute_corner_run_encoding(corners, _frames(), model=model)
    run_cols = _run_column_names(model)
    assert out.iloc[0][run_cols].isna().all()  # NaN run row, not a reflected-as-right vector


def test_non_vacuous_spread():
    model = _model()
    out = compute_corner_run_encoding(_corners(), _frames(), model=model)
    run_cols = _run_column_names(model)
    direct = out[out["corner_id"] == 0].iloc[0][run_cols].to_numpy(dtype=float)
    assert (direct > 1e-6).sum() > 1  # two players to two zones -> more than one run-pair populated


def test_empty_corners_returns_empty_schema():
    model = _model()
    empty = _corners().iloc[0:0]
    out = compute_corner_run_encoding(empty, _frames(), model=model)
    assert out.empty
    assert list(out.columns) == ["game_id", "corner_id", *_run_column_names(model)]


def test_sb360_corner_sources_init_from_kick_frame():
    # SB360: no frame at t_init (t_kick-2s); the kick freeze frame is at t_kick. The encoder must
    # source init from t_kick for corner_source == "sb360_event" (spec §7), so it still encodes.
    model = _model()
    corners = pd.DataFrame(
        {
            "game_id": ["g1"],
            "corner_id": [0],
            "period_id": [1],
            "attacking_team_id": ["tA"],
            "corner_side": pd.Series(["left"], dtype="category"),
            "corner_kind": pd.Series(["direct"], dtype="category"),
            "corner_source": pd.Series(["sb360_event"], dtype="category"),
            "t_kick": [100.0],
            "t_init": [98.0],  # no frame here on SB360
            "t_target": [101.5],
        }
    )

    def block(fid, t, xy):
        return pd.DataFrame(
            {
                "frame_id": [fid, fid],
                "period_id": [1, 1],
                "time_seconds": [t, t],
                "team_id": ["tA", "tA"],
                "player_id": ["pa", "pb"],
                "x": [xy[0][0], xy[1][0]],
                "y": [xy[0][1], xy[1][1]],
                "is_ball": [False, False],
            }
        )

    # freeze frames ONLY at t_kick (100) and the first-event target (101.5) -- none at t_init 98
    frames = pd.concat(
        [block(1, 100.0, [(90.0, 25.0), (90.0, 43.0)]), block(2, 101.5, [(100.0, 28.0), (100.0, 40.0)])],
        ignore_index=True,
    )
    out = compute_corner_run_encoding(corners, frames, model=model)
    run_cols = _run_column_names(model)
    assert out.iloc[0][run_cols].to_numpy(dtype=float).sum() > 0  # encoded via the kick freeze frame


def test_pure_does_not_mutate():
    model = _model()
    corners, frames = _corners(), _frames()
    cb, fb = corners.copy(deep=True), frames.copy(deep=True)
    compute_corner_run_encoding(corners, frames, model=model)
    pd.testing.assert_frame_equal(corners, cb)
    pd.testing.assert_frame_equal(frames, fb)
