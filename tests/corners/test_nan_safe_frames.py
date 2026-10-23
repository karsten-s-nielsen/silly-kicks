"""Regression: corner enrichers tolerate the tracking-frame ball row, whose player_id / team_id are
``<NA>`` by construction (ADR-058); both use the NA-safe ``id_compat`` helpers (ADR-019), never a raw ``==``.

Two distinct cases:
- ``add_corner_windows`` (``_windows._team_of``) is the REAL crash: the raw ``==`` fed a numpy
  boolean-index, so the ball row's ``<NA>`` player_id raised ``boolean value of NA is ambiguous`` on
  every SkillCorner match (the TF-46 smoke crash). ``test_add_corner_windows_*`` red-greens it.
- ``compute_corner_run_encoding`` (``_encode._nearest_frame_xy``) is DEFENSIVE hardening of the same
  anti-pattern: its raw ``==`` resolved to a plain-``bool`` mask (NA→False) and never actually crashed
  (measured on object / category / Int64 team_id), so ``test_run_encoding_*`` is a correctness guard
  (the ball row is excluded, the attacking players still encode), NOT a red-green crash-pin."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.corners import (
    RunZoneModel,
    add_corner_windows,
    compute_corner_run_encoding,
)
from tests.corners._frame_helpers import with_keepers

_CORNER_CROSSED = 5


def _toy_zone_model() -> RunZoneModel:
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


def _frames_with_ball_row() -> pd.DataFrame:
    """Two attacking players + a BALL row (player_id/team_id <NA>, as real tracking frames carry it)."""
    rows = []
    for fid, t in ((1, 98.0), (2, 101.5)):
        rows += [
            {
                "frame_id": fid,
                "period_id": 1,
                "time_seconds": t,
                "team_id": "tA",
                "player_id": "pa",
                "x": 90.0,
                "y": 25.0,
                "is_ball": False,
            },
            {
                "frame_id": fid,
                "period_id": 1,
                "time_seconds": t,
                "team_id": "tA",
                "player_id": "pb",
                "x": 90.0,
                "y": 43.0,
                "is_ball": False,
            },
            # ball row: NA ids by construction -- the trigger for the boolean-NA crash
            {
                "frame_id": fid,
                "period_id": 1,
                "time_seconds": t,
                "team_id": pd.NA,
                "player_id": pd.NA,
                "x": 95.0,
                "y": 34.0,
                "is_ball": True,
            },
        ]
    df = pd.DataFrame(rows)
    df["player_id"] = df["player_id"].astype("object")
    df["team_id"] = df["team_id"].astype("object")
    return with_keepers(df)


def test_add_corner_windows_tolerates_ball_row_na_player_id():
    corners = pd.DataFrame(
        {
            "game_id": ["g1"],
            "corner_id": [0],
            "period_id": [1],
            "t_kick": [100.0],
            "attacking_team_id": ["tA"],
            "corner_side": pd.Series(["left"], dtype="category"),
            "corner_kind": pd.Series(["direct"], dtype="category"),
            "corner_source": pd.Series(["spadl_corner"], dtype="category"),
        }
    )
    frames = _frames_with_ball_row()
    # A first contact whose frame carries the <NA>-id ball row -> _team_of must not crash.
    contacts = pd.DataFrame(
        {
            "period_id": [1],
            "frame_id": [2],
            "time_seconds": [101.5],
            "player_id": pd.Series(["pa"], dtype="object"),
            "contact_source": pd.Series(["velocity_break"], dtype="object"),
        }
    )
    from ._gate_helpers import passing

    out = add_corner_windows(corners, frames, contacts, **passing())  # must NOT raise
    assert len(out) == 1
    assert out["winner_team_id"].iloc[0] == "tA"  # resolved past the ball row (gate injected for the live path)


def test_run_encoding_tolerates_ball_row_na_team_id():
    # DEFENSIVE guard (not a red-green crash-pin): the old raw-== resolved to a plain-bool mask and did
    # not crash; this asserts the ids_match path still excludes the ball row and encodes the players.
    corners = pd.DataFrame(
        {
            "game_id": ["g1"],
            "corner_id": [0],
            "period_id": [1],
            "attacking_team_id": ["tA"],
            "corner_side": pd.Series(["left"], dtype="category"),
            "corner_kind": pd.Series(["direct"], dtype="category"),
            "t_init": [98.0],
            "t_target": [101.5],
        }
    )
    frames = _frames_with_ball_row()
    enc = compute_corner_run_encoding(corners, frames, model=_toy_zone_model())  # must NOT raise
    run_cols = [c for c in enc.columns if c.startswith("run_")]
    vals = enc[run_cols].to_numpy(dtype=float)
    assert not np.isnan(vals).all()  # the two attacking players encoded past the ball row
