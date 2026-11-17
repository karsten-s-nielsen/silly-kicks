"""A windowed corner table with one unmeasured direct corner (no kick time, no window, no side) for the corner
drivers' tests (supervisor C, bar A3, action-time repair spec section 5.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def windowed_corners() -> pd.DataFrame:
    """Corner 0: measured (t_kick 100 s, side left). Corner 1: unmeasured (every window time NaN, side <NA>)."""
    return pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "corner_id": [0, 1],
            "period_id": [1, 1],
            "attacking_team_id": ["tA", "tA"],
            "corner_side": pd.Series(["left", pd.NA], dtype="category"),
            "corner_kind": pd.Series(["direct", "direct"], dtype="category"),
            "corner_source": ["spadl", "spadl"],
            "t_kick": [100.0, np.nan],
            "t_init": [98.0, np.nan],
            "t_target": [101.5, np.nan],
            "t_outcome": [106.0, np.nan],
            "first_contact_time": [101.0, np.nan],
        }
    )


def frames() -> pd.DataFrame:
    """Two attacking players of tA at 0 s (the period's first frame), 98 s, 100 s and 101.5 s."""

    def block(fid: int, t: float, xy: list[tuple[float, float]]) -> pd.DataFrame:
        return pd.DataFrame(
            {"frame_id": [fid, fid], "period_id": [1, 1], "time_seconds": [t, t], "team_id": ["tA", "tA"],
             "player_id": ["pa", "pb"], "x": [xy[0][0], xy[1][0]], "y": [xy[0][1], xy[1][1]],
             "is_ball": [False, False]}
        )  # fmt: skip

    return pd.concat(
        [
            block(0, 0.0, [(50.0, 30.0), (52.0, 38.0)]),
            block(1, 98.0, [(90.0, 25.0), (90.0, 43.0)]),
            block(2, 100.0, [(92.0, 26.0), (92.0, 42.0)]),
            block(3, 101.5, [(100.0, 28.0), (100.0, 40.0)]),
        ],
        ignore_index=True,
    )


def stub_corner_pipeline(monkeypatch, table: pd.DataFrame | None = None) -> None:
    """Make identify_corners / add_corner_windows / detect_ball_contacts return ``table`` (default: the one above)."""
    import silly_kicks.corners as corners_pkg
    import silly_kicks.tracking._ball_contact as contact

    table = windowed_corners() if table is None else table
    monkeypatch.setattr(corners_pkg, "identify_corners", lambda actions, provider=None, **kw: table.copy())
    monkeypatch.setattr(corners_pkg, "add_corner_windows", lambda corners, frames, contacts, **kw: corners.copy())
    monkeypatch.setattr(contact, "detect_ball_contacts", lambda frames, **kw: pd.DataFrame())
