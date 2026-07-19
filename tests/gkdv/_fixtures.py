"""Shared gkdv frame fixtures.

Anchored on ``tests/tracking/test_ghost_gk.py::_make_ghost_gk_frames`` (the verified
ghost-GK-shaped fixture) rather than hand-rolled, per the plan's fixture convention.

**Why a derived variant is necessary.** The anchor fixture parks the ball on the halfway
line (x=50) while the in-possession (away) team attacks the x=0 goal, so the ball is 50 m
from the attacked goal -- comfortably OUTSIDE the engine's 35 m spec §4.1 domain. Every
engine test written against the anchor as-is therefore scores ZERO frames, and every
assertion over "the scored rows" passes vacuously on an empty frame. These helpers move
the ball and the attackers into the attacking third so the domain is genuinely exercised;
the schema, provider tag and column set stay the anchor's.

This module deliberately contains NO model construction, so the bundled-weight scanner in
``test_import_allowlist.py`` (which globs ``test_*.py``) keeps a complete surface.
"""

from __future__ import annotations

import pandas as pd

from tests.tracking.test_ghost_gk import _make_ghost_gk_frames

#: Away (team 2) attacks the x=0 goal, so an in-domain ball sits near x=0.
_BALL_XY = (20.0, 34.0)
#: Away attackers, one of them ON the ball so `infer_ball_carrier` resolves team 2.
_ATTACKER_XY = {"a10": (18.0, 30.0), "a11": (21.0, 34.0), "a12": (19.0, 40.0), "a13": (20.2, 34.0)}
#: Home defenders, retreated goal-side of the ball.
_DEFENDER_XY = {"p10": (10.0, 25.0), "p11": (12.0, 30.0), "p12": (11.0, 38.0), "p13": (13.0, 45.0)}


def _place(frames: pd.DataFrame, player_id: str, xy: tuple[float, float]) -> None:
    mask = frames["player_id"] == player_id
    frames.loc[mask, "x"] = xy[0]
    frames.loc[mask, "y"] = xy[1]


def in_domain_frames(**kwargs) -> pd.DataFrame:
    """The anchor fixture with play moved into the away team's attacking third.

    Resulting geometry: home (team 1) keeps the x=0 goal and IS the defending team; away
    (team 2) is in possession and attacks x=0; the ball sits 20 m from the attacked goal.
    """
    frames = _make_ghost_gk_frames(**kwargs).copy()
    ball = frames["is_ball"].astype(bool)
    frames.loc[ball, "x"] = _BALL_XY[0]
    frames.loc[ball, "y"] = _BALL_XY[1]
    for pid, xy in {**_ATTACKER_XY, **_DEFENDER_XY}.items():
        _place(frames, pid, xy)
    return frames


def multi_frame_in_domain(n_frames: int = 6) -> pd.DataFrame:
    """``n_frames`` consecutive in-domain frames (for stride / aggregation tests)."""
    parts = []
    for i in range(n_frames):
        part = in_domain_frames(frame_id=i, timestamp=1.0 + i * 0.04)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
