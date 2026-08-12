"""player_influence threat-grid orientation (DEFECT B, ADR-041).

``ExpectedThreat.xT`` stores rows y-INVERTED (row 0 = the TOP of the pitch), and
``interpolator()`` preserves that storage orientation in its output. ``rate()`` compensates
by indexing with the same inversion; ``compute_player_influence`` did NOT -- it multiplied
the raw interpolator output elementwise against ascending-y pitch-control surfaces, so the
threat weighting was y-mirrored. It stayed invisible in practice only because a fitted xT
surface is close to y-symmetric.

The fixture below is deliberately NOT y-symmetric: the grid value equals the physical
y-centre of its band, so a mirrored weighting is unmissable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from silly_kicks.tracking._player_influence import compute_player_influence
from silly_kicks.xthreat import ExpectedThreat

_HOME = 1
_HIGH_Y_PID, _LOW_Y_PID = 11, 12


def _y_ramp_xt() -> ExpectedThreat:
    """Grid value == the physical y-centre of the storage band it occupies."""
    m = ExpectedThreat()
    w, _l = m.xT.shape
    cell_w = 68.0 / w
    for i in range(w):
        m.xT[i, :] = (w - 1 - i + 0.5) * cell_w  # storage row 0 = TOP of pitch
    return m


def _frame() -> pd.DataFrame:
    """Two same-team outfielders, mirrored in y, otherwise identical.

    Home team attacks right, and it is the attacking team, so the away-team x-flip in
    ``compute_player_influence`` is NOT in play -- the y axis is isolated.
    """

    def row(pid, team, gk, x, y, is_ball=False):
        return {
            "game_id": 1,
            "period_id": 1,
            "frame_id": 100,
            "time_seconds": 4.0,
            "frame_rate": 25.0,
            "player_id": pid,
            "team_id": team,
            "is_ball": is_ball,
            "is_goalkeeper": gk,
            "x": float(x),
            "y": float(y),
            "z": 0.0,
            "speed": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "speed_source": "native",
            "ball_state": "alive",
            "team_attacking_direction": "ltr" if team == _HOME else "rtl",
            "confidence": None,
            "visibility": None,
            "source_provider": "synthetic",
            "is_goalkeeper_source": "native",
        }

    return pd.DataFrame(
        [
            row(None, None, False, 52.5, 34.0, is_ball=True),
            row(_HIGH_Y_PID, _HOME, False, 60.0, 56.0),  # high y
            row(_LOW_Y_PID, _HOME, False, 60.0, 12.0),  # low y, mirrored
            row(21, 2, False, 70.0, 56.0),
            row(22, 2, False, 70.0, 12.0),
            row(1, _HOME, True, 5.0, 34.0),
            row(2, 2, True, 100.0, 34.0),
        ]
    )


def test_high_y_player_receives_high_y_threat():
    """PRE-FIX THIS INVERTS: the raw interpolator hands back the y-mirrored surface."""
    out = compute_player_influence(_frame(), _y_ramp_xt(), attacking_team_id=_HOME, attacks_rtl=False)
    high = out[_HIGH_Y_PID].off_ball_xt
    low = out[_LOW_Y_PID].off_ball_xt

    assert np.isfinite([high, low]).all()
    assert max(high, low) > 0.0, "no off-ball xT for either player - comparison is vacuous"
    assert high > low, (
        f"threat grid is y-mirrored: the player at y=56 scored {high:.6g}, not above the "
        f"player at y=12 ({low:.6g}), under a grid whose value increases with y"
    )


def test_away_attack_reflects_the_threat_grid_on_BOTH_axes():
    """The away flip is a 180-degree POINT reflection (ADR-028), not an x-only mirror.

    The first repair flipped only ``[:, ::-1]``, which is exact for a y-symmetric grid --
    the reason it passed the test above, where the attacking team is HOME and the flip is
    never taken. Here the AWAY team attacks, so the flip IS taken: under the y-ramp grid,
    frame y=56 is action-LTR y=12, so the LOW-y-scoring player in the ramp must be the one
    standing at frame HIGH y. An x-only flip leaves the rows untouched and inverts this.
    """
    frame = _frame()
    # Give the away team the two mirrored outfielders so it is the one being valued.
    frame.loc[frame["player_id"].isin([_HIGH_Y_PID, _LOW_Y_PID]), "team_id"] = 2
    frame.loc[frame["player_id"].isin([21, 22]), "team_id"] = _HOME

    out = compute_player_influence(frame, _y_ramp_xt(), attacking_team_id=2, attacks_rtl=True)
    at_frame_high_y = out[_HIGH_Y_PID].off_ball_xt
    at_frame_low_y = out[_LOW_Y_PID].off_ball_xt

    assert np.isfinite([at_frame_high_y, at_frame_low_y]).all()
    assert max(at_frame_high_y, at_frame_low_y) > 0.0, "no off-ball xT for either player - vacuous"
    assert at_frame_low_y > at_frame_high_y, (
        f"threat grid is not y-reflected for the away team: the player at frame y=12 "
        f"(action-LTR y=56, the HIGH end of the ramp) scored {at_frame_low_y:.6g}, not above "
        f"the player at frame y=56 (action-LTR y=12) at {at_frame_high_y:.6g}"
    )


def test_unfitted_xt_fails_loud():
    """The shared guard now applies here too (it previously would not have raised)."""
    with pytest.raises(NotFittedError):
        compute_player_influence(_frame(), ExpectedThreat(), attacking_team_id=_HOME, attacks_rtl=False)
