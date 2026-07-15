"""The S1 geometry gate (spec 4.4). Registered thresholds are calibrated on the public 10.

CRITICAL: the last test pins a LIMITATION on purpose. This gate catches catastrophic sign/origin
breaks; it CANNOT see a pitch-dimension error (measured: 0.00095 vs a clean 0.00086). Nor can
action-frame co-location, since events and tracking read the same metadata and move together.
If someone later 'fixes' that test, they have misunderstood the gate.
"""

import numpy as np
import pandas as pd

from silly_kicks.tracking.skillcorner import (
    _BALL_OFF_PITCH_RATE_MAX,
    _PLAYER_OFF_PITCH_RATE_MAX,
    _TOL_BALL,
    convert_to_frames,
    geometry_rate_gate,
)


def test_tol_ball_is_calibrated_below_the_measured_headroom():
    """Calibration-PROVENANCE check (not a firing check -- the gate's firing is covered by
    test_a_wild_ball_is_excluded). The largest real ball excursion measured on the public 10 is
    9.00 m; the tolerance must sit above that (so clean data never trips) yet far below 30.0 m
    (which could not trip on any real break). This pins the calibrated VALUE, not the behaviour."""
    assert 9.0 < _TOL_BALL < 30.0
    assert _TOL_BALL == 15.0  # the specific calibrated value (spec 4.4)


def _rows(player_exc: np.ndarray, ball_exc: np.ndarray) -> pd.DataFrame:
    """Frames whose SPADL x sits `exc` metres beyond the goal line (0 = on-pitch)."""
    p = pd.DataFrame({"x": 105.0 + player_exc, "y": 34.0, "is_ball": False})
    b = pd.DataFrame({"x": 105.0 + ball_exc, "y": 34.0, "is_ball": True})
    return pd.concat([p, b], ignore_index=True)


def test_clean_match_passes():
    player = np.zeros(100_000)
    player[:86] = 5.0
    ball = np.zeros(5_000)
    report = geometry_rate_gate(_rows(player, ball))
    assert report.excluded is False


def test_catastrophic_break_is_excluded():
    player = np.zeros(100_000)
    player[:34_000] = 20.0
    report = geometry_rate_gate(_rows(player, np.zeros(5_000)))
    assert report.excluded is True
    assert "player" in report.reason


def test_a_wild_ball_is_excluded():
    ball = np.zeros(5_000)
    ball[:50] = 25.0
    report = geometry_rate_gate(_rows(np.zeros(100_000), ball))
    assert report.excluded is True
    assert "ball" in report.reason


def test_a_pitch_dimension_error_is_INVISIBLE_to_this_gate():
    """PINNED LIMITATION -- do not 'fix' this test.

    A 4 m pitch-length error produces player_frac(>3m) = 0.00095 against a clean worst of
    0.00086. It does not, and cannot, trip. The only instruments for pitch dims are provenance
    (spec 1.6.2) and asking SkillCorner. A gate that appeared to cover this would be worse than
    no gate at all.
    """
    player = np.zeros(100_000)
    player[:95] = 3.5
    report = geometry_rate_gate(_rows(player, np.zeros(5_000)))
    assert report.excluded is False
    # Referenced so the calibrated thresholds are pinned as import contract, not dead symbols.
    assert 0.0 < _PLAYER_OFF_PITCH_RATE_MAX < 1.0
    assert 0.0 < _BALL_OFF_PITCH_RATE_MAX < 1.0


def test_convert_to_frames_reports_the_exclusion():
    """The gate must reach a CONSUMER. A pure function nobody calls excludes nothing.

    KILL-LINE: delete `geometry_excluded=gate.excluded` in convert_to_frames -> this fails.

    NOTE on the off-pitch offset: raw x=52.0 maps to SPADL x=104.5 (the goal-line region on a
    105 m pitch). The plan's +20 m offset lands players at SPADL x=124.5, which trips the
    SEPARATE catastrophic-coord backstop in derive_goalkeepers (bound x<=120) BEFORE the gate
    runs. +12 m (SPADL x=116.5) keeps them in the S1 warn-band -- off-pitch by >3 m yet below the
    crash bound -- so the gate can see them. The gate uses the pitch tolerance (3 m), independent
    of derive_goalkeepers' wide bound.
    """
    n = 1000
    rows = []
    for i in range(n):
        off = 12.0 if i < 340 else 0.0
        rows.append(
            {
                "match_id": "m1",
                "period": 1,
                "frame": i,
                "timestamp": float(i),
                "player_id": f"p{i % 22}",
                "team_id": "A" if i % 2 else "B",
                "is_goalkeeper": False,
                "x": 52.0 + off,
                "y": 0.0,
                "ball_x": 0.0,
                "ball_y": 0.0,
                "ball_z": 0.0,
                "is_visible": True,
                "frame_rate": 10.0,
                "pitch_length": 105.0,
                "pitch_width": 68.0,
            }
        )
    _frames, report = convert_to_frames(pd.DataFrame(rows), home_team_id="A", output_convention="absolute_frame")
    assert report.geometry_excluded is True
    assert "player" in report.geometry_reason
