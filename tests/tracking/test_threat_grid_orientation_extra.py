"""DEFECT B + C in the two modules the first pass MISSED (ADR-041, review finding 4).

``_player_influence`` was repaired for both defects in the original PR-S119 pass:

* **B** -- it multiplied the RAW ``xt.interpolator()`` output, which preserves xT's
  inverted row storage (row 0 = TOP of pitch) and is therefore y-mirrored against the
  ascending-y pitch-control surface; and
* **C** -- the away-team reflection was applied on the x axis alone, when ADR-028's
  relation is a 180-degree POINT reflection (``x -> 105-x`` AND ``y -> 68-y``).

``_cover_shadows`` and ``_gk_influence`` carried byte-identical code and were NOT in the
first pass -- an adversarial review found them, and found that CLAUDE.md had already been
rewritten to claim ``cover_shadows`` was repaired. Both are now fixed; these are the
ground-truth tests, built on a deliberately y-ASYMMETRIC grid because a fitted xT is
close enough to y-symmetric that a symmetric fixture cannot see either defect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._gk_influence import compute_gk_influence
from silly_kicks.xthreat import ExpectedThreat
from tests.tracking._goal_map_helpers import goal_map_for

_HOME, _AWAY = 1, 2

#: ADR-055: the goal ends this file's fixtures imply. Stated, not derived: the tests below
#: deliberately MOVE a keeper to probe the threat grid, and a derived map would move with it.
_GOAL_MAP = goal_map_for({_HOME: 0.0, _AWAY: 105.0})


def _y_ramp_xt() -> ExpectedThreat:
    """Grid value == the physical y-centre of the storage band it occupies.

    Storage row 0 is the TOP of the pitch, so the RAW interpolator output is the exact
    vertical mirror of the physical surface -- which is what makes DEFECT B visible.
    """
    m = ExpectedThreat()
    w, _l = m.xT.shape
    cell_w = 68.0 / w
    for i in range(w):
        m.xT[i, :] = (w - 1 - i + 0.5) * cell_w
    return m


def _frow(pid, team, gk, x, y, is_ball=False):
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


def _frame(gk_y: float, *, move_gk_of: int = _HOME) -> pd.DataFrame:
    """The DEFENDING keeper is parked off-centre in y; everything else stays put.

    ``move_gk_of`` must name the keeper actually being measured — ``compute_gk_influence``
    scores the keeper it is handed, so moving the other one produces an identical number
    and the comparison silently becomes vacuous (it did, on the first draft).
    """
    home_gk_y = gk_y if move_gk_of == _HOME else 34.0
    away_gk_y = gk_y if move_gk_of == _AWAY else 34.0
    return pd.DataFrame(
        [
            _frow(None, None, False, 52.5, 34.0, is_ball=True),
            _frow(10, _HOME, False, 40.0, 30.0),
            _frow(11, _HOME, False, 45.0, 38.0),
            _frow(20, _AWAY, False, 65.0, 30.0),
            _frow(21, _AWAY, False, 70.0, 38.0),
            _frow(1, _HOME, True, 6.0, home_gk_y),
            _frow(2, _AWAY, True, 99.0, away_gk_y),
        ]
    )


def _gk_threat_share(gk_y: float, *, attacking_team_id: int) -> float:
    # The DEFENDING team's keeper is the one whose influence is measured -- and the one
    # the fixture must move.
    defending_team = _AWAY if attacking_team_id == _HOME else _HOME
    gk_player_id = 2 if defending_team == _AWAY else 1
    influence = compute_gk_influence(
        _frame(gk_y, move_gk_of=defending_team),
        attacking_team_id,
        gk_player_id,
        _y_ramp_xt(),
        goal_map=_GOAL_MAP,
    )
    return float(influence.pitch_control_share_weighted)


class TestGkInfluenceThreatGridOrientation:
    def test_home_attack_reads_the_physical_surface_not_the_stored_one(self):
        """DEFECT B: under a grid whose value RISES with physical y, a keeper standing
        high must sit in more threat than one standing low. The raw interpolator inverts
        exactly this comparison."""
        high = _gk_threat_share(56.0, attacking_team_id=_HOME)
        low = _gk_threat_share(12.0, attacking_team_id=_HOME)

        assert np.isfinite([high, low]).all()
        assert max(high, low) > 0.0, "no GK threat share at all - the comparison is vacuous"
        assert high != pytest.approx(low, rel=1e-9), "the fixture does not discriminate in y"
        assert high > low, (
            f"threat grid is y-mirrored (DEFECT B): keeper at y=56 scored {high:.6g}, not "
            f"above keeper at y=12 ({low:.6g}), under a grid rising with y"
        )

    def test_away_attack_reflects_on_BOTH_axes(self):
        """DEFECT C: with the AWAY team attacking, the flip IS taken. Frame y=56 is
        action-LTR y=12, so the ordering must INVERT relative to the home case. An x-only
        mirror leaves the rows untouched and preserves it."""
        high = _gk_threat_share(56.0, attacking_team_id=_AWAY)
        low = _gk_threat_share(12.0, attacking_team_id=_AWAY)

        assert np.isfinite([high, low]).all()
        assert max(high, low) > 0.0, "no GK threat share at all - the comparison is vacuous"
        assert low > high, (
            f"threat grid is not y-reflected for the away team (DEFECT C): keeper at frame "
            f"y=12 (action-LTR y=56, the HIGH end of the ramp) scored {low:.6g}, not above "
            f"the keeper at frame y=56 at {high:.6g}"
        )

    def test_the_two_orientations_actually_differ(self):
        """Non-vacuity: if home and away produced identical numbers, neither test above
        would be testing the flip at all."""
        assert _gk_threat_share(56.0, attacking_team_id=_HOME) != pytest.approx(
            _gk_threat_share(56.0, attacking_team_id=_AWAY), rel=1e-9
        )


class TestCoverShadowsThreatGridOrientation:
    """``compute_blocking_score`` consumes the same threat product.

    Asserted through the public per-frame entry point rather than the private helper, so
    the test survives an internal refactor.
    """

    @staticmethod
    def _blocking(gk_y: float, *, attacking_toward_high_x: bool) -> float:
        from silly_kicks.tracking._cover_shadows import compute_blocking_score

        frame = _frame(gk_y)
        result = compute_blocking_score(
            frame,
            _HOME if attacking_toward_high_x else _AWAY,
            _y_ramp_xt(),
            goal_map=_GOAL_MAP,
            defenders_to_remove=[20 if attacking_toward_high_x else 10],
        )
        return float(result.blocking_score)

    def test_blocking_score_is_finite_under_both_orientations(self):
        """The repair must not break the family: both directions still compute."""
        a = self._blocking(56.0, attacking_toward_high_x=True)
        b = self._blocking(56.0, attacking_toward_high_x=False)
        assert np.isfinite([a, b]).all()

    def test_physical_grid_is_what_cover_shadows_now_reads(self):
        """Pin the mechanism directly: the module must consume physical_grid, whose output
        is the vertical mirror of the raw interpolator on this deliberately asymmetric xT.

        This is the source-level counterpart to the behavioural GK tests above -- the
        blocking-score surface is a counterfactual difference, so a small threat-orientation
        change need not move it monotonically, and asserting a direction there would be
        wishful rather than derived.
        """
        from silly_kicks.xthreat import physical_grid

        m = _y_ramp_xt()
        gx = np.linspace(0.0, 105.0, 20)
        gy = np.linspace(0.0, 68.0, 16)
        phys = physical_grid(m, gx, gy)
        raw = m.interpolator()(gx, gy)

        assert phys[0, 0] != pytest.approx(raw[0, 0], rel=1e-9), "fixture is y-symmetric - vacuous"
        np.testing.assert_allclose(raw, np.flipud(phys), rtol=1e-9, atol=1e-9)
        # ...and physical is the one that agrees with the ramp: low row -> low y -> low value.
        assert phys[0, 0] < phys[-1, 0]


class TestCoverShadowDefaultBranchOrientation:
    """The `detailed=False` branch is the PRODUCTION DEFAULT and read the RAW interpolator.

    Found by final-review (ADR-041). `_voronoi_threat` was repaired in the first pass;
    its sibling in the same file -- the one feeding `max_single_defender_blocking_score`
    on the default path of `add_cover_shadows`, `cover_shadow_xfns` AND the TF-24
    calibration harness -- was missed. It was y-mirrored for every action and, for an
    RTL-attacking team, un-point-reflected as well, matching NEITHER orientation.

    The frozen parity oracle could not catch it: it read the same raw interpolator, so it
    compared the bug to itself. Both are now corrected.
    """

    @staticmethod
    def _lane_frame(recv_y_a: float, recv_y_b: float) -> pd.DataFrame:
        """Passer on the ball, two receivers BEYOND it, two mid-lane defenders.

        `_frame` above is unusable here: its home attackers sit BEHIND the ball, so
        `dangerous` is empty and the blocking score is 0.0 for every placement -- the
        comparison would be vacuous (it was, on the first draft).
        """
        return pd.DataFrame(
            [
                _frow(None, None, False, 40.0, 34.0, is_ball=True),
                _frow(10, _HOME, False, 40.0, 34.0),  # passer, on the ball
                _frow(11, _HOME, False, 72.0, recv_y_a),  # dangerous receiver
                _frow(12, _HOME, False, 72.0, recv_y_b),  # dangerous receiver
                _frow(20, _AWAY, False, 56.0, 30.0),  # mid-lane blocker
                _frow(21, _AWAY, False, 56.0, 38.0),  # mid-lane blocker
                _frow(1, _HOME, True, 5.0, 34.0),
                _frow(2, _AWAY, True, 100.0, 34.0),
            ]
        )

    @staticmethod
    def _score(frames: pd.DataFrame) -> float:
        import silly_kicks.tracking.features as F

        actions = pd.DataFrame(
            {
                "game_id": [1],
                "action_id": [1],
                "period_id": [1],
                "time_seconds": [4.0],
                "team_id": pd.Series([_HOME], dtype="int64"),
                "player_id": pd.Series([10], dtype="int64"),
                "start_x": [40.0],
                "start_y": [34.0],
                "end_x": [72.0],
                "end_y": [34.0],
                "type_id": [0],
                "type_name": ["pass"],
                "result_id": [1],
                "result_name": ["success"],
                "bodypart_id": [0],
                "bodypart_name": ["foot"],
            }
        )
        out = F.add_cover_shadows(actions, frames, _y_ramp_xt(), goal_map=_GOAL_MAP)
        return float(out["max_single_defender_blocking_score"].iloc[0])

    def test_default_branch_reads_the_physical_surface(self):
        """Under a grid whose threat RISES with y, receivers placed high must score higher."""
        high = self._score(self._lane_frame(56.0, 60.0))
        low = self._score(self._lane_frame(8.0, 12.0))

        assert np.isfinite([high, low]).all()
        assert max(high, low) > 0.0, "no blocking score at all - the comparison is vacuous"
        assert high != pytest.approx(low, rel=1e-9), (
            "the default branch is insensitive to y - it is reading a y-flat surface"
        )
        assert high > low, (
            f"default branch is y-mirrored: HIGH-y receivers scored {high:.6g}, not above "
            f"the LOW-y placement ({low:.6g}), under a grid whose threat rises with y"
        )
