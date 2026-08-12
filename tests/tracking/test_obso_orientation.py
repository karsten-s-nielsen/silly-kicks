"""OBSO per-action orientation (DEFECT A, ADR-041 / ADR-028 amendment).

``convert_to_frames`` emits home-attacks-right frames; SPADL actions are per-acting-team
LTR. For an AWAY action the two conventions are a 180-degree point reflection apart, and
OBSO handled neither side of that:

* ``add_obso`` passed the raw action-LTR ``end_x``/``end_y`` as ``target_position`` while
  the pitch-control surfaces come from home-attacks-right frames, so the away target was
  sampled at the reflected point; and
* the EPV grid (synthetic ramp or injected xT) always increases toward +x, i.e. toward the
  HOME team's attacked goal, so away actions were valued toward their OWN goal.

``home_team_id`` was accepted by ``_precompute_obso_lookup`` and never read -- a dead
parameter. ADR-028 had classified obso as "self-reconciling"; it was not, it simply never
handled orientation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import features as F

pytestmark = pytest.mark.filterwarnings("ignore::silly_kicks.tracking.SyntheticEPVWarning")

_HOME, _AWAY = 1, 2


def _frow(pid, team, gk, x, y, t, *, is_ball=False, vx=1.4, vy=0.0):
    return {
        "game_id": 1,
        "period_id": 1,
        "frame_id": round(t * 25),
        "time_seconds": float(t),
        "frame_rate": 25.0,
        "player_id": pid,
        "team_id": team,
        "is_ball": is_ball,
        "is_goalkeeper": gk,
        "x": float(np.clip(x, 0.5, 104.5)),
        "y": float(np.clip(y, 0.5, 67.5)),
        "z": 0.0,
        "speed": float(np.hypot(vx, vy)),
        "vx": float(vx),
        "vy": float(vy),
        "speed_source": "native",
        "ball_state": "alive",
        # Home attacks right; the AWAY team attacks right-to-left.
        "team_attacking_direction": "ltr" if team == _HOME else "rtl",
        "confidence": None,
        "visibility": None,
        "source_provider": "gradientsports",
        "is_goalkeeper_source": "native",
    }


def _away_frames(ball_x: float = 52.5, ball_y: float = 34.0) -> pd.DataFrame:
    """Away team spread across the pitch so both candidate targets are reachable."""
    rows = []
    for t in (9.6, 9.8, 10.0, 19.6, 19.8, 20.0):
        rows.append(_frow(None, None, False, ball_x, ball_y, t, is_ball=True))
        # AWAY (acting) players either side of the ball, symmetric in x about the centre
        for pid, (x, y) in {
            21: (30.0, 30.0),
            22: (45.0, 40.0),
            23: (70.0, 30.0),
            24: (88.0, 38.0),
        }.items():
            rows.append(_frow(pid, _AWAY, False, x, y, t, vx=-1.4))
        # HOME defenders, deliberately symmetric in x so they cannot bias the comparison
        for pid, (x, y) in {
            11: (36.0, 46.0),
            12: (52.0, 22.0),
            13: (66.0, 46.0),
            14: (82.0, 22.0),
        }.items():
            rows.append(_frow(pid, _HOME, False, x, y, t, vx=1.4))
        rows.append(_frow(1, _HOME, True, 5.0, 34.0, t))
        rows.append(_frow(2, _AWAY, True, 100.0, 34.0, t))
    return pd.DataFrame(rows)


def _away_actions() -> pd.DataFrame:
    """Two AWAY passes, mirrored in action-LTR x: one FORWARD, one BACKWARD.

    In action-LTR the acting (away) team attacks x=105, so end_x=90 is a forward ball into
    the final third and end_x=15 is a backward ball toward its own goal. A correctly
    oriented OBSO must value the forward one higher.
    """
    return pd.DataFrame(
        {
            "game_id": [1, 1],
            "action_id": [10, 11],
            "period_id": [1, 1],
            "time_seconds": [10.0, 20.0],
            "team_id": pd.Series([_AWAY, _AWAY], dtype="int64"),
            "player_id": pd.Series([22, 22], dtype="int64"),
            "start_x": [52.5, 52.5],
            "start_y": [34.0, 34.0],
            "end_x": [90.0, 15.0],
            "end_y": [34.0, 34.0],
            "type_id": [0, 0],
            "type_name": ["pass", "pass"],
            "result_id": [1, 1],
            "result_name": ["success", "success"],
            "bodypart_id": [0, 0],
            "bodypart_name": ["foot", "foot"],
        }
    )


def _away_control_at_low_x() -> pd.DataFrame:
    """Away outfielders concentrated in frame LOW-x (the end they attack)."""
    frames = _away_frames()
    mask = frames["team_id"].eq(_AWAY) & ~frames["is_ball"].astype(bool) & ~frames["is_goalkeeper"].astype(bool)
    out = frames.copy()
    out.loc[mask, "x"] = 15.0
    out.loc[mask, "y"] = [30.0, 34.0, 38.0, 42.0] * (int(mask.sum()) // 4)
    return out


class TestEpvDirection:
    """The EPV grid must be x-flipped for an away-acting team (DEFECT A, half 1).

    NOTE this cannot be tested through ``obso_actual``: the target reflection and the grid
    flip are JOINTLY INVARIANT at the target itself, since
    ``epv_rtl[col(105 - x)] == epv[col(x)]`` by construction. The EPV orientation is
    therefore observable only where the whole surface is scanned -- i.e. ``obso_optimal``.
    """

    @staticmethod
    def _optimal_with_epv_mass_at(col_lo: int, col_hi: int) -> float:
        epv = np.zeros((68, 104))
        epv[:, col_lo:col_hi] = 1.0
        actions = _away_actions().iloc[[0]].reset_index(drop=True)
        out = F.add_obso(actions, _away_control_at_low_x(), epv_grid=epv)
        return float(out["obso_optimal"].iloc[0])

    def test_away_epv_is_mirrored_into_frame_orientation(self):
        # Away control sits at frame x~15. In the ATTACK-LTR grid the away team's attacked
        # goal is high-x, so EPV mass at grid high-x must land on that control once
        # flipped; mass at grid low-x must land at frame high-x, where they have none.
        toward_attacked_goal = self._optimal_with_epv_mass_at(84, 104)
        toward_own_goal = self._optimal_with_epv_mass_at(0, 20)

        assert np.isfinite([toward_attacked_goal, toward_own_goal]).all()
        assert max(toward_attacked_goal, toward_own_goal) > 0.0, (
            "no OBSO under either EPV placement - the comparison is vacuous"
        )
        assert toward_attacked_goal > toward_own_goal, (
            f"EPV is not mirrored for the away team: attack-side mass scored "
            f"{toward_attacked_goal:.6g}, own-goal-side mass {toward_own_goal:.6g}"
        )


class TestEpvIsReflectedOnBothAxes:
    """The EPV reflection is a 180-degree POINT reflection, not an x-only mirror.

    ADR-028 defines the action-LTR <-> frame relation as ``x -> 105-x`` AND ``y -> 68-y``.
    The first repair of DEFECT A flipped the EPV grid on the x axis alone, which is exact
    only for a y-symmetric grid -- true of the synthetic ramp default, and *approximately*
    true of a fitted xT surface, which is precisely why it survived the x-axis tests. On a
    y-ASYMMETRIC grid (an injected xT, or any real EPV) the away team's threat was read off
    the wrong half of the pitch.
    """

    @staticmethod
    def _optimal_with_epv_mass_in_rows(row_lo: int, row_hi: int) -> float:
        """EPV mass confined to a band of ATTACK-LTR y-rows (row 0 = y=0, ascending)."""
        epv = np.zeros((68, 104))
        epv[row_lo:row_hi, :] = 1.0
        actions = _away_actions().iloc[[0]].reset_index(drop=True)
        frames = _away_frames()
        # Away control concentrated at frame y=48 (== action-LTR y=20, i.e. LOW rows).
        away_out = frames["team_id"].eq(_AWAY) & ~frames["is_ball"].astype(bool) & ~frames["is_goalkeeper"].astype(bool)
        moved = frames.copy()
        moved.loc[away_out, "y"] = 48.0
        out = F.add_obso(actions, moved, epv_grid=epv)
        return float(out["obso_optimal"].iloc[0])

    def test_away_epv_is_mirrored_on_the_y_axis_too(self):
        # Control sits at frame y=48. Reflected, that is action-LTR y=20 -> LOW attack-LTR
        # rows. So EPV mass in low rows must out-score mass in high rows. Under an x-only
        # flip the row index is untouched and the preference inverts.
        mass_low_rows = self._optimal_with_epv_mass_in_rows(0, 24)
        mass_high_rows = self._optimal_with_epv_mass_in_rows(44, 68)

        assert np.isfinite([mass_low_rows, mass_high_rows]).all()
        assert max(mass_low_rows, mass_high_rows) > 0.0, (
            "no OBSO under either EPV placement - the comparison is vacuous"
        )
        assert mass_low_rows > mass_high_rows, (
            f"EPV is not y-reflected for the away team: mass on the rows that reflect ONTO "
            f"the acting team's control scored {mass_low_rows:.6g}, not above the opposite "
            f"band at {mass_high_rows:.6g}"
        )


class TestHomeUnaffected:
    def test_home_actions_are_byte_identical_to_the_pre_repair_path(self):
        """Home rows never flip, so the repair must not move them at all.

        Asserted structurally: a home action's OBSO must equal the value obtained with the
        orientation machinery bypassed, i.e. an explicitly unflipped grid on a frame whose
        directions are all "ltr".
        """
        actions = _away_actions().assign(team_id=pd.Series([_HOME, _HOME], dtype="int64"))
        frames = _away_frames()
        all_ltr = frames.assign(team_attacking_direction="ltr")

        out = F.add_obso(actions, frames)
        ref = F.add_obso(actions, all_ltr)
        a = out["obso_actual"].to_numpy(dtype=float)
        b = ref["obso_actual"].to_numpy(dtype=float)
        assert np.isfinite(a).any(), "no OBSO values - the comparison is vacuous"
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=0)


class TestTargetReprojection:
    """The target point itself must be reflected into frame coordinates for away actions."""

    @staticmethod
    def _actions_to(end_x: float, end_y: float) -> pd.DataFrame:
        a = _away_actions().iloc[[0]].copy()
        a["end_x"] = end_x
        a["end_y"] = end_y
        return a.reset_index(drop=True)

    def test_target_is_sampled_at_the_reflected_point(self):
        """An away action-LTR target of (90, 20) is frame (15, 48).

        Cluster the away team's control around frame (15, 48) and starve frame (90, 20);
        the action-LTR (90, 20) target must then score HIGHER than the action-LTR (15, 48)
        target, because the latter reflects to frame (90, 20) where the acting team has no
        control. Pre-fix the raw target is used and the preference inverts.
        """
        frames = _away_frames()
        # Concentrate AWAY control near frame (15, 48); remove it from frame (90, 20).
        moved = frames.copy()
        away_out = moved["team_id"].eq(_AWAY) & ~moved["is_ball"].astype(bool) & ~moved["is_goalkeeper"].astype(bool)
        moved.loc[away_out, "x"] = 15.0
        moved.loc[away_out, "y"] = 48.0

        near = F.add_obso(self._actions_to(90.0, 20.0), moved)
        far = F.add_obso(self._actions_to(15.0, 48.0), moved)
        near_v = float(near["obso_actual"].iloc[0])
        far_v = float(far["obso_actual"].iloc[0])

        assert np.isfinite([near_v, far_v]).all(), "no OBSO values - comparison is vacuous"
        assert near_v > far_v, (
            f"action-LTR (90,20) [-> frame (15,48), where the acting team controls] scored "
            f"{near_v:.6g}, not above action-LTR (15,48) [-> frame (90,20), empty] at "
            f"{far_v:.6g}: the target is not being re-projected"
        )
