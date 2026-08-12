"""TF-4 toward-goal orientation authority + the mislabelled-frame guard (ADR-041, S13).

TF-4 was the LAST module keyed on home/away identity for orientation, while
``acting_team_attacks_rtl`` already had 7 production call sites (``features.py:1720``,
``:2126``, ``:4123``, ``:4765``, ``utils.py:854``, ``_gk_geometry.py:196``,
``_kernels.py:877``). Re-keying it closes that inconsistency rather than creating one.

The re-key was only safe once ``validate_period_directions`` started rejecting frames whose
per-team labels are physically impossible: ``_validate_ltr`` alone accepts every row being
``"ltr"`` (it merely requires that ``"ltr"`` appears), and on such frames
``acting_team_attacks_rtl`` silently resolves to "no flip" for the away team.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

_HOME, _AWAY = 1, 2
_N = 5
_DUR = 1.5


def _frames(*, away_direction: str = "rtl", home_direction: str = "ltr") -> pd.DataFrame:
    """Home runner advances +x; away runner advances -x. Both move toward their OWN goal."""
    rows = []
    for fi in range(_N):
        t = fi * (_DUR / (_N - 1))
        frac = fi / (_N - 1)
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=fi + 1,
                time_seconds=t,
                frame_rate=(_N - 1) / _DUR,
                player_id=np.nan,
                team_id=np.nan,
                is_ball=True,
                is_goalkeeper=False,
                x=50.0,
                y=34.0,
                ball_state="alive",
                team_attacking_direction=None,
                source_provider="synthetic",
            )
        )
        movers = [
            # (pid, team, x, direction) -- each advances 6 m toward the goal it attacks
            (10, _HOME, 40.0, home_direction),  # actor (home)
            (11, _HOME, 45.0 + 6.0 * frac, home_direction),  # home runner: +x
            (20, _AWAY, 60.0, away_direction),  # actor (away)
            (21, _AWAY, 55.0 - 6.0 * frac, away_direction),  # away runner: -x
        ]
        for pid, team, x, direction in movers:
            rows.append(
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=fi + 1,
                    time_seconds=t,
                    frame_rate=(_N - 1) / _DUR,
                    player_id=pid,
                    team_id=team,
                    is_ball=False,
                    is_goalkeeper=False,
                    x=x,
                    y=34.0,
                    ball_state="alive",
                    team_attacking_direction=direction,
                    source_provider="synthetic",
                )
            )
    return pd.DataFrame(rows)


def _action(team_id: int, actor: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [1],
            "period_id": [1],
            "action_id": [0],
            "team_id": [team_id],
            "player_id": [actor],
            "type_id": [0],
            "result_id": [1],
            "start_x": [50.0],
            "start_y": [34.0],
            "end_x": [60.0],
            "end_y": [34.0],
            "time_seconds": [_DUR],
        }
    )


class TestAuthoritiesAgree:
    """On CORRECTLY-labelled frames the direction authority reproduces home/away keying."""

    @pytest.mark.parametrize(("team", "actor", "runner"), [(_HOME, 10, 11), (_AWAY, 20, 21)])
    def test_runner_toward_own_goal_counts(self, team, actor, runner):
        out = _off_ball_runs_kernel(_action(team, actor), _frames(), home_team_id=_HOME)
        assert out["n_off_ball_runners_pre_window"].iloc[0] >= 1, (
            "no runner detected - the toward-goal assertion would be vacuous"
        )
        assert out["n_off_ball_runners_toward_goal_pre_window"].iloc[0] == 1, (
            f"team {team}'s runner advances toward the goal it attacks and must count"
        )

    def test_home_and_away_are_symmetric(self):
        """Neither team is privileged: the same physical situation scores the same."""
        home = _off_ball_runs_kernel(_action(_HOME, 10), _frames(), home_team_id=_HOME)
        away = _off_ball_runs_kernel(_action(_AWAY, 20), _frames(), home_team_id=_HOME)
        assert (
            home["n_off_ball_runners_toward_goal_pre_window"].iloc[0]
            == away["n_off_ball_runners_toward_goal_pre_window"].iloc[0]
            == 1
        )


class TestSelfContradictionRaises:
    """ONLY a team that contradicts itself is impossible. Everything else is accepted."""

    def test_team_carrying_both_directions_raises(self):
        frames = _frames()
        away_players = frames["team_id"].eq(_AWAY) & ~frames["is_ball"].astype(bool)
        half = frames.index[away_players][: int(away_players.sum() // 2)]
        frames.loc[half, "team_attacking_direction"] = "ltr"  # same team, both directions
        with pytest.raises(ValueError, match="carry BOTH 'ltr' and 'rtl'"):
            _off_ball_runs_kernel(_action(_AWAY, 20), frames, home_team_id=_HOME)

    def test_error_names_the_orientation_helpers(self):
        """The message must be actionable (ADR-029 helpers), not just a rejection."""
        frames = _frames()
        away_players = frames["team_id"].eq(_AWAY) & ~frames["is_ball"].astype(bool)
        half = frames.index[away_players][: int(away_players.sum() // 2)]
        frames.loc[half, "team_attacking_direction"] = "ltr"
        with pytest.raises(ValueError, match="orient_frames_to_ltr"):
            _off_ball_runs_kernel(_action(_AWAY, 20), frames, home_team_id=_HOME)

    def test_correctly_labelled_frames_do_not_raise(self):
        """Non-vacuity: the guard must not reject valid input."""
        out = _off_ball_runs_kernel(_action(_AWAY, 20), _frames(), home_team_id=_HOME)
        assert out["n_off_ball_runners_pre_window"].iloc[0] >= 1


class TestUnorientedFramesAreAccepted:
    """Unoriented != mislabelled. These three shapes are produced BY THE LIBRARY.

    An earlier draft of the guard rejected all of them, regressing paths that had always
    worked: ``snapshot_to_tracking_frames`` (``_snapshot.py:92``, uniform "ltr" because
    snapshot frames are already action-LTR), ``output_convention="absolute_frame"``
    SkillCorner/Metrica (``skillcorner.py:282`` / ``metrica.py:180``, all-null -- the shape
    ``scripts/_loader_pining.py`` feeds the training corpora), and period-5 shootouts
    (``direction.py:29``). These tests pin that they are ACCEPTED.
    """

    def test_all_null_direction_is_accepted(self):
        """absolute_frame convention: no orientation asserted -- accepted, but now AUDIBLE.

        ADR-028 D2 made the unresolved orientation warn instead of passing silently. Accepting
        the shape and announcing it are not in tension: the values are unchanged, and a consumer
        who cannot tolerate unoriented geometry can escalate the category. Asserted with
        ``pytest.warns`` rather than filtered, so the warning is part of the pinned contract --
        if a future change makes this path silent again, this test fails.
        """
        from silly_kicks.tracking import OrientationUnresolvedWarning

        frames = _frames()
        frames["team_attacking_direction"] = None
        with pytest.warns(OrientationUnresolvedWarning):
            out = _off_ball_runs_kernel(_action(_AWAY, 20), frames, home_team_id=_HOME)
        assert out["n_off_ball_runners_pre_window"].iloc[0] >= 1

    def test_uniform_label_is_accepted(self):
        """snapshot_to_tracking_frames labels every player "ltr" on purpose."""
        out = _off_ball_runs_kernel(_action(_AWAY, 20), _frames(away_direction="ltr"), home_team_id=_HOME)
        assert out["n_off_ball_runners_pre_window"].iloc[0] >= 1

    def test_period_five_is_exempt(self):
        """PSO orientation is undefined; direction.py:29 excludes it, so must this guard."""
        frames = _frames()
        frames["period_id"] = 5
        frames["team_attacking_direction"] = None
        actions = _action(_AWAY, 20)
        actions["period_id"] = 5
        out = _off_ball_runs_kernel(actions, frames, home_team_id=_HOME)
        assert len(out) == 1  # ran without raising

    def test_unoriented_behaviour_is_pinned_unknown_not_zero(self):
        """DOCUMENTED, not incidental: with no orientation asserted, direction is UNKNOWN.

        The away runner advances -x. On correctly-labelled frames that counts as toward-goal.
        On unoriented frames the direction does not resolve at all, so as of 4.80.0 the count
        is ``pd.NA`` -- not the ``0`` this test previously pinned.

        That earlier ``0`` is the defect the D3 re-key removes, in miniature. It is
        indistinguishable from a genuine "the runner ran away from goal", so a corpus of
        unoriented frames reported confident zeros that a consumer had no way to tell from a
        measurement. ``pd.NA`` says the one true thing: nobody knows which way this team was
        attacking.

        PER-COLUMN, deliberately: runner count, max displacement and mean speed are
        displacement MAGNITUDES and stay live -- only the direction-dependent column goes NA.
        The companion assertion below pins that, so a future "fix" that NaNs the whole row
        fails here.
        """
        from silly_kicks.tracking import OrientationUnresolvedWarning

        oriented = _off_ball_runs_kernel(_action(_AWAY, 20), _frames(), home_team_id=_HOME)
        unoriented_frames = _frames()
        unoriented_frames["team_attacking_direction"] = None
        # The unoriented leg warns by design (ADR-028 D2); the oriented leg above must NOT, which
        # is what makes this pairing a real discriminator rather than a blanket tolerance.
        with pytest.warns(OrientationUnresolvedWarning):
            unoriented = _off_ball_runs_kernel(_action(_AWAY, 20), unoriented_frames, home_team_id=_HOME)
        assert oriented["n_off_ball_runners_toward_goal_pre_window"].iloc[0] == 1
        assert pd.isna(unoriented["n_off_ball_runners_toward_goal_pre_window"].iloc[0])
        # The flip-invariant columns must survive -- otherwise this is a row-level refusal
        # wearing a per-column label.
        assert unoriented["n_off_ball_runners_pre_window"].iloc[0] >= 1
        assert np.isfinite(unoriented["max_off_ball_run_displacement_pre_window"].iloc[0])
