"""Tests for resolve_restart_geometry (general restart-coordinate enrichment).

Spec: docs/superpowers/specs/2026-06-10-general-restart-coordinate-enrichment-design.md
"""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._gk_geometry import _tracking_ball_xy, resolve_restart_geometry

_GK, _CORNER_C, _THROW, _PEN, _PASS = 22, 5, 2, 12, 0


def _frame(**over):
    # `team_attacking_direction` is present so the acting team's direction RESOLVES. Without it
    # `acting_team_attacks_rtl` returns <NA> for every action and the tracking tier is skipped
    # entirely (4.80.0) -- which would make the tests below fail for a reason none of them is
    # about: they test ball-row selection, is_ball coercion and tier ordering, not orientation.
    # The unoriented-frames behaviour itself is pinned separately, in
    # test_off_ball_runs_orientation.py.
    base = dict(
        game_id=[9, 9],
        period_id=[1, 1],
        frame_id=[1250, 1250],
        time_seconds=[50.0, 50.0],
        team_id=[1, 0],
        player_id=[10, -1],
        is_goalkeeper=[False, False],
        is_ball=[False, True],
        x=[50.0, 104.5],
        y=[20.0, 0.5],
        source_provider=["gradientsports", "gradientsports"],
        team_attacking_direction=["ltr", None],
    )
    base.update(over)
    return pd.DataFrame(base)


def test_tracking_ball_xy_selects_ball_row():
    from silly_kicks.tracking._kernels import resolve_frame_ids_by_position

    a = pd.DataFrame(
        dict(
            game_id=[9],
            period_id=[1],
            action_id=[7],
            team_id=[1],
            type_id=[_CORNER_C],
            time_seconds=[50.0],
            start_x=[np.nan],
            start_y=[np.nan],
            end_x=[95.0],
            end_y=[10.0],
        )
    )
    # Minor-9: assert the fixture actually links first, so a linkage-fixture failure is
    # distinguishable from a ball-selection bug.
    assert np.isfinite(resolve_frame_ids_by_position(a, _frame(), links=None)[0]), "fixture linkage failed"
    xy = _tracking_ball_xy(a, _frame(), links=None)
    assert xy[0, 0] == pytest.approx(104.5)
    assert xy[0, 1] == pytest.approx(0.5)


def test_tracking_ball_xy_coerces_string_is_ball():
    # ADR-019: object/string is_ball must be coerced, not assumed bool (~is_ball no-op bug).
    a = pd.DataFrame(
        dict(
            game_id=[9],
            period_id=[1],
            action_id=[7],
            team_id=[1],
            type_id=[_CORNER_C],
            time_seconds=[50.0],
            start_x=[np.nan],
            start_y=[np.nan],
            end_x=[95.0],
            end_y=[10.0],
        )
    )
    fr = _frame(is_ball=["False", "True"])
    xy = _tracking_ball_xy(a, fr, links=None)
    assert xy[0, 0] == pytest.approx(104.5)


def _restart(type_id, **over):
    base = dict(
        game_id=[9, 9],
        period_id=[1, 1],
        action_id=[0, 1],
        team_id=[1, 1],
        player_id=[10, 11],
        time_seconds=[5.0, 6.0],
        type_id=[type_id, _PASS],
        start_x=[np.nan, 50.0],
        start_y=[np.nan, 30.0],
        end_x=[np.nan, 60.0],
        end_y=[np.nan, 30.0],
    )
    base.update(over)
    return pd.DataFrame(base)


class TestResolveRestartGeometryEventsOnly:
    def test_goalkick_origin_rule_point(self):
        g = resolve_restart_geometry(_restart(_GK), frames=None)
        assert g.loc[0, "enriched_start_x"] == pytest.approx(5.5)
        assert g.loc[0, "start_coord_source"] == "restart_prior"
        assert g.loc[0, "start_coord_confidence"] == pytest.approx(0.2)

    def test_penalty_origin_rule_point(self):
        g = resolve_restart_geometry(_restart(_PEN), frames=None)
        assert g.loc[0, "enriched_start_x"] == pytest.approx(94.0)
        assert g.loc[0, "enriched_start_y"] == pytest.approx(34.0)
        assert g.loc[0, "start_coord_confidence"] == pytest.approx(0.5)

    def test_corner_side_from_native_end_y(self):
        # native end_y=10 (<34) -> near corner (105, 0)
        g = resolve_restart_geometry(_restart(_CORNER_C, end_x=[95.0, 60.0], end_y=[10.0, 30.0]), frames=None)
        assert g.loc[0, "enriched_start_x"] == pytest.approx(105.0)
        assert g.loc[0, "enriched_start_y"] == pytest.approx(0.0)
        assert g.loc[0, "start_coord_source"] == "restart_prior"

    def test_corner_side_unresolvable_stays_unresolved(self):
        # no native end, no next-event y, no frames -> cannot determine side -> unresolved
        g = resolve_restart_geometry(
            _restart(
                _CORNER_C,
                end_x=[np.nan, np.nan],
                end_y=[np.nan, np.nan],
                start_x=[np.nan, np.nan],
                start_y=[np.nan, np.nan],
            ),
            frames=None,
        )
        assert g.loc[0, "start_coord_source"] == "unresolved"

    def test_openplay_pass_no_rule_point(self):
        # a pass with NaN origin gets NO rule-point (events-only -> unresolved)
        g = resolve_restart_geometry(_restart(_PASS), frames=None)
        assert g.loc[0, "start_coord_source"] == "unresolved"

    def test_dest_next_event_full_frame(self):
        g = resolve_restart_geometry(_restart(_GK), frames=None)
        # row0 NaN end -> next row (action 1) start (50,30)
        assert g.loc[0, "enriched_end_x"] == pytest.approx(50.0)
        assert g.loc[0, "end_coord_source"] == "next_event"

    def test_does_not_mutate_input(self):
        a = _restart(_GK)
        before_sx, before_ex = a["start_x"].copy(), a["end_x"].copy()
        resolve_restart_geometry(a, frames=None)
        pd.testing.assert_series_equal(a["start_x"], before_sx)
        pd.testing.assert_series_equal(a["end_x"], before_ex)

    def test_emits_new_column_contract(self):
        g = resolve_restart_geometry(_restart(_GK), frames=None)
        assert {
            "enriched_start_x",
            "enriched_start_y",
            "start_coord_source",
            "start_coord_confidence",
            "enriched_end_x",
            "enriched_end_y",
            "end_coord_source",
            "end_coord_confidence",
        } <= set(g.columns)


class TestResolveRestartGeometryFrames:
    def _frames(self, **over):
        # Row 0 is a PLAYER of the acting team (1), row 1 is the ball. The player row exists so
        # the acting team's direction RESOLVES: `acting_team_attacks_rtl` filters ball rows out,
        # so a ball-only frame can never resolve a direction, and since 4.80.0 that means <NA>
        # and no tracking tier at all. These tests are about TIER ORDERING, not orientation, so
        # the fixture has to clear the orientation bar for them to test what they claim to.
        base = dict(
            game_id=[9, 9],
            period_id=[1, 1],
            frame_id=[1250, 1250],
            time_seconds=[5.0, 5.0],
            team_id=[1, 0],
            player_id=[10, -1],
            is_goalkeeper=[False, False],
            is_ball=[False, True],
            x=[60.0, 104.6],
            y=[34.0, 0.4],
            source_provider=["gradientsports", "gradientsports"],
            team_attacking_direction=["ltr", None],
        )
        base.update(over)
        return pd.DataFrame(base)

    def test_corner_origin_tracking_ball_beats_prior(self):
        a = _restart(_CORNER_C)
        a.loc[0, "action_id"], a.loc[1, "action_id"] = 0, 1
        g = resolve_restart_geometry(a, frames=self._frames())
        assert g.loc[0, "start_coord_source"] == "tracking_ball"
        assert g.loc[0, "enriched_start_x"] == pytest.approx(104.6)
        assert g.loc[0, "start_coord_confidence"] == pytest.approx(0.8)

    def test_goalkick_never_uses_tracking_ball(self):
        # goalkick + ball tracked at midfield: must NOT pick the ball; in-area-GK absent -> rule-point
        # The x/y overrides carry BOTH rows (player, ball) -- the ball is the second element.
        a = _restart(_GK)
        g = resolve_restart_geometry(a, frames=self._frames(x=[60.0, 50.0], y=[34.0, 20.0]))
        assert g.loc[0, "start_coord_source"] == "restart_prior"  # never tracking_ball


from silly_kicks.tracking._gk_geometry import apply_restart_tripwire  # noqa: E402


class TestTripwire:
    def _enriched(self, source, x, y, type_id=_CORNER_C):
        # minimal enriched frame (as resolve_restart_geometry would emit) for one origin row
        return pd.DataFrame(
            {
                "type_id": [type_id],
                "enriched_start_x": [x],
                "enriched_start_y": [y],
                "start_coord_source": [source],
                "start_coord_confidence": [{"native": 1.0, "restart_prior": 0.4}.get(source, 0.8)],
                "enriched_end_x": [60.0],
                "enriched_end_y": [30.0],
                "end_coord_source": ["native"],
                "end_coord_confidence": [1.0],
            }
        )

    def test_imputed_out_of_region_reverts_to_tripwire_reverted(self):
        # an imputed corner at midfield (x=50) violates the corner region (x>=100) -> reverted
        df = self._enriched("tracking_ball", 50.0, 20.0)
        with pytest.warns(UserWarning):
            n = apply_restart_tripwire(df)
        assert n == 1
        assert df.loc[0, "start_coord_source"] == "tripwire_reverted"
        assert df.loc[0, "start_coord_confidence"] == pytest.approx(0.0)
        assert np.isnan(df.loc[0, "enriched_start_x"])  # type: ignore[arg-type]

    def test_native_out_of_region_warns_not_reverted(self):
        # native coord out of region is provider truth -> warn-only, keep native
        df = self._enriched("native", 80.0, 34.0, type_id=_GK)
        with pytest.warns(UserWarning):
            n = apply_restart_tripwire(df)
        assert n == 0
        assert df.loc[0, "start_coord_source"] == "native"
        assert df.loc[0, "enriched_start_x"] == pytest.approx(80.0)

    def test_in_region_imputed_untouched(self):
        df = self._enriched("restart_prior", 105.0, 0.0)
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error")  # no warning expected
            n = apply_restart_tripwire(df)
        assert n == 0
        assert df.loc[0, "start_coord_source"] == "restart_prior"


from silly_kicks.tracking._restart_report import RestartCoordinateReport  # noqa: E402


class TestReport:
    def test_from_frame_counts_match_value_counts(self):
        g = resolve_restart_geometry(
            pd.concat([_restart(_GK), _restart(_PEN)], ignore_index=True).assign(
                action_id=range(4), game_id=9, period_id=1
            ),
            frames=None,
        )
        rep = RestartCoordinateReport.from_frame(g)
        assert rep.n_rows == 4
        assert rep.start_source_counts == {str(k): int(v) for k, v in g["start_coord_source"].value_counts().items()}

    def test_n_tripwire_reversions_counts_tagged_rows(self):
        # frame with one tripwire_reverted row -> report surfaces it
        g = pd.DataFrame(
            {
                "start_coord_source": ["restart_prior", "tripwire_reverted", "native"],
                "end_coord_source": ["next_event", "native", "native"],
            }
        )
        rep = RestartCoordinateReport.from_frame(g)
        assert rep.n_tripwire_reversions == 1


class TestTrackingBallReprojection:
    """ADR-028 RC2: `_tracking_ball_xy` returns an ACTION-LTR quantity sampled from FRAME-LTR rows.

    These exist because the fix's ball half had zero coverage when it landed: every other fixture in
    this module omits ``team_attacking_direction``, so ``acting_team_attacks_rtl`` resolves nothing,
    ``flip`` is all-False, and the reprojection branch never executes. A test suite that cannot
    execute a branch cannot guard it.

    Unlike `_tracking_gk_xy` there is NO goal-area clamp here, so an unreprojected away ball is not
    quietly dropped to a prior -- it moves by up to a full pitch length (spec section 2.2 measured
    101.24 m on GS, 99.58 m on IDSSE).
    """

    @staticmethod
    def _frames(direction_by_team):
        rows = []
        for team, direction in direction_by_team.items():
            rows.append(
                dict(
                    game_id=9,
                    period_id=1,
                    frame_id=1250,
                    time_seconds=50.0,
                    team_id=team,
                    player_id=100 + team,
                    is_goalkeeper=False,
                    is_ball=False,
                    x=52.5,
                    y=34.0,
                    team_attacking_direction=direction,
                    source_provider="gradientsports",
                )
            )
        return rows

    @staticmethod
    def _action(team_id):
        return pd.DataFrame(
            dict(
                game_id=[9],
                period_id=[1],
                action_id=[7],
                team_id=[team_id],
                type_id=[_CORNER_C],
                time_seconds=[50.0],
                start_x=[np.nan],
                start_y=[np.nan],
                end_x=[np.nan],
                end_y=[np.nan],
            )
        )

    def _ball_xy(self, *, team_id, ball_xy):
        """Resolve one action against a frame whose ball sits at ``ball_xy`` (FRAME coords)."""
        frames = pd.DataFrame(
            [
                *self._frames({1: "ltr", 2: "rtl"}),
                dict(
                    game_id=9,
                    period_id=1,
                    frame_id=1250,
                    time_seconds=50.0,
                    team_id=np.nan,
                    player_id=-1,
                    is_goalkeeper=False,
                    is_ball=True,
                    x=ball_xy[0],
                    y=ball_xy[1],
                    team_attacking_direction=None,
                    source_provider="gradientsports",
                ),
            ]
        )
        out = _tracking_ball_xy(self._action(team_id), frames, None)
        return float(out[0, 0]), float(out[0, 1])

    def test_home_and_away_mirrored_scenes_agree_in_action_ltr(self):
        """The SAME physical situation must yield the SAME action-LTR ball position for both teams."""
        home = self._ball_xy(team_id=1, ball_xy=(30.0, 20.0))
        # The away team's physically-identical scene is the 180 degree point reflection.
        away = self._ball_xy(team_id=2, ball_xy=(105.0 - 30.0, 68.0 - 20.0))

        assert home == pytest.approx((30.0, 20.0))
        assert away == pytest.approx(home), (
            f"away action-LTR ball {away} must equal the home twin {home}; a raw frame sample would give (75.0, 48.0)"
        )

    def test_away_result_differs_from_the_raw_frame_value(self):
        """Non-vacuity: the reprojection must MEASURABLY move the away ball.

        Without this, the test above would still pass if `flip` were all-False and BOTH legs simply
        returned raw frame coords for a y-symmetric, x-centred scene.
        """
        raw = (75.0, 48.0)
        away = self._ball_xy(team_id=2, ball_xy=raw)
        assert away != pytest.approx(raw), "reprojection did not fire -- the branch is unexercised"
        assert away == pytest.approx((30.0, 20.0))

    def test_home_row_is_byte_identical_to_the_raw_frame_value(self):
        """Home actions must be untouched: RC2 changes away rows only."""
        assert self._ball_xy(team_id=1, ball_xy=(75.0, 48.0)) == pytest.approx((75.0, 48.0))


class TestNextEventCrossTeamReprojection:
    """ADR-051: `_next_event_start` borrows the NEXT action's coords as a destination proxy.

    SPADL is per-ACTING-team LTR, so a next action belonging to the OTHER team describes the same
    physical point in the opposite convention. Borrowing it verbatim placed the anchor's destination
    at the wrong end of the pitch. Fixed 4.71.0.

    This is an ACTION-vs-ACTION mismatch, so the frames-based mirror registry is structurally blind
    to it -- these tests are the only guard, which is why they assert the wrong value is NOT produced
    as well as that the right one is.
    """

    _PHYS = (60.0, 48.0)  # the shared point in the ANCHOR team's action-LTR frame
    _OPP = (105.0 - 60.0, 68.0 - 48.0)  # the same point as the OPPONENT records it

    @classmethod
    def _actions(cls, next_team, *, next_xy=None):
        return pd.DataFrame(
            dict(
                game_id=[9, 9],
                period_id=[1, 1],
                action_id=[1, 2],
                team_id=[1, next_team],
                start_x=[5.5, (next_xy or cls._OPP)[0]],
                start_y=[34.0, (next_xy or cls._OPP)[1]],
            )
        )

    def test_cross_team_next_event_is_reprojected(self):
        from silly_kicks.tracking._gk_geometry import _next_event_start

        nx, ny = _next_event_start(self._actions(2))
        assert (nx[0], ny[0]) == pytest.approx(self._PHYS), (
            "a cross-team next event must be point-reflected into the anchor's frame"
        )
        assert (nx[0], ny[0]) != pytest.approx(self._OPP), "borrowed the opponent's raw coords"

    def test_same_team_next_event_is_borrowed_verbatim(self):
        """Control: no reflection when the teams match -- otherwise the fix would break the common case."""
        from silly_kicks.tracking._gk_geometry import _next_event_start

        nx, ny = _next_event_start(self._actions(1, next_xy=self._PHYS))
        assert (nx[0], ny[0]) == pytest.approx(self._PHYS)

    @pytest.mark.parametrize("teams", [(1, None), (None, 2), (None, None)])
    def test_unattested_team_id_never_decides(self, teams):
        """ADR-027: an NA team id must NOT trigger a reflection -- "cannot tell" is not "reflect"."""
        from silly_kicks.tracking._gk_geometry import _next_event_start

        a = self._actions(2)
        a["team_id"] = pd.array(list(teams), dtype="Int64")
        nx, ny = _next_event_start(a)
        assert (nx[0], ny[0]) == pytest.approx(self._OPP), (
            "an unattested team id must leave the borrowed coordinate untouched"
        )

    def test_period_boundary_still_nans_the_borrow(self):
        """The pre-existing boundary guard must survive the reflection change."""
        from silly_kicks.tracking._gk_geometry import _next_event_start

        a = self._actions(2)
        a.loc[1, "period_id"] = 2
        nx, ny = _next_event_start(a)
        assert np.isnan(nx[0]) and np.isnan(ny[0])
