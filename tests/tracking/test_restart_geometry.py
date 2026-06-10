"""Tests for resolve_restart_geometry (general restart-coordinate enrichment).

Spec: docs/superpowers/specs/2026-06-10-general-restart-coordinate-enrichment-design.md
"""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._gk_geometry import _tracking_ball_xy, resolve_restart_geometry

_GK, _CORNER_C, _THROW, _PEN, _PASS = 22, 5, 2, 12, 0


def _frame(**over):
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
        base = dict(
            game_id=[9],
            period_id=[1],
            frame_id=[1250],
            time_seconds=[5.0],
            team_id=[0],
            player_id=[-1],
            is_goalkeeper=[False],
            is_ball=[True],
            x=[104.6],
            y=[0.4],
            source_provider=["gradientsports"],
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
        a = _restart(_GK)
        g = resolve_restart_geometry(a, frames=self._frames(x=[50.0], y=[20.0]))
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
        assert np.isnan(df.loc[0, "enriched_start_x"])

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
