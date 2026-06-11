"""Tests for resolve_gk_geometry (xT-GK goal-kick coordinate derivation).

Per docs/superpowers/plans/2026-06-08-xt-gk-goalkick-coverage-implementation.md (Task A1).
"""

import pathlib

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._gk_geometry import resolve_gk_geometry

_GK = 22  # goalkick type_id


def _actions(**over):
    base = dict(
        game_id=[9, 9],
        action_id=[0, 1],
        team_id=[1, 1],
        player_id=[10, 10],
        period_id=[1, 1],
        time_seconds=[5.0, 50.0],
        type_id=[_GK, _GK],
        start_x=[5.0, np.nan],
        start_y=[34.0, np.nan],
        end_x=[55.0, 60.0],
        end_y=[34.0, 30.0],
    )
    base.update(over)
    return pd.DataFrame(base)


class TestResolveGkGeometry:
    def test_native_origin_kept(self):
        g = resolve_gk_geometry(_actions(), frames=None)
        assert g.loc[0, "origin_x"] == 5.0
        assert g.loc[0, "origin_source"] == "native"
        assert g.loc[0, "origin_confidence"] == pytest.approx(1.0)

    def test_nan_origin_falls_to_rule_point_when_no_frames(self):
        g = resolve_gk_geometry(_actions(), frames=None)
        assert g.loc[1, "origin_x"] == pytest.approx(5.5)
        assert g.loc[1, "origin_y"] == pytest.approx(34.0)
        assert g.loc[1, "origin_source"] == "goalkick_prior"
        assert g.loc[1, "origin_confidence"] < 0.7  # type: ignore[operator]

    def test_tracking_gk_used_only_when_in_goal_area(self):
        frames = pd.DataFrame(
            {
                "game_id": [9],
                "period_id": [1],
                "frame_id": [1250],
                "time_seconds": [50.0],
                "team_id": [1],
                "player_id": [10],
                "is_goalkeeper": [True],
                "is_ball": [False],
                "x": [4.0],
                "y": [33.0],
                "source_provider": ["sportec"],
            }
        )
        g = resolve_gk_geometry(_actions(), frames=frames)
        assert g.loc[1, "origin_source"] == "tracking_gk"
        assert g.loc[1, "origin_x"] == pytest.approx(4.0)
        assert 0.6 <= g.loc[1, "origin_confidence"] < 1.0  # type: ignore[operator]

    def test_tracking_gk_offposition_clamped_to_prior(self):
        frames = pd.DataFrame(
            {
                "game_id": [9],
                "period_id": [1],
                "frame_id": [1250],
                "time_seconds": [50.0],
                "team_id": [1],
                "player_id": [10],
                "is_goalkeeper": [True],
                "is_ball": [False],
                "x": [40.0],
                "y": [33.0],
                "source_provider": ["sportec"],
            }
        )
        g = resolve_gk_geometry(_actions(), frames=frames)
        assert g.loc[1, "origin_source"] == "goalkick_prior"
        assert g.loc[1, "origin_x"] == pytest.approx(5.5)

    def test_dest_native_kept_and_nan_dest_unresolved(self):
        # row 1 has NaN end and is the LAST row -> no next-event -> unresolved
        g = resolve_gk_geometry(_actions(end_x=[55.0, np.nan]), frames=None)
        assert g.loc[0, "dest_source"] == "native"
        assert np.isnan(g.loc[1, "dest_x"])  # type: ignore[arg-type]
        assert g.loc[1, "dest_source"] == "unresolved"

    def test_nan_dest_uses_next_event_within_period(self):
        a = _actions()
        a.loc[0, "end_x"] = np.nan
        a.loc[0, "end_y"] = np.nan
        # next row (action 1) start is NaN here; give it a value so next-event fires
        a.loc[1, "start_x"] = 40.0
        a.loc[1, "start_y"] = 30.0
        g = resolve_gk_geometry(a, frames=None)
        assert g.loc[0, "dest_x"] == pytest.approx(40.0)
        assert g.loc[0, "dest_source"] == "next_event"

    def test_next_event_not_across_period_boundary(self):
        a = _actions()
        a.loc[0, "end_x"] = np.nan
        a.loc[0, "end_y"] = np.nan
        a.loc[1, "period_id"] = 2  # next row is a different period
        a.loc[1, "start_x"] = 40.0
        a.loc[1, "start_y"] = 30.0
        g = resolve_gk_geometry(a, frames=None)
        assert np.isnan(g.loc[0, "dest_x"])  # type: ignore[arg-type]  # boundary guard -> not used
        assert g.loc[0, "dest_source"] == "unresolved"

    def test_does_not_mutate_input(self):
        a = _actions()
        before = a["start_x"].copy()
        resolve_gk_geometry(a, frames=None)
        pd.testing.assert_series_equal(a["start_x"], before)


_THROW = 2  # throw_in type_id


class TestResolveGkGeometryFrozenContract:
    """Pins the pre-promotion contract so the Task-5 delegation shim stays byte-identical."""

    def test_exact_output_columns_no_dest_confidence(self):
        g = resolve_gk_geometry(_actions(), frames=None)
        assert set(g.columns) == {
            "origin_x",
            "origin_y",
            "origin_source",
            "origin_confidence",
            "dest_x",
            "dest_y",
            "dest_source",
        }  # note: NO dest_confidence column in the frozen contract

    def test_nongoalkick_throwin_not_imputed(self):
        # A throw_in with NaN origin must stay native-or-unresolved (goalkick-only imputation).
        a = _actions(type_id=[_THROW, _GK], start_x=[np.nan, 5.0], start_y=[np.nan, 34.0])
        g = resolve_gk_geometry(a, frames=None)
        assert g.loc[0, "origin_source"] == "unresolved"
        assert np.isnan(g.loc[0, "origin_x"])  # type: ignore[arg-type]

    def test_offposition_gk_goalkick_falls_to_rule_point(self):
        # (Major-2a) off-position GK must NOT be used; falls to goalkick_prior.
        frames = pd.DataFrame(
            {
                "game_id": [9],
                "period_id": [1],
                "frame_id": [1250],
                "time_seconds": [50.0],
                "team_id": [1],
                "player_id": [10],
                "is_goalkeeper": [True],
                "is_ball": [False],
                "x": [40.0],
                "y": [33.0],
                "source_provider": ["sportec"],
            }
        )
        g = resolve_gk_geometry(_actions(), frames=frames)
        assert g.loc[1, "origin_source"] == "goalkick_prior"
        assert g.loc[1, "origin_x"] == pytest.approx(5.5)

    def test_goalkick_no_native_end_no_next_event_unresolved(self):
        # (Major-2b) NaN end + last row -> dest unresolved (must STAY unresolved post-refactor).
        g = resolve_gk_geometry(_actions(end_x=[55.0, np.nan], end_y=[34.0, np.nan]), frames=None)
        assert g.loc[1, "dest_source"] == "unresolved"
        assert np.isnan(g.loc[1, "dest_x"])  # type: ignore[arg-type]


_FIX = pathlib.Path(__file__).parent / "_fixtures"


class TestGoldenSnapshot:
    """Byte-identical guard for the Task-5 refactor: full-frame snapshot pins column set + order +
    dtypes + every cell on a multi-type fixture (goalkick / corner / throw-in / open-play pass)."""

    def _multi(self):
        return pd.DataFrame(
            dict(
                game_id=[9, 9, 9, 9],
                period_id=[1, 1, 1, 1],
                action_id=[0, 1, 2, 3],
                team_id=[1, 1, 1, 1],
                player_id=[10, 11, 12, 10],
                type_id=[22, 5, 2, 0],
                time_seconds=[5.0, 6.0, 7.0, 70.0],
                start_x=[np.nan, np.nan, np.nan, 50.0],
                start_y=[np.nan, np.nan, np.nan, 30.0],
                end_x=[60.0, 95.0, 40.0, np.nan],
                end_y=[30.0, 10.0, 20.0, np.nan],
            )
        )

    def _frames(self):
        return pd.DataFrame(
            dict(
                game_id=[9],
                period_id=[1],
                frame_id=[1250],
                time_seconds=[5.0],
                team_id=[1],
                player_id=[10],
                is_goalkeeper=[True],
                is_ball=[False],
                x=[4.0],
                y=[33.0],
                source_provider=["sportec"],
            )
        )

    def test_golden_noframes(self):
        got = resolve_gk_geometry(self._multi(), frames=None)
        pd.testing.assert_frame_equal(got, pd.read_parquet(_FIX / "gk_geometry_golden_noframes.parquet"))

    def test_golden_frames(self):
        got = resolve_gk_geometry(self._multi(), frames=self._frames())
        pd.testing.assert_frame_equal(got, pd.read_parquet(_FIX / "gk_geometry_golden_frames.parquet"))


class TestShimNoTripwireLeak:
    def test_resolve_gk_geometry_emits_no_warning_on_out_of_region_native(self):
        # The engine is pure; the shim never tripwires -> a native-out-of-region goalkick is SILENT
        # through resolve_gk_geometry (the frozen contract emitted no warnings).
        import warnings as _w

        a = _actions(start_x=[80.0, 5.0], start_y=[34.0, 34.0])  # row0 goalkick native x=80 (out of area)
        with _w.catch_warnings():
            _w.simplefilter("error")  # any warning -> test failure
            g = resolve_gk_geometry(a, frames=None)
        assert g.loc[0, "origin_source"] == "native"
        assert g.loc[0, "origin_x"] == pytest.approx(80.0)
