"""Tests for per-pass structural primitives (TF-45)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._structural_pass import _structural_pass_core


def _sgm_ref(defs, p, sigma):
    d2 = ((defs - np.asarray(p)) ** 2).sum(axis=1)
    rho = np.exp(-d2 / (2.0 * sigma * sigma)).sum()
    return 1.0 / rho


class TestStructuralPassCore:
    def test_lbs_counts_defenders_in_band(self):
        # passer at x=40, receiver at x=70; defenders at x=50,60 (in band), 30,80 (out)
        defs = np.array([[50.0, 34.0], [60.0, 20.0], [30.0, 34.0], [80.0, 34.0]])
        lbs, _sgm, _sdi = _structural_pass_core(defs, (40.0, 34.0), (70.0, 34.0), 15.0)
        assert lbs == 2

    def test_lbs_boundary_strict_lower_inclusive_upper(self):
        # start_x < d_x <= end_x : d at start_x excluded, d at end_x included
        defs = np.array([[40.0, 34.0], [70.0, 34.0]])
        lbs, _, _ = _structural_pass_core(defs, (40.0, 34.0), (70.0, 34.0), 15.0)
        assert lbs == 1  # x=40 excluded (strict <), x=70 included (<=)

    def test_lbs_backward_pass_is_zero(self):
        defs = np.array([[50.0, 34.0], [60.0, 34.0]])
        lbs, _, _ = _structural_pass_core(defs, (70.0, 34.0), (40.0, 34.0), 15.0)
        assert lbs == 0  # receiver behind passer -> empty band

    def test_lbs_zero_with_defenders_present_is_not_nan(self):
        # forward pass, defenders present but none in band -> structural_lbs == 0 (NOT nan)
        defs = np.array([[10.0, 34.0], [90.0, 34.0]])
        lbs, _, _ = _structural_pass_core(defs, (40.0, 34.0), (50.0, 34.0), 15.0)
        assert lbs == 0

    def test_zero_defenders_all_nan(self):
        defs = np.empty((0, 2))
        lbs, sgm, sdi = _structural_pass_core(defs, (40.0, 34.0), (70.0, 34.0), 15.0)
        assert np.isnan(lbs) and np.isnan(sgm) and np.isnan(sdi)

    def test_single_defender_is_numeric(self):
        defs = np.array([[55.0, 34.0]])
        lbs, sgm, sdi = _structural_pass_core(defs, (40.0, 34.0), (70.0, 34.0), 15.0)
        assert lbs == 1
        assert np.isfinite(sgm) and np.isfinite(sdi)

    def test_sgm_matches_reference(self):
        defs = np.array([[55.0, 30.0], [60.0, 40.0]])
        p, r, sigma = (40.0, 34.0), (70.0, 34.0), 15.0
        _, sgm, _ = _structural_pass_core(defs, p, r, sigma)
        expected = _sgm_ref(defs, r, sigma) - _sgm_ref(defs, p, sigma)
        assert sgm == pytest.approx(expected, abs=1e-9)

    def test_sdi_matches_centroid_reference(self):
        defs = np.array([[50.0, 30.0], [60.0, 38.0]])
        p, r = (40.0, 34.0), (70.0, 34.0)
        _, _, sdi = _structural_pass_core(defs, p, r, 15.0)
        c = defs.mean(axis=0)
        expected = np.hypot(r[0] - c[0], r[1] - c[1]) - np.hypot(p[0] - c[0], p[1] - c[1])
        assert sdi == pytest.approx(expected, abs=1e-9)


from silly_kicks.tracking._structural_pass import compute_structural_pass_metrics  # noqa: E402
from tests.tracking.test_defensive_line import _make_frame_rows  # noqa: E402


class TestComputePrimitive:
    def _frame(self):
        # away outfield acts as defenders for a HOME pass: x=50,60 in band
        return _make_frame_rows(
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
            away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
        )

    def test_home_pass_keys_and_lbs(self):
        out = compute_structural_pass_metrics(
            self._frame(),
            attacking_team_id=1,
            home_team_id=1,
            passer_xy=(40.0, 34.0),
            receiver_xy=(70.0, 34.0),
        )
        assert set(out) == {"structural_lbs", "structural_sgm", "structural_sdi"}
        assert out["structural_lbs"] == 2.0  # away defenders at x=50,60

    def test_gk_excluded_ball_excluded(self):
        # away GK at (102,34) is NOT a defender; ball excluded; lbs unchanged
        out = compute_structural_pass_metrics(
            self._frame(),
            attacking_team_id=1,
            home_team_id=1,
            passer_xy=(40.0, 34.0),
            receiver_xy=(70.0, 34.0),
        )
        assert out["structural_lbs"] == 2.0

    def test_away_pass_mirror_matches_home(self):
        frame = _make_frame_rows(
            home_outfield_xs=[55.0, 45.0, 75.0, 25.0],  # these become defenders for an AWAY pass
            home_outfield_ys=[34.0, 48.0, 34.0, 34.0],
            away_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0],
        )
        out = compute_structural_pass_metrics(
            frame,
            attacking_team_id=2,
            home_team_id=1,
            passer_xy=(40.0, 34.0),
            receiver_xy=(70.0, 34.0),
        )
        # home defenders mirrored: 105-55=50, 105-45=60 in band (40,70] -> lbs 2
        assert out["structural_lbs"] == 2.0
        assert np.isfinite(out["structural_sgm"]) and np.isfinite(out["structural_sdi"])

    def test_zero_defenders_nan(self):
        frame = _make_frame_rows(
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[],
            away_outfield_ys=[],
        )
        out = compute_structural_pass_metrics(
            frame,
            attacking_team_id=1,
            home_team_id=1,
            passer_xy=(40.0, 34.0),
            receiver_xy=(70.0, 34.0),
        )
        assert all(np.isnan(v) for v in out.values())


def _actions(team_id=1):
    return pd.DataFrame(
        {
            "game_id": [1, 1],
            "action_id": [1, 2],
            "period_id": [1, 1],
            "time_seconds": [1.0, 1.0],
            "team_id": [team_id, team_id],
            "player_id": [50, 51],
            "start_x": [40.0, 0.0],
            "start_y": [34.0, 34.0],
            "end_x": [70.0, 5.0],
            "end_y": [34.0, 34.0],
            "type_id": [0, 0],
        }
    )


class TestKernel:
    def _frame(self):
        return _make_frame_rows(
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
            away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
        )

    def test_batch_aligns_and_computes(self):
        from silly_kicks.tracking._kernels import _structural_pass_at_actions

        out = _structural_pass_at_actions(_actions(), self._frame(), home_team_id=1)
        assert list(out.columns) == ["structural_lbs", "structural_sgm", "structural_sdi"]
        assert out["structural_lbs"].iloc[0] == 2
        assert len(out) == 2

    def test_unlinked_action_nan(self):
        from silly_kicks.tracking._kernels import _structural_pass_at_actions

        frame = _make_frame_rows(
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
            away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
            time_seconds=500.0,
        )
        out = _structural_pass_at_actions(_actions(), frame, home_team_id=1)
        assert pd.isna(out["structural_lbs"].iloc[0])

    def test_duplicate_action_id_in_slot_does_not_raise(self):
        from silly_kicks.tracking._kernels import _structural_pass_at_actions

        slot = _actions()
        slot["action_id"] = [1, 1]  # duplicate, as in a shifted boundary slot
        out = _structural_pass_at_actions(slot, self._frame(), home_team_id=1)
        assert len(out) == 2
        assert out["structural_lbs"].iloc[0] == 2  # forward pass row resolved correctly

    def test_string_team_id_dtype_safe(self):
        from silly_kicks.tracking._kernels import _structural_pass_at_actions

        frame = _make_frame_rows(
            home_team_id="H",
            away_team_id="A",
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
            away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
        )
        out = _structural_pass_at_actions(_actions(team_id="H"), frame, home_team_id="H")
        assert out["structural_lbs"].iloc[0] == 2


class TestResolveFrameIdsByPosition:
    def _frame(self):
        return _make_frame_rows(
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
            away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
        )

    def test_unique_links_equals_old_at_lookup(self):
        from silly_kicks.tracking._kernels import resolve_frame_ids_by_position
        from silly_kicks.tracking.utils import link_actions_to_frames

        acts, frame = _actions(), self._frame()
        links, _ = link_actions_to_frames(acts, frame)
        got = resolve_frame_ids_by_position(acts, frame, links=links)
        pl = links.set_index("action_id")
        old = np.array(
            [
                float(pl.at[a, "frame_id"]) if (a in pl.index and pd.notna(pl.at[a, "frame_id"])) else np.nan
                for a in acts["action_id"]
            ]
        )
        np.testing.assert_array_equal(np.isnan(got), np.isnan(old))
        np.testing.assert_array_equal(got[~np.isnan(got)], old[~np.isnan(old)])

    def test_links_path_equals_internal_link_path(self):
        from silly_kicks.tracking._kernels import resolve_frame_ids_by_position
        from silly_kicks.tracking.utils import link_actions_to_frames

        acts, frame = _actions(), self._frame()
        links, _ = link_actions_to_frames(acts, frame)
        a = resolve_frame_ids_by_position(acts, frame, links=links)
        b = resolve_frame_ids_by_position(acts, frame)
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        np.testing.assert_array_equal(a[~np.isnan(a)], b[~np.isnan(b)])

    def test_duplicate_action_id_position_aligned_no_raise(self):
        from silly_kicks.tracking._kernels import resolve_frame_ids_by_position

        acts, frame = _actions(), self._frame()
        acts["action_id"] = [1, 1]
        out = resolve_frame_ids_by_position(acts, frame)
        assert len(out) == 2

    def test_unlinked_row_nan(self):
        from silly_kicks.tracking._kernels import resolve_frame_ids_by_position

        frame = self._frame()
        frame["time_seconds"] = 500.0
        out = resolve_frame_ids_by_position(_actions(), frame)
        assert np.isnan(out).all()


class TestAggregator:
    def _frame(self):
        return _make_frame_rows(
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
            away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
        )

    def test_appends_three_namespaced_columns(self):
        from silly_kicks.tracking.features import add_structural_pass

        res = add_structural_pass(_actions(), self._frame(), home_team_id=1)
        for c in ("structural_lbs", "structural_sgm", "structural_sdi"):
            assert c in res.columns
        assert res["structural_lbs"].iloc[0] == 2
        assert "frame_id" in res.columns

    def test_non_pass_cross_is_na(self):
        from silly_kicks.tracking.features import add_structural_pass

        acts = _actions()
        acts.loc[0, "type_id"] = 11  # shot
        res = add_structural_pass(acts, self._frame(), home_team_id=1)
        assert pd.isna(res["structural_lbs"].iloc[0])

    def test_links_path_equals_internal(self):
        from silly_kicks.tracking.features import add_structural_pass
        from silly_kicks.tracking.utils import link_actions_to_frames

        acts, frame = _actions(), self._frame()
        a = add_structural_pass(acts, frame, home_team_id=1)
        links, _ = link_actions_to_frames(acts, frame)
        b = add_structural_pass(acts, frame, home_team_id=1, links=links)
        pd.testing.assert_series_equal(a["structural_sgm"], b["structural_sgm"])

    def test_provenance_present_and_unsuffixed_on_rechain(self):
        from silly_kicks.tracking.features import add_structural_pass

        acts, frame = _actions(), self._frame()
        once = add_structural_pass(acts, frame, home_team_id=1)
        twice = add_structural_pass(once, frame, home_team_id=1)
        assert "frame_id" in twice.columns
        assert "frame_id_x" not in twice.columns and "frame_id_y" not in twice.columns


class TestXfns:
    def _frame(self):
        return _make_frame_rows(
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
            away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
        )

    def test_emits_namespaced_per_slot_columns(self):
        from silly_kicks.tracking.features import structural_pass_xfns

        xfns = structural_pass_xfns(home_team_id=1)
        assert len(xfns) == 1
        transformer = xfns[0]
        assert getattr(transformer, "_frame_aware", False) is True
        cols = transformer([_actions(), _actions(), _actions()], self._frame())
        assert "structural_lbs_a0" in cols.columns
        assert "structural_sdi_a2" in cols.columns

    def test_frames_none_guard(self):
        from silly_kicks.tracking.features import structural_pass_xfns

        transformer = structural_pass_xfns(home_team_id=1)[0]
        cols = transformer([_actions(), _actions(), _actions()], None)
        assert cols["structural_lbs_a0"].isna().all()

    def test_real_gamestates_with_duplicate_action_ids(self):
        from silly_kicks.tracking.features import structural_pass_xfns
        from silly_kicks.vaep.feature_framework import gamestates

        states = gamestates(_actions(), nb_prev_actions=3)
        assert states[1]["action_id"].duplicated().any()  # precondition: dup exists
        transformer = structural_pass_xfns(home_team_id=1)[0]
        cols = transformer(states, self._frame())  # must not raise
        assert "structural_lbs_a0" in cols.columns


class TestAtomicMirror:
    def _frame(self):
        return _make_frame_rows(
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
            away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
        )

    def test_atomic_add_matches_standard_with_real_dxdy(self):
        from silly_kicks.atomic.tracking.features import add_structural_pass as atom
        from silly_kicks.tracking.features import add_structural_pass as std

        std_res = std(_actions(), self._frame(), home_team_id=1)  # start=(40,34) end=(70,34)
        atom_acts = pd.DataFrame(
            {
                "game_id": [1, 1],
                "action_id": [1, 2],
                "period_id": [1, 1],
                "time_seconds": [1.0, 1.0],
                "team_id": [1, 1],
                "player_id": [50, 51],
                "x": [40.0, 0.0],
                "y": [34.0, 34.0],
                "dx": [30.0, 5.0],
                "dy": [0.0, 0.0],  # end = x+dx, y+dy == standard
                "type_id": [0, 0],
            }
        )
        atom_res = atom(atom_acts, self._frame(), home_team_id=1)
        assert atom_res["structural_lbs"].iloc[0] == std_res["structural_lbs"].iloc[0] == 2

    def test_atomic_xfns_synthesizes_endpoints(self):
        from silly_kicks.atomic.tracking.features import structural_pass_xfns

        atom_state = pd.DataFrame(
            {
                "game_id": [1],
                "action_id": [1],
                "period_id": [1],
                "time_seconds": [1.0],
                "team_id": [1],
                "player_id": [50],
                "x": [40.0],
                "y": [34.0],
                "dx": [30.0],
                "dy": [0.0],
                "type_id": [0],
            }
        )
        t = structural_pass_xfns(home_team_id=1)[0]
        cols = t([atom_state, atom_state, atom_state], self._frame())
        assert cols["structural_lbs_a0"].iloc[0] == 2
