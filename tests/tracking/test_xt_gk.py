"""Tests for xT-GK (Eyestone) — expected threat for goalkeepers.

Built task-by-task per docs/superpowers/plans/2026-06-08-xt-gk-implementation.md.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking._xt_gk import (
    XtGkParams,
    XtGkReport,
    _base,
    _composite,
    _convolve_grid,
    _counter_value,
    _dzv,
    _gk_distribution_mask,
    _grid_value,
    _normalize_pressure,
    _pev,
    _possession_depth,
    _progress,
    _rav,
    _temporal,
    compute_xt_gk,
)
from silly_kicks.tracking.features import add_xt_gk, xt_gk_xfns
from silly_kicks.xthreat import ExpectedThreat

_PHILOSOPHIES = ["possession", "counter", "direct", "high_press", "low_block"]
_XT_GK_COLS = ["xt_gk_base", "xt_gk_pev", "xt_gk_rav", "xt_gk_dzv", "xt_gk_pressure", "xt_gk"]
_PROVENANCE_COLS = [
    "xt_gk_origin_source",
    "xt_gk_dest_source",
    "xt_gk_origin_confidence",
    "xt_gk_completion_variant",
    "xt_gk_completion_source",
]


def _fitted_xt():
    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))  # value rises toward goal (+x)
    return xt


def _gk_realistic_xt():
    """GK-zone-REALISTIC grid: own third xT ~0.001-0.01 rising toward goal -- keeps the revalued
    surface V_GK = xT*phi small in the defensive third (the keeper-zone scale Eyestone's DZV/PEV
    fidelity terms assume), so per-action DZV lands O(0.01) not O(unity). The flat ramp (_fitted_xt)
    puts the defensive third at ~0.2, two orders of magnitude above real GK-zone xT (~0.001-0.005),
    which inflates V_GK and the DZV scale. Do NOT simplify this back to the ramp. (back-pass origin
    x=25 -> xi=3 -> (3/15)**3 = 0.008.)"""
    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16) ** 3, (12, 1))
    return xt


def _deep_flat_xt():
    """Raw xT FLAT (~0.005) across the defensive third, rising only past it. On this grid a
    short deep build-out has ~zero RAW forward gain -- the keeper-zone flatline Eyestone fixes.
    Revaluation (V_GK = xT*phi) restores a positive gain. Shape (12, 16)."""
    xt = ExpectedThreat(l=16, w=12)
    ramp = np.concatenate([np.full(6, 0.005), np.linspace(0.005, 1.0, 10)])  # cols 0-5 flat, then rise
    xt.xT = np.tile(ramp, (12, 1))
    return xt


def _gk_actions():
    # Two GK distributions by the GK (player 10, team 1): a forward goalkick + a back-pass.
    return pd.DataFrame(
        {
            "game_id": [9, 9],
            "action_id": [0, 1],
            "team_id": [1, 1],
            "player_id": [10, 10],
            "period_id": [1, 1],
            "time_seconds": [5.0, 50.0],
            "type_id": [22, 0],
            "start_x": [5.0, 25.0],
            "start_y": [34.0, 34.0],
            "end_x": [55.0, 10.0],
            "end_y": [34.0, 34.0],
        }
    )


def _frames_for(actions):
    """DAS-valid frames: get_xc -> _prepare_frames -> _validate_das_inputs hard-raises on
    missing vx/vy/team_in_possession, so they must be present with non-degenerate
    velocities and a set possession."""
    rows = []
    for fid, (t, period) in enumerate([(5.0, 1), (50.0, 1)]):
        for pid, team, gk, x, y, vx, vy in [
            (10, 1, True, 5.0, 34.0, 0.5, 0.0),
            (11, 1, False, 30.0, 30.0, 1.0, 0.2),
            (12, 1, False, 45.0, 40.0, 1.2, -0.1),
            (20, 2, True, 100.0, 34.0, -0.3, 0.0),
            (21, 2, False, 40.0, 40.0, -1.0, 0.1),
            (22, 2, False, 55.0, 28.0, -0.8, 0.0),
            (-1, -1, False, 6.0, 34.0, 0.5, 0.0),  # ball
        ]:
            rows.append(
                dict(
                    game_id=9,
                    period_id=period,
                    frame_id=fid,
                    time_seconds=t,
                    frame_rate=25.0,
                    team_id=team,
                    player_id=pid,
                    is_goalkeeper=gk,
                    is_ball=(pid == -1),
                    x=x,
                    y=y,
                    vx=vx,
                    vy=vy,
                    team_in_possession=1,
                    source_provider="sportec",
                    team_attacking_direction="ltr",
                )
            )
    return pd.DataFrame(rows)


class TestXtGkParams:
    def test_default_is_frozen_and_in_range(self):
        p = XtGkParams()
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.gamma = 0.5  # type: ignore[misc]  # frozen
        assert 0.1 <= p.gamma <= 0.4
        assert 0.3 <= p.delta <= 0.8
        assert 0.8 <= p.eta <= 0.9
        assert p.phi > 0.0
        assert p.dzv_alpha == pytest.approx(2.1)  # canonical (Eyestone 2026-06-27)
        assert p.dzv_beta == pytest.approx(0.8)  # canonical (Eyestone 2026-06-27)
        assert p.dzv_d_max > 0.0
        assert p.defensive_third_boundary > 0.0  # = D_threshold
        assert p.pressure_scale > 0.0
        assert p.convolution_sigma >= 0.0
        assert p.pressure_method == "andrienko_oval"

    @pytest.mark.parametrize("name", _PHILOSOPHIES)
    def test_for_philosophy_in_range(self, name):
        p = XtGkParams.for_philosophy(name)
        assert 0.1 <= p.gamma <= 0.4
        assert 0.3 <= p.delta <= 0.8
        assert 0.8 <= p.eta <= 0.9

    def test_for_philosophy_are_distinct(self):
        sigs = {(p.gamma, p.delta, p.phi, p.eta) for p in (XtGkParams.for_philosophy(n) for n in _PHILOSOPHIES)}
        assert len(sigs) == len(_PHILOSOPHIES)  # all five distinct

    def test_for_philosophy_rejects_unknown(self):
        with pytest.raises(ValueError, match="unknown"):
            XtGkParams.for_philosophy("tiki_taka")


class TestPureHelpers:
    def _ramp_grid(self):
        # shape (w, l) = (12, 16); value increases along x (columns) like xT toward goal
        return np.tile(np.linspace(0.0, 1.0, 16), (12, 1))

    def test_convolve_sigma_zero_is_identity(self):
        g = self._ramp_grid()
        out = _convolve_grid(g, 0.0)
        np.testing.assert_array_equal(out, g)

    def test_convolve_preserves_shape_and_smooths(self):
        g = self._ramp_grid()
        out = _convolve_grid(g, 1.0)
        assert out.shape == g.shape
        assert not np.array_equal(out, g)

    def test_grid_value_matches_inverted_row_convention(self):
        g = self._ramp_grid()
        v_far = _grid_value(g, np.array([100.0]), np.array([34.0]))[0]
        v_near = _grid_value(g, np.array([5.0]), np.array([34.0]))[0]
        assert v_far > v_near
        # x=5 -> xi=int(5/105*16)=0 ; y=34 -> yj=int(34/68*12)=6 ; row=(12-1)-6=5
        assert v_near == pytest.approx(g[5, 0])

    def test_grid_value_nan_coords_return_nan(self):
        # real provider data carries NaN coords (e.g. GS goal-kicks with missing destination);
        # _get_cell_indexes' .astype(int) would raise -> _grid_value must guard.
        g = self._ramp_grid()
        out = _grid_value(g, np.array([np.nan, 10.0]), np.array([34.0, np.nan]))
        assert np.isnan(out[0])
        assert np.isnan(out[1])
        # a finite coord alongside NaNs still resolves
        mixed = _grid_value(g, np.array([np.nan, 100.0]), np.array([34.0, 34.0]))
        assert np.isnan(mixed[0])
        assert not np.isnan(mixed[1])

    def test_counter_value_is_point_reflection(self):
        g = self._ramp_grid()
        x, y = np.array([10.0]), np.array([20.0])
        L, W = spadlconfig.field_length, spadlconfig.field_width
        expected = _grid_value(g, np.array([L - 10.0]), np.array([W - 20.0]))[0]
        assert _counter_value(g, x, y)[0] == pytest.approx(expected)

    def test_normalize_pressure_exp_cdf(self):
        assert _normalize_pressure(np.array([0.0]), 50.0)[0] == pytest.approx(0.0)
        assert _normalize_pressure(np.array([-5.0]), 50.0)[0] == pytest.approx(0.0)
        mid = _normalize_pressure(np.array([50.0]), 50.0)[0]
        assert mid == pytest.approx(1 - np.exp(-1.0))
        assert 0.0 <= mid < 1.0
        # realistic large pressure stays strictly below 1; extreme input saturates to 1.0
        # (float underflow of exp(-large)) -- bounded, which is all PEV needs.
        assert _normalize_pressure(np.array([300.0]), 50.0)[0] < 1.0
        assert _normalize_pressure(np.array([1e6]), 50.0)[0] <= 1.0

    def test_possession_depth_counts_within_team_run(self):
        actions = pd.DataFrame({"team_id": [1, 1, 1, 2, 2, 1], "period_id": [1, 1, 1, 1, 1, 1]})
        k = _possession_depth(actions)
        np.testing.assert_array_equal(k, [0, 1, 2, 0, 1, 0])

    def test_possession_depth_resets_on_period(self):
        actions = pd.DataFrame({"team_id": [1, 1, 1], "period_id": [1, 1, 2]})
        np.testing.assert_array_equal(_possession_depth(actions), [0, 1, 0])

    def test_phi_of_d_canonical_values(self):
        from silly_kicks.tracking._xt_gk import _phi_of_d

        phi = _phi_of_d(np.array([0.0, 5.0, 34.0, 35.0, 60.0]), alpha=2.1, beta=0.8, d_max=105.0, d_threshold=35.0)
        assert phi[0] == pytest.approx(2.1)  # d=0 -> alpha
        assert phi[1] == pytest.approx(2.1 * (1 - 5 / 105) ** -0.8)
        assert phi[2] == pytest.approx(2.1 * (1 - 34 / 105) ** -0.8)
        assert phi[2] > 2.8  # ~2.9 just below the threshold
        assert phi[3] == pytest.approx(1.0)  # at threshold -> cliffs to 1
        assert phi[4] == pytest.approx(1.0)  # outside def third -> 1
        assert (phi >= 1.0).all()

    def test_phi_grid_is_row_constant_and_matches_phi_of_d(self):
        from silly_kicks.tracking._xt_gk import _phi_grid, _phi_of_d

        g = _phi_grid((12, 16), alpha=2.1, beta=0.8, d_max=105.0, d_threshold=35.0)
        assert g.shape == (12, 16)
        np.testing.assert_allclose(g[0], g[5])  # depends on x (col) only -> rows identical
        xc = spadlconfig.field_length * (np.arange(16) + 0.5) / 16  # column-centre x
        np.testing.assert_allclose(g[0], _phi_of_d(xc, 2.1, 0.8, 105.0, 35.0))

    def test_grid_value_pinned_to_expected_threat_rate(self):
        """H1 anti-circularity: _grid_value's convention must equal ExpectedThreat.rate's,
        not merely match _xt_gk's own arithmetic. A successful pass's rate() value is
        xT(z') - xT(z); with sigma=0, _progress must reproduce it exactly."""
        xt = ExpectedThreat(l=16, w=12)
        xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
        a = pd.DataFrame(
            {
                "type_id": [spadlconfig.actiontype_id["pass"]],
                "result_id": [spadlconfig.result_id["success"]],
                "start_x": [20.0],
                "start_y": [30.0],
                "end_x": [80.0],
                "end_y": [40.0],
            }
        )
        rate_val = xt.rate(a)[0]
        prog = _progress(
            _grid_value(xt.xT, a["end_x"].to_numpy(), a["end_y"].to_numpy()),
            _grid_value(xt.xT, a["start_x"].to_numpy(), a["start_y"].to_numpy()),
        )[0]
        assert prog == pytest.approx(rate_val)


class TestComponents:
    def test_progress_is_destination_minus_origin(self):
        assert _progress(np.array([0.5]), np.array([0.1]))[0] == pytest.approx(0.4)

    def test_base_is_negative_origin(self):
        # Option B: base owns only the origin cost; RAV owns the destination value.
        assert _base(np.array([0.1]))[0] == pytest.approx(-0.1)

    def test_pev_rewards_forward_escape_under_pressure(self):
        progress = np.array([0.4, 0.4, -0.4])
        rho = np.array([0.0, 0.8, 0.8])
        pev = _pev(rho, progress)
        assert pev[0] == pytest.approx(0.0)  # no pressure -> no reward
        assert pev[1] == pytest.approx(0.8 * 0.4)  # pressure + forward -> reward
        assert pev[2] == pytest.approx(0.0)  # negative progress clamped

    def test_rav_completion_weighted_minus_risk(self):
        rav = _rav(
            p=np.array([0.7]),
            xt_star_dest=np.array([0.5]),
            xt_star_counter=np.array([0.3]),
            delta=0.5,
        )
        assert rav[0] == pytest.approx(0.7 * 0.5 - 0.5 * (1 - 0.7) * 0.3)

    def test_dzv_is_revaluation_increment_in_defensive_third(self):
        # Option A: DZV = (M-1)*V_GK(z), M = phi(z,d)*(1 - V_GK(z)/max V_GK); gated to def third.
        from silly_kicks.tracking._xt_gk import _phi_of_d

        vgk_origin = np.array([0.02, 0.02])  # small deep possession value
        start_x = np.array([10.0, 60.0])  # in def third, outside
        vgk_max = 1.0
        out = _dzv(start_x, vgk_origin, vgk_max, alpha=2.1, beta=0.8, d_max=105.0, boundary=35.0)
        phi0 = _phi_of_d(np.array([10.0]), 2.1, 0.8, 105.0, 35.0)[0]
        m0 = phi0 * (1 - 0.02 / 1.0)
        assert out[0] == pytest.approx((m0 - 1.0) * 0.02)  # positive increment in def third
        assert out[0] > 0.0
        assert out[1] == pytest.approx(0.0)  # outside def third -> 0

    def test_temporal_is_eta_to_the_k(self):
        np.testing.assert_allclose(_temporal(np.array([0, 1, 2]), 0.85), [1.0, 0.85, 0.85**2])

    def test_composite_discounts_threat_terms_only_not_dzv(self):
        # xT-GK = T*(base + gamma*PEV + RAV) + phi*DZV
        out = _composite(
            t=np.array([0.5]),
            base=np.array([-0.1]),
            pev=np.array([0.2]),
            rav=np.array([0.1]),
            dzv=np.array([0.03]),
            gamma=0.25,
            phi=1.0,
        )
        expected = 0.5 * (-0.1 + 0.25 * 0.2 + 0.1) + 1.0 * 0.03
        assert out[0] == pytest.approx(expected)


class TestDomainFilter:
    def _frames(self):
        # GK of team 1 is player 10; GK of team 2 is player 20
        return pd.DataFrame(
            {
                "game_id": [9, 9, 9, 9],
                "team_id": [1, 2, 1, 2],
                "player_id": [10, 20, 11, 21],
                "is_goalkeeper": [True, True, False, False],
                "is_ball": [False, False, False, False],
            }
        )

    def test_goalkick_always_in_scope(self):
        actions = pd.DataFrame({"game_id": [9], "team_id": [1], "player_id": [11], "type_id": [22]})
        assert _gk_distribution_mask(actions, self._frames()).tolist() == [True]

    def test_open_pass_in_scope_only_if_actor_is_gk(self):
        actions = pd.DataFrame(
            {
                "game_id": [9, 9],
                "team_id": [1, 1],
                "player_id": [10, 11],  # GK, outfielder
                "type_id": [0, 0],
            }
        )
        assert _gk_distribution_mask(actions, self._frames()).tolist() == [True, False]

    def test_throw_in_by_gk_in_scope(self):
        actions = pd.DataFrame({"game_id": [9], "team_id": [2], "player_id": [20], "type_id": [2]})
        assert _gk_distribution_mask(actions, self._frames()).tolist() == [True]

    def test_shot_never_in_scope(self):
        actions = pd.DataFrame({"game_id": [9], "team_id": [1], "player_id": [10], "type_id": [11]})
        assert _gk_distribution_mask(actions, self._frames()).tolist() == [False]

    def test_id_dtype_mismatch_string_frames(self):
        # ADR-019: numeric action ids vs string frame ids must still resolve actor-is-GK
        frames = self._frames().astype({"team_id": str, "player_id": str})
        actions = pd.DataFrame({"game_id": [9], "team_id": [1], "player_id": [10], "type_id": [0]})
        assert _gk_distribution_mask(actions, frames).tolist() == [True]


class TestComputeXtGk:
    def test_emits_all_columns_for_in_scope_only(self):
        actions = _gk_actions()
        actions.loc[2] = dict(  # an outfield pass -> out of scope
            game_id=9,
            action_id=2,
            team_id=1,
            player_id=11,
            period_id=1,
            time_seconds=5.0,
            type_id=0,
            start_x=40.0,
            start_y=34.0,
            end_x=60.0,
            end_y=34.0,
        )
        frames = _frames_for(actions)
        out = compute_xt_gk(actions, frames, xt=_fitted_xt())
        assert list(out.columns) == _XT_GK_COLS + _PROVENANCE_COLS  # value cols + provenance
        assert len(out) == len(actions)
        assert np.isnan(out.loc[2, "xt_gk"])  # type: ignore[arg-type]  # out-of-scope -> NaN
        assert out.loc[2, "xt_gk_origin_source"] is None  # off-scope -> NaN provenance
        assert not np.isnan(out.loc[0, "xt_gk_base"])  # type: ignore[arg-type]  # in-scope -> value
        assert out.loc[0, "xt_gk_origin_source"] == "native"  # native goalkick origin
        # Task 8 provenance: sportec frames -> "gs" variant; scored row -> "model" source; off-scope None.
        assert out.loc[0, "xt_gk_completion_variant"] == "gs"
        assert out.loc[0, "xt_gk_completion_source"] == "model"
        assert out.loc[2, "xt_gk_completion_variant"] is None

    def test_imputed_origin_goalkick_is_scored_and_tagged(self):
        # NaN-origin goalkick -> derived origin FEEDS compute -> non-NaN composite + tag (m7/m8).
        # The acting GK is moved off-position at the goalkick frame so the origin falls to the
        # rule-point prior (not the in-area tracking-GK tier).
        actions = _gk_actions()
        actions.loc[0, "start_x"] = np.nan  # row 0 goalkick: origin NaN
        frames = _frames_for(actions)
        off = (frames["frame_id"] == 0) & (frames["player_id"] == 10)
        frames.loc[off, "x"] = 40.0  # GK off position -> tracking tier declines -> prior
        out = compute_xt_gk(actions, frames, xt=_fitted_xt())
        assert not np.isnan(out.loc[0, "xt_gk"])  # type: ignore[arg-type]  # scored (was NaN before)
        assert out.loc[0, "xt_gk_origin_source"] == "goalkick_prior"
        assert out.loc[0, "xt_gk_origin_confidence"] < 0.7  # type: ignore[operator]

    def test_unresolvable_destination_routes_to_nan(self):
        # an in-scope goalkick whose destination cannot be resolved (NaN native end AND no
        # in-period next-event -> it is the LAST row) is honestly NaN (no z' => no RAV/xT*(z')),
        # NOT base-rated. Single-row fixture so there is no next event to borrow.
        actions = _gk_actions().iloc[[0]].copy()
        actions.loc[0, "end_x"] = np.nan
        actions.loc[0, "end_y"] = np.nan
        frames = _frames_for(actions)
        out = compute_xt_gk(actions, frames, xt=_fitted_xt())
        assert np.isnan(out.loc[0, "xt_gk"])  # type: ignore[arg-type]
        assert out.loc[0, "xt_gk_dest_source"] == "unresolved"
        assert len(out) == len(actions)

    def test_unlinked_in_scope_row_routes_to_nan(self):
        # an in-scope goalkick with NO matching frame (unlinked) cannot get a pressure value
        # -> NaN composite (no crash). Real cause: time-base gaps / truncated tracking windows.
        actions = _gk_actions()
        actions.loc[0, "time_seconds"] = 999.0  # no frame near this time -> unlinked
        frames = _frames_for(actions)
        out = compute_xt_gk(actions, frames, xt=_fitted_xt())
        assert np.isnan(out.loc[0, "xt_gk"])  # type: ignore[arg-type]  # unlinked -> NaN, no crash
        assert len(out) == len(actions)

    def test_rav_uses_completion_model_not_das(self):
        # with accessible-space monkeypatched ABSENT, RAV/composite is still computed (M4/R8):
        # the completion model replaced the open-play get_xc, so [das] is no longer required.
        import builtins

        real_import = builtins.__import__

        def no_as(name, *a, **k):
            if name == "accessible_space" or name.startswith("accessible_space."):
                raise ImportError("accessible_space disabled for test")
            return real_import(name, *a, **k)

        actions = _gk_actions()
        frames = _frames_for(actions)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(builtins, "__import__", no_as)
            out = compute_xt_gk(actions, frames, xt=_fitted_xt())
        assert not out["xt_gk_rav"].isna().all()  # RAV computed without [das]

    def test_uses_injected_grid_not_self_fit(self):
        # Option B: base = -xT*(origin); with sigma=0, xT* == injected grid -> deterministic.
        xt = _fitted_xt()
        actions = _gk_actions().iloc[[0]].copy()  # goalkick from start_x=5, start_y=34
        frames = _frames_for(actions)
        params = XtGkParams(convolution_sigma=0.0)
        out = compute_xt_gk(actions, frames, xt=xt, params=params)
        expected_base = -_grid_value(xt.xT, np.array([5.0]), np.array([34.0]))[0]
        assert out.loc[0, "xt_gk_base"] == pytest.approx(expected_base)

    def test_rejects_unfitted_grid(self):
        # M1 leakage contract: an all-zero (unfitted) grid raises.
        actions = _gk_actions().iloc[[0]].copy()
        frames = _frames_for(actions)
        unfitted = ExpectedThreat(l=16, w=12)  # xT.all() == 0
        with pytest.raises(ValueError, match="FITTED"):
            compute_xt_gk(actions, frames, xt=unfitted)

    def test_backpass_penalty_corrected_upward(self):
        # DZV raises deep-zone value (phi*DZV > 0): a defensive-third origin gets the published
        # revaluation increment (M-1)*V_GK(z) > 0 (phi(z,d) > 1 there). Uses the GK-zone-realistic
        # grid so V_GK stays small and DZV lands O(0.01). See _gk_realistic_xt docstring.
        actions = _gk_actions().iloc[[1]].copy()  # start_x=25 -> defensive third
        frames = _frames_for(actions)
        out = compute_xt_gk(actions, frames, xt=_gk_realistic_xt(), params=XtGkParams(phi=1.0))
        assert out.loc[1, "xt_gk_dzv"] > 0.0  # type: ignore[operator]
        # NOTE (review #3): we deliberately do NOT also assert with_dzv.xt_gk >
        # without_dzv.xt_gk -- that routes the claim through the composite's xC (can be NaN
        # on a synthetic fixture -> NaN>NaN false-fail). assertion 1 + the P1-3 _composite
        # unit oracle already prove "DZV raises the composite" xC-free.

    def test_pev_reads_revalued_surface_not_raw(self):
        # CHANGE 1: a short deep build-out (origin x=5 -> dest x=25, both in the flat def third)
        # has ~zero RAW gain -> PEV ~0 with revaluation OFF; canonical phi lights it up. PEV is
        # pressure-gated (rho*max(0,progress)), so the GK must face real pressure for PEV to
        # surface the gain at all -> add two near opponents at the goalkick frame.
        actions = _gk_actions().iloc[[0]].copy()
        actions.loc[0, "start_x"] = 5.0
        actions.loc[0, "end_x"] = 25.0  # stays in the flat defensive third
        frames = _frames_for(actions)
        pressers = pd.DataFrame(
            [
                dict(
                    game_id=9,
                    period_id=1,
                    frame_id=0,
                    time_seconds=5.0,
                    frame_rate=25.0,
                    team_id=2,
                    player_id=pid,
                    is_goalkeeper=False,
                    is_ball=False,
                    x=x,
                    y=y,
                    vx=-0.5,
                    vy=0.0,
                    team_in_possession=1,
                    source_provider="sportec",
                    team_attacking_direction="ltr",
                )
                for pid, x, y in [(23, 6.0, 35.0), (24, 7.0, 33.0)]
            ]
        )
        frames = pd.concat([frames, pressers], ignore_index=True)
        off = XtGkParams(dzv_alpha=1.0, dzv_beta=0.0)  # phi == 1 -> V_GK == xT (revaluation disabled)
        on = XtGkParams()  # canonical alpha=2.1, beta=0.8
        pev_off = compute_xt_gk(actions, frames, xt=_deep_flat_xt(), params=off).loc[0, "xt_gk_pev"]
        pev_on = compute_xt_gk(actions, frames, xt=_deep_flat_xt(), params=on).loc[0, "xt_gk_pev"]
        assert pev_off == pytest.approx(0.0, abs=1e-4)  # raw deep gain flatlines (even under pressure)
        assert pev_on > pev_off  # type: ignore[operator]  # revaluation restores the gain

    def test_phi_shape_changes_only_pev_and_dzv_not_base_or_rav(self):
        # The invariant Eyestone fixed: phi enters value via PEV + DZV ONLY. Base + RAV (raw xT*)
        # must be byte-identical when the phi shape changes.
        actions = _gk_actions()
        frames = _frames_for(actions)
        a = compute_xt_gk(actions, frames, xt=_gk_realistic_xt(), params=XtGkParams(dzv_alpha=2.1, dzv_beta=0.8))
        b = compute_xt_gk(actions, frames, xt=_gk_realistic_xt(), params=XtGkParams(dzv_alpha=3.5, dzv_beta=1.5))
        np.testing.assert_array_equal(a["xt_gk_base"].to_numpy(), b["xt_gk_base"].to_numpy())
        np.testing.assert_array_equal(a["xt_gk_rav"].to_numpy(), b["xt_gk_rav"].to_numpy())
        assert not np.allclose(  # at least one of PEV/DZV moved (the in-scope, in-def-third rows)
            np.nan_to_num(a[["xt_gk_pev", "xt_gk_dzv"]].to_numpy()),
            np.nan_to_num(b[["xt_gk_pev", "xt_gk_dzv"]].to_numpy()),
        )

    def test_dzv_scale_is_order_hundredth_not_unity(self):
        # Scale anchor (Eyestone): per-action DZV must land O(0.01), not the literal multiplier
        # O(2.5). Back-pass origin x=25 is in the defensive third on the realistic grid.
        actions = _gk_actions().iloc[[1]].copy()
        frames = _frames_for(actions)
        dzv = compute_xt_gk(actions, frames, xt=_gk_realistic_xt()).loc[1, "xt_gk_dzv"]
        assert 0.0 < dzv < 0.5  # type: ignore[operator]  # positive bar, two-plus orders below raw ~2.5

    def test_higher_pressure_gives_higher_pev(self):
        actions = _gk_actions().iloc[[0]].copy()
        low = _frames_for(actions)
        extra = pd.DataFrame(
            [
                dict(
                    game_id=9,
                    period_id=1,
                    frame_id=0,
                    time_seconds=5.0,
                    frame_rate=25.0,
                    team_id=2,
                    player_id=23,
                    is_goalkeeper=False,
                    is_ball=False,
                    x=6.0,
                    y=35.0,
                    vx=-0.5,
                    vy=0.0,
                    team_in_possession=1,
                    source_provider="sportec",
                    team_attacking_direction="ltr",
                ),
                dict(
                    game_id=9,
                    period_id=1,
                    frame_id=0,
                    time_seconds=5.0,
                    frame_rate=25.0,
                    team_id=2,
                    player_id=24,
                    is_goalkeeper=False,
                    is_ball=False,
                    x=7.0,
                    y=33.0,
                    vx=-0.5,
                    vy=0.0,
                    team_in_possession=1,
                    source_provider="sportec",
                    team_attacking_direction="ltr",
                ),
            ]
        )
        high = pd.concat([low, extra], ignore_index=True)
        out_low = compute_xt_gk(actions, low, xt=_fitted_xt())
        out_high = compute_xt_gk(actions, high, xt=_fitted_xt())
        assert out_high.loc[0, "xt_gk_pressure"] > out_low.loc[0, "xt_gk_pressure"]  # type: ignore[operator]
        assert out_high.loc[0, "xt_gk_pev"] >= out_low.loc[0, "xt_gk_pev"]  # type: ignore[operator]


class TestAddXtGk:
    def test_merges_columns_and_provenance(self):
        actions = _gk_actions()
        frames = _frames_for(actions)
        out = add_xt_gk(actions, frames, _fitted_xt(), home_team_id=1)
        for c in _XT_GK_COLS + _PROVENANCE_COLS:  # value cols + xT-GK provenance
            assert c in out.columns
        assert "frame_id" in out.columns  # linkage provenance
        assert len(out) == len(actions)
        # every in-scope scored row carries a non-null origin/dest source (spec section 6)
        assert out["xt_gk_origin_source"].notna().all()
        assert out["xt_gk_dest_source"].notna().all()

    def test_idempotent_provenance_on_chained_calls(self):
        actions = _gk_actions()
        frames = _frames_for(actions)
        once = add_xt_gk(actions, frames, _fitted_xt(), home_team_id=1)
        twice = add_xt_gk(once, frames, _fitted_xt(), home_team_id=1)
        assert "frame_id_x" not in twice.columns
        assert "frame_id_y" not in twice.columns

    def test_nan_identifier_routes_to_default_no_crash(self):
        actions = _gk_actions()
        actions.loc[0, "player_id"] = np.nan
        frames = _frames_for(actions)
        out = add_xt_gk(actions, frames, _fitted_xt(), home_team_id=1)
        assert len(out) == len(actions)
        assert np.isnan(out.loc[0, "xt_gk"])  # type: ignore[arg-type]


class TestXtGkXfns:
    def test_factory_returns_frame_aware_transformer(self):
        fns = xt_gk_xfns(_fitted_xt(), home_team_id=1)
        assert len(fns) == 1
        assert getattr(fns[0], "_frame_aware", False) is True

    def test_produces_values_on_duplicate_action_ids(self):
        base = _gk_actions()
        frames = _frames_for(base)
        slot = pd.concat([base, base.iloc[[0]]], ignore_index=True)  # dup action_id=0
        states = [slot, slot, slot]
        fn = xt_gk_xfns(_fitted_xt(), home_team_id=1)[0]
        res = fn(states, frames)
        assert "xt_gk_a0" in res.columns
        assert len(res) == len(slot)
        # the two rows sharing action_id=0 link to the same frame -> same value
        assert res["xt_gk_base_a0"].iloc[0] == pytest.approx(res["xt_gk_base_a0"].iloc[-1], nan_ok=True)

    def test_none_frames_yields_nan_columns(self):
        base = _gk_actions()
        fn = xt_gk_xfns(_fitted_xt(), home_team_id=1)[0]
        res = fn([base, base, base], None)
        assert res["xt_gk_a0"].isna().all()


class TestExports:
    def test_public_surface_importable_from_tracking(self):
        import silly_kicks.tracking as T

        for name in ("compute_xt_gk", "add_xt_gk", "xt_gk_xfns", "XtGkParams", "XtGkReport"):
            assert hasattr(T, name), name
            assert name in T.__all__, name


class TestXtGkReport:
    def test_counts_equal_column_value_counts(self):
        # spec section 6: XtGkReport counts == the output columns' value_counts (by construction).
        actions = _gk_actions()
        actions.loc[0, "start_x"] = np.nan  # one imputed-origin goalkick
        frames = _frames_for(actions)
        off = (frames["frame_id"] == 0) & (frames["player_id"] == 10)
        frames.loc[off, "x"] = 40.0  # -> goalkick_prior origin
        out = compute_xt_gk(actions, frames, xt=_fitted_xt())
        rep = XtGkReport.from_frame(out)
        assert rep.n_rows == len(out)
        assert rep.n_scored == int(out["xt_gk"].notna().sum())
        assert rep.origin_source_counts == {
            str(k): int(v) for k, v in out["xt_gk_origin_source"].value_counts(dropna=True).items()
        }
        assert rep.dest_source_counts == {
            str(k): int(v) for k, v in out["xt_gk_dest_source"].value_counts(dropna=True).items()
        }
        assert "goalkick_prior" in rep.origin_source_counts  # the imputed row is tallied
        # completion-variant provenance (Task 8, m-c): single provider -> not spanning variants
        assert rep.completion_variant_counts == {
            str(k): int(v) for k, v in out["xt_gk_completion_variant"].value_counts(dropna=True).items()
        }
        assert rep.spans_multiple_variants is False

    def test_spans_multiple_variants_flags_mixed(self):
        # A frame manually stitched from two variants trips the runtime no-pool signal (m-c).
        actions = _gk_actions()
        frames = _frames_for(actions)
        out = compute_xt_gk(actions, frames, xt=_fitted_xt())
        mixed = pd.concat(
            [out.assign(xt_gk_completion_variant="gs"), out.assign(xt_gk_completion_variant="skillcorner")]
        )
        assert XtGkReport.from_frame(mixed).spans_multiple_variants is True


class TestAtomicMirror:
    def test_atomic_add_xt_gk_matches_standard_via_synthesis(self):
        from silly_kicks.atomic.tracking.features import add_xt_gk as atomic_add_xt_gk

        std = _gk_actions()
        frames = _frames_for(std)
        std_out = add_xt_gk(std, frames, _fitted_xt(), home_team_id=1)

        atom = std.rename(columns={"start_x": "x", "start_y": "y"}).copy()
        atom["dx"] = std["end_x"].to_numpy() - std["start_x"].to_numpy()
        atom["dy"] = std["end_y"].to_numpy() - std["start_y"].to_numpy()
        atom = atom.drop(columns=["end_x", "end_y"])
        atom_out = atomic_add_xt_gk(atom, frames, _fitted_xt(), home_team_id=1)

        np.testing.assert_allclose(
            atom_out["xt_gk_base"].to_numpy(),
            std_out["xt_gk_base"].to_numpy(),
            equal_nan=True,
            rtol=1e-9,
        )
        # provenance columns ride through the atomic synthesis unchanged (not dropped)
        for c in _PROVENANCE_COLS:
            assert c in atom_out.columns
        assert atom_out["xt_gk_origin_source"].to_numpy().tolist() == std_out["xt_gk_origin_source"].to_numpy().tolist()
        # the completion variant/source provenance (Task 8) is identical through the atomic mirror
        assert (
            atom_out["xt_gk_completion_variant"].to_numpy().tolist()
            == std_out["xt_gk_completion_variant"].to_numpy().tolist()
        )
        assert (
            atom_out["xt_gk_completion_source"].to_numpy().tolist()
            == std_out["xt_gk_completion_source"].to_numpy().tolist()
        )

    def test_atomic_exports(self):
        import silly_kicks.atomic.tracking.features as AF

        assert "add_xt_gk" in AF.__all__
        assert "xt_gk_xfns" in AF.__all__
