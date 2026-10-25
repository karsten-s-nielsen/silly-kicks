"""Task 2.8 origin tiers (spec section 10.1): the origin_* provenance columns, the gated stationary_window
tier, and the commit-2 dormant behaviour (empty gate -> no new tier ships, existing tiers unchanged)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as cfg
from silly_kicks.tracking import _restart_validation as rv
from silly_kicks.tracking._gk_geometry import resolve_restart_geometry
from silly_kicks.tracking._restart_validation import DataProfile, GateEntry
from tests.tracking._restart_fixtures import GateTableStub

_P = DataProfile("skillcorner", "r1", "g1", "2025")
_FKS = cfg.actiontype_id["freekick_short"]


def _acts(**extra):
    base = {
        "game_id": "g",
        "period_id": 1,
        "action_id": 0,
        "time_seconds": 10.0,
        "type_id": _FKS,
        "start_x": 30.0,
        "start_y": 40.0,
        "end_x": 50.0,
        "end_y": 34.0,
    }
    base.update(extra)
    return pd.DataFrame([base])


def test_origin_columns_present_and_dormant_in_c2():
    out = resolve_restart_geometry(_acts(kick_rest_x=31.0, kick_rest_y=41.0), data_profile=None)
    assert {"origin_evidence", "origin_gate_version", "origin_cell_resolution"} <= set(out.columns)
    assert pd.isna(out.loc[0, "origin_evidence"])
    assert out.loc[0, "start_coord_source"] == "native"  # no new tier -> unchanged


def test_stationary_window_ships_with_injected_gate(monkeypatch):
    monkeypatch.setattr(rv, "gate", lambda *a, **k: GateEntry("transfer", "idsse", 0.95, 0.91, 0.91, 40, 5, "cell"))
    out = resolve_restart_geometry(
        _acts(start_x=np.nan, start_y=np.nan, kick_rest_x=31.0, kick_rest_y=41.0), data_profile=_P
    )
    assert out.loc[0, "start_coord_source"] == "stationary_window"
    assert out.loc[0, "enriched_start_x"] == 31.0
    assert out.loc[0, "origin_evidence"] == "transfer" and out.loc[0, "origin_cell_resolution"] == "cell"


# ---- the remaining section 10.1 tiers, each behind its OWN cell (restart type x origin tier) ----

_TI = cfg.actiontype_id["throw_in"]
_GK = cfg.actiontype_id["goalkick"]


class _Cells(GateTableStub):
    """A gate table passing only the listed (restart_type, cell_state) origin cells."""

    def __init__(self, *cells):
        self.cells = set(cells)

    def get(self, key, default=None):
        _profile, rt, sig, cs = key
        ok = sig == "origin" and (rt, cs) in self.cells
        return GateEntry("transfer", "idsse", 0.95, 0.91, 0.91, 40, 5, "cell") if ok else default


def _frames(ball_xy=(40.0, 0.4), gk_xy=(4.0, 34.0), gk_team="H", visible=True):
    """Two frames around t = 10.0: the ball at ``ball_xy``; the H keeper at ``gk_xy``; an A keeper at the other
    end. H attacks +x (its keeper defends x = 0), so action frame == frame convention for H."""
    rows = []
    for fid, t in ((1, 9.96), (2, 10.0)):
        rows.append(
            dict(
                game_id="g",
                period_id=1,
                frame_id=fid,
                time_seconds=t,
                player_id=None,
                team_id=None,
                is_ball=True,
                is_goalkeeper=False,
                x=ball_xy[0],
                y=ball_xy[1],
                visibility=True,
            )
        )
        rows.append(
            dict(
                game_id="g",
                period_id=1,
                frame_id=fid,
                time_seconds=t,
                player_id="hk",
                team_id=gk_team,
                is_ball=False,
                is_goalkeeper=True,
                x=gk_xy[0],
                y=gk_xy[1],
                visibility=visible,
            )
        )
        rows.append(
            dict(
                game_id="g",
                period_id=1,
                frame_id=fid,
                time_seconds=t,
                player_id="ak",
                team_id="A",
                is_ball=False,
                is_goalkeeper=True,
                x=101.0,
                y=34.0,
                visibility=True,
            )
        )
    df = pd.DataFrame(rows)
    # ADR-028: the frame-sampled tiers re-project via acting_team_attacks_rtl, which reads the per-row labels a
    # real converter emits (H attacks +x, A attacks -x); the ball row carries none.
    df["team_attacking_direction"] = df["team_id"].map({"H": "ltr", "A": "rtl"})
    df["source_provider"] = "sportec"
    return df


def test_t_kick_ball_tier_for_a_throw_in_needs_its_own_cell():
    acts = _acts(type_id=_TI, team_id="H", start_x=np.nan, start_y=np.nan, t_kick=10.0)
    frames = _frames(ball_xy=(40.0, 0.4))
    out = resolve_restart_geometry(acts, frames=frames, data_profile=_P, gate_table=_Cells(("throw_in", "t_kick_ball")))
    assert out.loc[0, "start_coord_source"] == "t_kick_ball"
    assert (out.loc[0, "enriched_start_x"], out.loc[0, "enriched_start_y"]) == (40.0, 0.4)
    # the stationary_window cell alone does not license the t_kick_ball tier
    out2 = resolve_restart_geometry(
        acts, frames=frames, data_profile=_P, gate_table=_Cells(("throw_in", "stationary_window"))
    )
    assert out2.loc[0, "start_coord_source"] != "t_kick_ball" and pd.isna(out2.loc[0, "origin_evidence"])


def test_t_kick_ball_withheld_when_t_kick_is_withheld():
    acts = _acts(type_id=_TI, team_id="H", start_x=np.nan, start_y=np.nan, t_kick=np.nan)
    out = resolve_restart_geometry(
        acts, frames=_frames(), data_profile=_P, gate_table=_Cells(("throw_in", "t_kick_ball"))
    )
    assert out.loc[0, "start_coord_source"] != "t_kick_ball"


def test_goal_kick_tracking_gk_is_anchored_to_the_bracket():
    acts = _acts(
        type_id=_GK, team_id="H", start_x=np.nan, start_y=np.nan, t_kick=10.0, kick_bracket_lo=9.0, kick_bracket_hi=11.0
    )
    cells = _Cells(("goal_kick", "tracking_gk"))
    out = resolve_restart_geometry(acts, frames=_frames(gk_xy=(4.0, 30.0)), data_profile=_P, gate_table=cells)
    assert out.loc[0, "start_coord_source"] == "tracking_gk" and out.loc[0, "origin_evidence"] == "transfer"
    assert (out.loc[0, "enriched_start_x"], out.loc[0, "enriched_start_y"]) == (4.0, 30.0)
    # a bracket that excludes every keeper detection -> no gated tier (fails closed)
    far = acts.assign(kick_bracket_lo=20.0, kick_bracket_hi=21.0)
    out2 = resolve_restart_geometry(far, frames=_frames(gk_xy=(4.0, 30.0)), data_profile=_P, gate_table=cells)
    assert pd.isna(out2.loc[0, "origin_evidence"])


def test_goal_kick_tracking_gk_off_position_keeper_is_not_an_origin():
    acts = _acts(
        type_id=_GK, team_id="H", start_x=np.nan, start_y=np.nan, t_kick=10.0, kick_bracket_lo=9.0, kick_bracket_hi=11.0
    )
    out = resolve_restart_geometry(
        acts, frames=_frames(gk_xy=(30.0, 34.0)), data_profile=_P, gate_table=_Cells(("goal_kick", "tracking_gk"))
    )
    assert pd.isna(out.loc[0, "origin_evidence"])  # beyond the 16.5 m goal-area depth


def test_frame_sampled_tier_is_reprojected_to_the_action_frame():
    # A attacks x = 0 (its keeper defends 105): a throw-in ball at raw (40, 0.4) is (65, 67.6) in A's action frame.
    acts = _acts(type_id=_TI, team_id="A", start_x=np.nan, start_y=np.nan, t_kick=10.0)
    out = resolve_restart_geometry(
        acts, frames=_frames(ball_xy=(40.0, 0.4)), data_profile=_P, gate_table=_Cells(("throw_in", "t_kick_ball"))
    )
    assert out.loc[0, "enriched_start_x"] == 65.0 and abs(float(out["enriched_start_y"].iloc[0]) - 67.6) < 1e-9


def test_a_gated_goal_kick_origin_survives_distrust():
    acts = _acts(
        type_id=_GK, team_id="H", start_x=20.0, start_y=34.0, t_kick=10.0, kick_bracket_lo=9.0, kick_bracket_hi=11.0
    )
    out = resolve_restart_geometry(
        acts,
        frames=_frames(gk_xy=(4.0, 30.0)),
        data_profile=_P,
        distrust_native_origin=True,
        gate_table=_Cells(("goal_kick", "tracking_gk")),
    )
    assert out.loc[0, "start_coord_source"] == "tracking_gk" and out.loc[0, "origin_evidence"] == "transfer"


def test_a_tripwire_reverted_gated_origin_drops_its_gate_provenance():
    # A gated t_kick_ball throw-in origin at (40, 30) is outside the throw-in Law region: the add_restart_coordinates
    # tripwire reverts it, and the reverted row must NOT keep claiming a gated origin (evidence / version / cell).
    import warnings

    from silly_kicks.spadl.utils import add_restart_coordinates

    acts = _acts(type_id=_TI, team_id="H", start_x=np.nan, start_y=np.nan, t_kick=10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = add_restart_coordinates(
            acts, frames=_frames(ball_xy=(40.0, 30.0)), data_profile=_P, gate_table=_Cells(("throw_in", "t_kick_ball"))
        )
    r = out.iloc[0]
    assert r["start_coord_source"] == "tripwire_reverted"
    assert pd.isna(r["origin_evidence"]) and pd.isna(r["origin_gate_version"]) and pd.isna(r["origin_cell_resolution"])
