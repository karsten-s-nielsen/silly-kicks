"""TF-64 GK detection-gate -- end-to-end FIRING tests for the three wired consumers whose gate
``test_gk_detection_gate.py`` leaves unexercised (gkdv engine, gk_decision Tier B reconstruction,
``pre_shot_gk_*``). Each is driven from a provider-tagged SYNTHETIC fixture rather than the
SkillCorner slim: the slim's per-player detection flag was not preserved (all-null), so these tag a
hand-built in-domain fixture ``source_provider="skillcorner"`` and flip the DEFENDING keeper's
``visibility`` -- the same idiom the restdefense test in the sibling file already uses.

Both sides per consumer (spec 2026-10-01 §4 test 1): ``visibility`` True -> numeric / scored, False
-> honest-NaN / counted drop, with a non-vacuity guard that the gate actually MOVED the output.
Without these, ``test_keeper_position_reader_gate`` (mechanical ADR-056, string-presence only) would
stay green through a refactor that keeps the gate symbol but breaks the drop -- a silent E7 regression
on exactly these three sites (reviewer A TF64-IMPL-02).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.id_compat import ids_match


def _keeper_undetected_mask(frames: pd.DataFrame, player_id: object) -> np.ndarray:
    """Positional mask for the keeper rows of ``player_id`` (dtype-safe; numpy throughout so the
    fresh-RangeIndex of ``ids_match`` never mis-aligns against a labelled Series)."""
    return (
        frames["is_goalkeeper"].astype(bool).to_numpy()
        & ~frames["is_ball"].astype(bool).to_numpy()
        & ids_match(frames["player_id"], player_id).to_numpy()
    )


# --- gkdv engine (build_ghost_frames domain cascade: _DROP_KEEPER_UNDETECTED) -------------------


def test_gkdv_engine_drops_undetected_defending_keeper_frame():
    from silly_kicks.gkdv import build_ghost_frames
    from tests.gkdv._fixtures import in_domain_frames
    from tests.tracking.test_ghost_gk import _fitted_model

    model = _fitted_model()[0]
    base = in_domain_frames()
    base["source_provider"] = "skillcorner"
    base["visibility"] = True

    # observed: the detection-aware frame is in-domain and scores, no keeper_undetected drop.
    _cf, _prov, rep_obs = build_ghost_frames(base, model=model, home_team_id=1)
    assert rep_obs.n_frames_scored == 1, rep_obs.drop_reasons
    assert rep_obs.drop_reasons.get("keeper_undetected", 0) == 0

    # gated: the DEFENDING keeper (home team 1, player p1) is present-but-undetected -> the frame
    # drops under _DROP_KEEPER_UNDETECTED rather than scoring on an ~80%-extrapolated position.
    und = base.copy()
    home_gk = _keeper_undetected_mask(und, "p1") & ids_match(und["team_id"], 1).to_numpy()
    und.loc[home_gk, "visibility"] = False
    _cf2, _prov2, rep_gate = build_ghost_frames(und, model=model, home_team_id=1)
    assert rep_gate.drop_reasons.get("keeper_undetected", 0) >= 1
    assert rep_gate.n_frames_scored == 0
    # non-vacuity: the gate moved a frame out of the scored set (not an already-empty domain).
    assert rep_obs.n_frames_scored > rep_gate.n_frames_scored


# --- gk_decision Tier B reconstruction (ReconstructedOptionSet: keeper_undetected drop) ----------


def test_gk_decision_reconstruct_drops_undetected_keeper():
    from silly_kicks.gk_decision import GkDecisionParams, ReconstructedOptionSet
    from tests.gk_decision.test_reconstruct import _FakeXPass, _one_decision

    actions, frames = _one_decision()
    frames = frames.copy()
    frames["source_provider"] = "skillcorner"
    frames["visibility"] = True

    def _rows(f):
        os_ = ReconstructedOptionSet(
            actions,
            f,
            xpass=_FakeXPass(),
            params=GkDecisionParams(),
            keeper_ids=[99],
            frame_convention="per_action_ltr",
        )
        return os_.option_rows(), os_.drop_counts()

    rows_obs, drops_obs = _rows(frames)
    assert len(rows_obs) >= 1
    assert drops_obs.get("keeper_undetected", 0) == 0

    und = frames.copy()
    und.loc[_keeper_undetected_mask(und, 99), "visibility"] = False
    rows_gate, drops_gate = _rows(und)
    assert drops_gate.get("keeper_undetected", 0) == 1
    assert len(rows_gate) == 0
    # non-vacuity: the observed decision really produced options the gate then suppressed.
    assert len(rows_obs) > len(rows_gate)


# --- pre_shot_gk_* enrichers (utils._resolve_action_frame_context defending_gk_rows gate) --------

_PRE_SHOT_GK_COLS = (
    "pre_shot_gk_x",
    "pre_shot_gk_y",
    "pre_shot_gk_distance_to_goal",
    "pre_shot_gk_distance_to_shot",
)


def _pre_shot_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """1 shot at (90,34), defending GK (team 2, player 99) at (104,34) in the linked frame -- the
    known-numeric geometry of ``test_features_standard.shot_actions_and_frames_with_gk``, tagged for
    the detection-aware provider."""
    from silly_kicks.spadl import config as spadlconfig

    actions = pd.DataFrame(
        {
            "action_id": [201],
            "period_id": [1],
            "time_seconds": [10.0],
            "team_id": [1],
            "player_id": [11],
            "type_id": [spadlconfig.actiontype_id["shot"]],
            "start_x": [90.0],
            "start_y": [34.0],
            "end_x": [105.0],
            "end_y": [34.0],
            "defending_gk_player_id": [99.0],
        }
    )

    def _row(player_id, team_id, *, is_ball, is_gk, x, y):
        return dict(
            game_id=1,
            period_id=1,
            frame_id=2000,
            time_seconds=10.0,
            frame_rate=25.0,
            player_id=player_id,
            team_id=team_id,
            is_ball=is_ball,
            is_goalkeeper=is_gk,
            x=x,
            y=y,
            z=float("nan"),
            speed=1.0,
            speed_source="native",
            ball_state="alive",
            team_attacking_direction="ltr",
            confidence=None,
            visibility=True,
            source_provider="skillcorner",
        )

    frames = pd.DataFrame(
        [
            _row(11, 1, is_ball=False, is_gk=False, x=90.0, y=34.0),
            _row(99, 2, is_ball=False, is_gk=True, x=104.0, y=34.0),
            _row(float("nan"), float("nan"), is_ball=True, is_gk=False, x=90.0, y=34.0),
        ]
    )
    return actions, frames


def test_pre_shot_gk_position_nan_when_defending_keeper_undetected():
    from silly_kicks.tracking.features import add_pre_shot_gk_position

    actions, frames = _pre_shot_fixture()

    obs = add_pre_shot_gk_position(actions, frames)
    assert np.isfinite(obs["pre_shot_gk_x"].iloc[0]), "observed keeper must yield a numeric position"

    und = frames.copy()
    und.loc[_keeper_undetected_mask(und, 99), "visibility"] = False
    gate = add_pre_shot_gk_position(actions, und)
    for col in _PRE_SHOT_GK_COLS:
        assert gate[col].isna().iloc[0], f"{col}: undetected keeper must honest-NaN, got {gate[col].iloc[0]}"
    # non-vacuity: observed numeric, gated NaN -> the gate changed the output.
    assert np.isfinite(obs["pre_shot_gk_x"].iloc[0]) and bool(gate["pre_shot_gk_x"].isna().iloc[0])
