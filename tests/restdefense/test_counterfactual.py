"""TF-60 Task 6: build_restdefense_ghost_frames -- PURE, conserving, moves only A's rearguard."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.restdefense._counterfactual import build_restdefense_ghost_frames
from tests.tracking.test_ghost_gk import _fitted_model
from tests.tracking.test_ghost_outfield_model import _fit_toy


def _carrier(team=1):
    return pd.DataFrame([{"game_id": "G1", "period_id": 1, "frame_id": 1000, "ball_carrier_team_id": team}])


def _committed_frame() -> pd.DataFrame:
    """team 1 in possession, defends x=0, ball at x=60 -> |60-0|=60 >= 52.5 committed-forward."""
    base = dict(
        game_id="G1",
        period_id=1,
        frame_id=1000,
        time_seconds=100.0,
        ball_state="alive",
        source_provider="gradientsports",
    )

    def _r(pid, team, x, y, *, gk=False, ball=False):
        return {
            **base,
            "player_id": (pd.NA if ball else pid),
            "team_id": (pd.NA if ball else team),
            "x": float(x),
            "y": float(y),
            "vx": 0.0,
            "vy": 0.0,
            "speed": 0.0,
            "is_ball": ball,
            "is_goalkeeper": gk,
            "team_attacking_direction": None,
        }

    rows = [
        _r("ball", None, 60, 34, ball=True),
        _r("h_gk", 1, 5, 34, gk=True),
        _r("h1", 1, 15, 20),
        _r("h2", 1, 18, 30),
        _r("h3", 1, 20, 40),
        _r("h4", 1, 22, 48),  # rearguard
        _r("h5", 1, 55, 30),
        _r("h6", 1, 58, 40),  # forward
        _r("a_gk", 2, 100, 34, gk=True),
        _r("a1", 2, 80, 20),
        _r("a2", 2, 82, 34),
        _r("a3", 2, 84, 48),
        _r("a4", 2, 86, 30),
    ]
    return pd.DataFrame(rows)


def test_pure_and_conserves():
    model, _ = _fit_toy()
    frames = _committed_frame()
    before = frames.copy(deep=True)
    _cf, _prov, rep = build_restdefense_ghost_frames(
        frames, which="rearguard", model=model, home_team_id=1, carrier=_carrier()
    )
    pd.testing.assert_frame_equal(frames, before)  # PURE
    assert rep.n_frames_scored + sum(rep.drop_reasons.values()) == rep.n_frames_in
    assert rep.n_frames_scored == 1


def test_writeback_moves_only_As_rearguard():
    model, _ = _fit_toy()
    frames = _committed_frame()
    cf, prov, _rep = build_restdefense_ghost_frames(
        frames, which="rearguard", model=model, home_team_id=1, carrier=_carrier()
    )
    scored = prov[prov["drop_reason"].isna()]
    assert len(scored) >= 1
    merged = cf.merge(frames, on=["game_id", "period_id", "frame_id", "player_id"], suffixes=("_cf", ""))
    changed = merged[(merged["x_cf"] != merged["x"]) | (merged["y_cf"] != merged["y"])]
    assert set(changed["player_id"]) == set(scored["player_id"])
    assert set(changed["player_id"]) <= {"h1", "h2", "h3", "h4"}  # team-1 deepest-4 rearguard only
    assert (changed["team_id"] == 1).all()


def test_not_committed_frame_is_dropped_and_counted_not_zero():
    model, _ = _fit_toy()
    frames = _committed_frame()
    frames.loc[frames["is_ball"], "x"] = 20.0  # |20-0|=20 < 52.5 -> not committed-forward
    cf, _prov, rep = build_restdefense_ghost_frames(
        frames, which="rearguard", model=model, home_team_id=1, carrier=_carrier()
    )
    assert rep.n_frames_scored == 0
    assert rep.drop_reasons.get("not_committed_forward", 0) == 1
    pd.testing.assert_frame_equal(cf, frames)  # nothing substituted -> cf == frames


def test_keeper_path_scores_and_moves_only_As_keeper():
    """which='keeper' must SCORE A's committed-forward frame and substitute A's keeper.

    Regression (found at T10): the ghost-GK serve is keyed (frame, gk_team_id) with NO player_id,
    while the provenance/write-back match on player_id -> the served keeper row joined nothing,
    the eligible frame fell to `no_ghost_served`, and the keeper was NEVER substituted (silent).
    """
    model = _fitted_model()[0]
    frames = _committed_frame()
    cf, prov, rep = build_restdefense_ghost_frames(
        frames, which="keeper", model=model, home_team_id=1, carrier=_carrier()
    )
    assert rep.n_frames_scored == 1
    assert rep.drop_reasons.get("no_ghost_served", 0) == 0
    scored = prov[prov["drop_reason"].isna()]
    assert set(scored["player_id"]) == {"h_gk"}  # A's (in-possession) keeper only
    merged = cf.merge(frames, on=["game_id", "period_id", "frame_id", "player_id"], suffixes=("_cf", ""))
    changed = merged[(merged["x_cf"] != merged["x"]) | (merged["y_cf"] != merged["y"])]
    assert set(changed["player_id"]) == {"h_gk"}
    assert (changed["team_id"] == 1).all()


def test_nonfinite_computed_ghost_on_scored_frame_raises(monkeypatch):
    model, _ = _fit_toy()
    frames = _committed_frame()

    def _bad_serve(fr, **kwargs):
        return pd.DataFrame(
            [
                {
                    "game_id": "G1",
                    "period_id": 1,
                    "frame_id": 1000,
                    "team_id": 1,
                    "slot_index": 1.0,
                    "player_id": "h1",
                    "ghost_gr_x": 10.0,
                    "ghost_gr_y": 20.0,
                    "ghost_x": np.nan,
                    "ghost_y": np.nan,
                    "ghost_outfield_source": "computed",
                }
            ]
        )

    import silly_kicks.tracking as tracking

    monkeypatch.setattr(tracking, "serve_ghost_outfield_positions", _bad_serve)
    with pytest.raises(ValueError, match="non-finite"):
        build_restdefense_ghost_frames(frames, which="rearguard", model=model, home_team_id=1, carrier=_carrier())
