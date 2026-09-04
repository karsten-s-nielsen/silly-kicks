"""compute_territorial_dominance -- exact conceded/prevented over a hand-placed scene + edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.territory import TerritoryParams, compute_territorial_dominance
from silly_kicks.xthreat import ExpectedThreat

_SUCCESS = spadlconfig.result_id["success"]
_FAIL = spadlconfig.result_id["fail"]
_TACKLE = spadlconfig.actiontype_id["tackle"]
_PASS = spadlconfig.actiontype_id["pass"]
_KEEP_ALL = TerritoryParams(trim_fraction=1.0)  # exact hull = every seeded action


def _toy_xt(value: float = 0.1) -> ExpectedThreat:
    # A uniform xT grid -> values_at_points returns `value` for ANY destination, so conceded/prevented
    # reduce to `value * count`, making the assertions exact.
    xt = ExpectedThreat()
    xt.xT = np.full(np.asarray(xt.xT).shape, value, dtype=float)
    return xt


def _def(game, player, team, x, y):
    return {
        "game_id": game,
        "period_id": 1,
        "team_id": team,
        "player_id": player,
        "type_id": _TACKLE,
        "result_id": _SUCCESS,
        "start_x": x,
        "start_y": y,
        "end_x": x,
        "end_y": y,
        "time_seconds": 10.0,
    }


def _pass(game, team, sx, sy, ex, ey, *, completed):
    return {
        "game_id": game,
        "period_id": 1,
        "team_id": team,
        "player_id": 99,
        "type_id": _PASS,
        "result_id": _SUCCESS if completed else _FAIL,
        "start_x": sx,
        "start_y": sy,
        "end_x": ex,
        "end_y": ey,
        "time_seconds": 20.0,
    }


def _actions(rows):
    df = pd.DataFrame(rows)
    df["action_id"] = range(len(df))
    return df


# Defender team 10, player 1: a rectangle hull x in [5,15], y in [20,48] (own half), centroid (10,34).
_HULL_CORNERS = [_def(1, 1, 10, 5, 20), _def(1, 1, 10, 15, 20), _def(1, 1, 10, 15, 48), _def(1, 1, 10, 5, 48)]


def test_conceded_prevented_exact():
    rows = [
        *_HULL_CORNERS,
        _pass(1, 20, 80, 40, 95, 40, completed=True),  # reflected (10,28) -> IN hull -> conceded
        _pass(1, 20, 80, 30, 98, 30, completed=False),  # reflected (7,38)  -> IN hull -> prevented
        _pass(1, 20, 40, 40, 50, 40, completed=True),  # reflected (55,28) -> OUT of hull -> ignored
    ]
    out, rep = compute_territorial_dominance(_actions(rows), xt=_toy_xt(0.1), params=_KEEP_ALL)
    r = out[out["player_id"] == 1].iloc[0]
    assert r["territory_xt_conceded"] == pytest.approx(0.1)
    assert r["territory_xt_prevented"] == pytest.approx(0.1)
    assert r["territory_xt_net"] == pytest.approx(0.0)
    assert r["territory_passes_into_hull"] == 2
    assert r["territory_hull_area_m2"] == pytest.approx(280.0)  # 10 x 28
    assert r["territory_hull_centroid_x"] == pytest.approx(10.0)
    assert r["territory_hull_source"] == "resolved"
    assert (rep.n_scored, rep.n_degenerate_hull, rep.n_players_in) == (1, 0, 1)


def test_failed_pass_end_is_death_location_undercount():
    # SHOULD-FIX 1: a failed pass AIMED into the territory but dying OUTSIDE the hull is NOT counted --
    # SPADL's `end` is the death location, not the intended target. This is the documented under-count.
    rows = [
        *_HULL_CORNERS,
        _pass(1, 20, 60, 40, 30, 40, completed=False),  # death end (30,40) -> reflected (75,28) OUT -> NOT prevented
    ]
    out, _ = compute_territorial_dominance(_actions(rows), xt=_toy_xt(0.1), params=_KEEP_ALL)
    r = out[out["player_id"] == 1].iloc[0]
    assert r["territory_xt_prevented"] == pytest.approx(0.0)
    assert r["territory_passes_into_hull"] == 0


def test_forward_flag_split():
    rows = [
        *_HULL_CORNERS,
        _pass(1, 20, 70, 40, 95, 40, completed=True),  # end_x 95 > start_x 70 -> forward, IN hull
    ]
    out, _ = compute_territorial_dominance(_actions(rows), xt=_toy_xt(0.1), params=_KEEP_ALL)
    r = out[out["player_id"] == 1].iloc[0]
    assert r["territory_xt_conceded"] == pytest.approx(0.1)
    assert r["territory_xt_conceded_forward"] == pytest.approx(0.1)


def test_rate_nan_on_zero_volume():
    out, _ = compute_territorial_dominance(_actions(_HULL_CORNERS), xt=_toy_xt(0.1), params=_KEEP_ALL)
    r = out.iloc[0]
    assert r["territory_passes_into_hull"] == 0
    assert pd.isna(r["territory_xt_conceded_rate"])


def test_degenerate_hull_row_kept_with_provenance():
    # 2 defensive actions -> no hull -> row kept with NaN metrics + hull_source='degenerate', counted.
    rows = [_def(1, 1, 10, 5, 20), _def(1, 1, 10, 15, 20)]
    out, rep = compute_territorial_dominance(_actions(rows), xt=_toy_xt(0.1), params=_KEEP_ALL)
    r = out[out["player_id"] == 1].iloc[0]
    assert r["territory_hull_source"] == "degenerate"
    assert pd.isna(r["territory_xt_conceded"])
    assert (rep.n_scored, rep.n_degenerate_hull, rep.n_players_in) == (0, 1, 1)


def test_injected_model_guard():
    acts = _actions(_HULL_CORNERS)
    with pytest.raises((ValueError, NotImplementedError)):
        compute_territorial_dominance(acts, xt=None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        compute_territorial_dominance(acts, xt="singh_counts")  # type: ignore[arg-type]


def test_method_family():
    acts = _actions(_HULL_CORNERS)
    with pytest.raises(NotImplementedError, match="counterfactual"):
        compute_territorial_dominance(acts, xt=_toy_xt(), method="counterfactual")
    with pytest.raises(ValueError, match="unknown method"):
        compute_territorial_dominance(acts, xt=_toy_xt(), method="bogus")


def test_mixed_dtype_player_id_does_not_fragment():
    # ADR-019: a defender whose player_id appears as int 1 and str "1" must NOT split into two rows.
    rows = [
        _def(1, 1, 10, 5, 20),
        _def(1, "1", 10, 15, 20),
        _def(1, 1, 10, 15, 48),
        _def(1, "1", 10, 5, 48),
    ]
    df = _actions(rows)
    df["player_id"] = df["player_id"].astype("object")
    out, _ = compute_territorial_dominance(df, xt=_toy_xt(0.1), params=_KEEP_ALL)
    assert len(out) == 1


def test_nan_team_pass_does_not_crash():
    # A pass carrying a NaN team_id must be EXCLUDED from the opponent set (not raise on the NA mask).
    rows = [
        *_HULL_CORNERS,
        _pass(1, 20, 80, 40, 95, 40, completed=True),  # legit opponent pass -> conceded 0.1
        _pass(1, np.nan, 80, 40, 95, 40, completed=True),  # NaN team -> excluded, must not crash
    ]
    df = _actions(rows)
    df["team_id"] = df["team_id"].astype("object")
    out, _ = compute_territorial_dominance(df, xt=_toy_xt(0.1), params=_KEEP_ALL)
    r = out[out["player_id"] == 1].iloc[0]
    assert r["territory_xt_conceded"] == pytest.approx(0.1)  # only the legit pass counted
    assert r["territory_passes_into_hull"] == 1


def test_window_aggregation_pools_and_rederives_hull():
    # PLAN-06: a 2-game window -> one per-player row, conceded pooled, hull re-derived over pooled actions.
    g1 = [*_HULL_CORNERS, _pass(1, 20, 80, 40, 95, 40, completed=True)]
    g2 = [
        _def(2, 1, 10, 5, 20),
        _def(2, 1, 10, 15, 20),
        _def(2, 1, 10, 15, 48),
        _def(2, 1, 10, 5, 48),
        _pass(2, 20, 80, 30, 98, 30, completed=True),
    ]
    out, _ = compute_territorial_dominance(_actions(g1 + g2), xt=_toy_xt(0.1), window=[1, 2], params=_KEEP_ALL)
    assert len(out) == 1  # per-player row
    r = out.iloc[0]
    assert pd.isna(r["game_id"])  # pooled grain
    assert r["territory_xt_conceded"] == pytest.approx(0.2)  # one completed into-hull pass per game
    # window=None returns per-(game, player) atoms unchanged (2 rows here).
    atoms, _ = compute_territorial_dominance(_actions(g1 + g2), xt=_toy_xt(0.1), params=_KEEP_ALL)
    assert len(atoms) == 2
