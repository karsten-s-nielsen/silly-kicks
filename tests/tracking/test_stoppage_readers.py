"""The 10 ``ball_state`` readers fail closed without the stoppage column (TF-46 Part 2 spec section 27.19).

Owner decisions 2026-10-26 (R-C2): on a provider without an observed dead-ball state (SkillCorner, Gradient Sports,
Metrica) every reader requires the ``stoppage_state`` column of ``derive_stoppage_state``; it raises without it, for
an unclassified provider, for frames without ``source_provider``, and for a column of another rule version.
Mutant for every case: a gate that fails open (returns False) makes each test fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.tracking import STOPPAGE_RULE_VERSION, STOPPAGE_SOURCES, derive_stoppage_state


def _frames(provider: str | None = "skillcorner") -> pd.DataFrame:
    rows = []
    for f in range(20):
        t = round(f * 0.1, 3)
        common = dict(game_id=1, period_id=1, frame_id=f, time_seconds=t, frame_rate=10.0, speed=1.0, vx=1.0, vy=0.0,
                      z=0.0, speed_source="native", ball_state="alive", visibility=True)  # fmt: skip
        rows.append({**common, "player_id": pd.NA, "team_id": pd.NA, "is_ball": True, "is_goalkeeper": False,
                     "x": 40.0, "y": 30.0, "team_attacking_direction": pd.NA})  # fmt: skip
        for pid, team, x, gk, d in ((11, 1, 39.0, False, "ltr"), (12, 1, 5.0, True, "ltr"),
                                    (21, 2, 45.0, False, "rtl"), (22, 2, 100.0, True, "rtl")):  # fmt: skip
            rows.append({**common, "player_id": pid, "team_id": team, "is_ball": False, "is_goalkeeper": gk,
                         "x": x, "y": 30.0, "team_attacking_direction": d})  # fmt: skip
    out = pd.DataFrame(rows)
    out["player_id"] = out["player_id"].astype("Int64")
    out["team_id"] = out["team_id"].astype("Int64")
    if provider is not None:
        out["source_provider"] = provider
    return out


def _actions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [1, 1],
            "period_id": [1, 1],
            "action_id": [1, 2],
            "time_seconds": [0.5, 1.5],
            "team_id": [1, 1],
            "player_id": [11, 11],
            "type_id": [spadlconfig.actiontype_id["pass"], spadlconfig.actiontype_id["throw_in"]],
            "result_id": [spadlconfig.result_id["success"]] * 2,
            "bodypart_id": [spadlconfig.bodypart_id["foot"]] * 2,
            "start_x": [40.0, 40.0],
            "start_y": [30.0, 0.0],
            "end_x": [50.0, 45.0],
            "end_y": [30.0, 10.0],
            "action_provenance": ["native", "derived"],
            "gi_interruption_after": pd.NA,  # stoppage rule version 3 (fix (A)): no penalty marker
        }
    )


def _xshot(frames, actions):
    from silly_kicks.tracking._xshot_occurrence import prepare_xshot_training_data

    return prepare_xshot_training_data(frames, actions.iloc[0:0], home_team_id=1)


def _xcross(frames, actions):
    from silly_kicks.tracking._xcross_attempt import prepare_xcross_training_data

    return prepare_xcross_training_data(frames, actions, home_team_id=1)


def _causal(frames, actions):
    from silly_kicks.causal.opportunities import build_opportunities

    meta = {"cross_types": ["cross"], "carrier_params": {"tolerance_m": 3.0, "beta": 0.0, "gamma": 0.25}}
    return build_opportunities(frames, actions, home_team_id=1, model_metadata=meta)


def _gkdv(frames, actions):
    from silly_kicks.gkdv._engine import _apply_domain

    return _apply_domain(frames, carrier=None, goal_map=None, params=None)  # type: ignore[arg-type]


def _positioning(frames, actions):
    from silly_kicks.positioning import compute_positioning_gap

    return compute_positioning_gap(frames, xt=None)


def _rd_counterfactual(frames, actions):
    from silly_kicks.restdefense._counterfactual import _apply_domain

    return _apply_domain(frames, carrier=None, goal_map=None, params=None, which="keeper")  # type: ignore[arg-type]


def _rd_samples(frames, actions):
    from silly_kicks.restdefense._windows import select_rest_defense_samples

    return select_rest_defense_samples(actions, frames, goal_map=None)


def _elastic(frames, actions):
    from silly_kicks.tracking._elastic_sync import align_events_to_frames

    return align_events_to_frames(actions, frames)


def _off_ball_runs(frames, actions):
    from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

    return _off_ball_runs_kernel(actions, frames)


def _run_values(frames, actions):
    from silly_kicks.tracking._run_values import detect_off_ball_runs

    return detect_off_ball_runs(actions, frames)


READERS = {
    "prepare_xshot_training_data": _xshot,
    "prepare_xcross_training_data": _xcross,
    "build_opportunities": _causal,
    "gkdv_apply_domain": _gkdv,
    "compute_positioning_gap": _positioning,
    "restdefense_apply_domain": _rd_counterfactual,
    "select_rest_defense_samples": _rd_samples,
    "align_events_to_frames": _elastic,
    "off_ball_runs": _off_ball_runs,
    "detect_off_ball_runs": _run_values,
}


@pytest.mark.parametrize("reader", sorted(READERS))
def test_a_reader_raises_on_skillcorner_without_the_column(reader):
    with pytest.raises(ValueError, match="derive_stoppage_state"):
        READERS[reader](_frames("skillcorner"), _actions())


@pytest.mark.parametrize("reader", sorted(READERS))
def test_a_reader_raises_on_an_unclassified_provider(reader):
    with pytest.raises(ValueError, match="unclassified provider"):
        READERS[reader](_frames("synthetic"), _actions())


@pytest.mark.parametrize("reader", sorted(READERS))
def test_a_reader_raises_without_source_provider(reader):
    with pytest.raises(ValueError, match="source_provider"):
        READERS[reader](_frames(None), _actions())


@pytest.mark.parametrize("reader", sorted(READERS))
def test_a_reader_raises_on_an_older_rule_version(reader):
    built = derive_stoppage_state(_frames("skillcorner"), actions=_actions())
    built["stoppage_source"] = (
        built["stoppage_source"].astype(object).replace(STOPPAGE_SOURCES[1], f"events_v{STOPPAGE_RULE_VERSION - 1}")
    )
    with pytest.raises(ValueError, match="not a current rule"):
        READERS[reader](built, _actions())


def test_the_fixture_is_full_rate_and_reaches_the_gate():
    # Precondition (ADR-056): the fixture passes derive_stoppage_state (full rate, one provider) so the raises above
    # come from the reader gate, not from the fixture.
    built = derive_stoppage_state(_frames("skillcorner"), actions=_actions())
    assert set(built["stoppage_state"].astype(object)) <= {"live", "unknown"}
    assert np.isin(built["stoppage_source"].astype(object), [STOPPAGE_SOURCES[1]]).all()
