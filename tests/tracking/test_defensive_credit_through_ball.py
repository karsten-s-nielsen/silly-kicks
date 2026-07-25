"""Item 3 line-break-gated through-ball behavioural tests (TF-51 v2, spec section 5).

Exercises rule_failed_marking_through_ball through the full orchestrator (real precompute + gating),
both directions plus the genuine short-circuit / unlinked conditions (Q3 -- BUILD the real state,
never force the signal, so a precompute bug that writes True onto a non-candidate is caught).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking.defensive_credit import DefensiveCreditParams, compute_defensive_credits
from tests.tracking._defensive_credit_fixtures import one_action
from tests.tracking.test_defensive_line import _make_frame_rows

_THROUGH = frozenset({"failed_marking_through_ball"})


def _scene(*, away_xs, away_ys, pass_start, pass_end, period_id=1, frame_period=1):
    frames = _make_frame_rows(
        home_outfield_xs=[8.0, 12.0, 16.0, 20.0, 24.0],
        home_outfield_ys=[10.0, 24.0, 34.0, 44.0, 58.0],
        away_outfield_xs=away_xs,
        away_outfield_ys=away_ys,
        frame_id=500,
        time_seconds=1.0,
        period_id=frame_period,
    )
    pass_a = one_action(
        action_id=1,
        type_name="pass",
        result_name="success",
        start_x=pass_start[0],
        start_y=pass_start[1],
        end_x=pass_end[0],
        end_y=pass_end[1],
        team_id=1,
        player_id=200,
        time_seconds=1.0,
        game_id=1,
        period_id=period_id,
    )
    shot_a = one_action(
        action_id=2,
        type_name="shot",
        result_name="fail",
        start_x=pass_end[0],
        start_y=pass_end[1],
        end_x=105.0,
        end_y=34.0,
        team_id=1,
        player_id=201,
        time_seconds=1.4,
        game_id=1,
        period_id=period_id,
    )
    actions = pd.concat([pass_a, shot_a], ignore_index=True)
    actions["shot_blocked"] = pd.array([pd.NA, False], dtype="boolean")
    actions["cross_blocked"] = pd.array([pd.NA, pd.NA], dtype="boolean")
    actions["shot_on_target_derived"] = pd.array([pd.NA, False], dtype="boolean")
    actions["xg"] = [np.nan, 0.2]
    return actions, frames


def _run(actions, frames, fitted_xt):
    return compute_defensive_credits(
        actions, frames, xg_column="xg", xt=fitted_xt, params=DefensiveCreditParams(rules=_THROUGH)
    )


def _fired(out) -> bool:
    return bool((out["rule"] == "failed_marking_through_ball").any())


def test_through_ball_fires_on_between_lines_break(fitted_xt):
    # A pass that threads between adjacent same-line defenders across 3 lines -> between_lines -> fires.
    actions, frames = _scene(
        away_xs=[50.0, 50.0, 50.0, 70.0, 70.0, 70.0, 90.0, 90.0, 90.0, 90.0],
        away_ys=[15.0, 34.0, 53.0, 15.0, 34.0, 53.0, 10.0, 24.0, 44.0, 58.0],
        pass_start=(48.0, 34.0),
        pass_end=(95.0, 34.0),
    )
    assert _fired(_run(actions, frames, fitted_xt))


def test_through_ball_does_not_fire_on_progressive_non_linebreak(fitted_xt):
    # A high-ΔxT forward pass (v1 WOULD fire on the ΔxT gate) that does NOT break a line between-lines:
    # a defensive line behind the origin (not crossed) + a lone defender ahead (around_line only).
    actions, frames = _scene(
        away_xs=[20.0, 20.0, 20.0, 61.0],
        away_ys=[20.0, 34.0, 48.0, 34.0],
        pass_start=(60.0, 34.0),
        pass_end=(100.0, 34.0),
    )
    assert not _fired(_run(actions, frames, fitted_xt))


def test_through_ball_no_fire_on_genuine_short_circuit(fitted_xt):
    # Q3: BUILD the real condition -- fewer than min_opponents (3) outfielders -> genuine
    # short-circuit-0 -> no fire (a real precompute must NOT write True onto this row).
    actions, frames = _scene(
        away_xs=[61.0, 70.0],
        away_ys=[34.0, 34.0],
        pass_start=(60.0, 34.0),
        pass_end=(100.0, 34.0),
    )
    assert not _fired(_run(actions, frames, fitted_xt))


def test_through_ball_no_fire_on_unlinked_action(fitted_xt):
    # Q3: the between_lines geometry WOULD fire if linked, but the action is in a period with no
    # frames -> genuine <NA> -> no fire (proves the precompute leaves unlinked rows NA, not True).
    actions, frames = _scene(
        away_xs=[50.0, 50.0, 50.0, 70.0, 70.0, 70.0, 90.0, 90.0, 90.0, 90.0],
        away_ys=[15.0, 34.0, 53.0, 15.0, 34.0, 53.0, 10.0, 24.0, 44.0, 58.0],
        pass_start=(48.0, 34.0),
        pass_end=(95.0, 34.0),
        period_id=2,
        frame_period=1,
    )
    assert not _fired(_run(actions, frames, fitted_xt))
