"""TF-54b Arm A: cf-actual threat suppression + the ADR-043 cache-collapse / mis-wire guards."""

import inspect

import numpy as np
import pytest

from silly_kicks.territorial_defense._arms import (
    arm_a_threat_suppressed,
    arm_a_threat_suppressed_batch,
    arm_b_threat_suppressed,
    contesting_defender_is_d,
    nearest_defender_pos,
)
from silly_kicks.territorial_defense._engine import remove_player_row

from ._fixtures import (
    ACTOR_PLAYER_ID,
    ATTACKING_TEAM_ID,
    DEFENDING_TEAM_ID,
    make_fitted_xt,
    make_rich_frame,
)


def _dpos(frame):
    return int(frame.index[frame["player_id"] == ACTOR_PLAYER_ID][0])


def test_arms_refuse_a_pitch_control_cache():
    """ADR-043 structural guard: a shared cache keys on frame identity (excludes positions), so it
    would serve the cf leg the factual surface -> silent 0. The parameter must never exist."""
    for fn in (arm_a_threat_suppressed, arm_a_threat_suppressed_batch, arm_b_threat_suppressed):
        assert "pitch_control_cache" not in inspect.signature(fn).parameters


def test_correct_path_nonzero_collapse_signature_and_miswire_raises():
    """SPEC-03, two-sided: (1) the CORRECT path (D removed) measurably differs from 0; (2) the
    collapse SIGNATURE -- two identical surfaces difference to exactly 0; (3) the arm REFUSES a
    mis-wire (cf == actual) with a raise, so the collapse can never surface as a silent 0."""
    from silly_kicks.tracking import SpearmanParams, compute_threat_pc, resolve_defended_goals

    actual = make_rich_frame()
    xt = make_fitted_xt()
    gm = resolve_defended_goals(actual)
    cf = remove_player_row(actual, player_pos=_dpos(actual))

    val = arm_a_threat_suppressed(actual, cf, attacking_team_id=ATTACKING_TEAM_ID, xt=xt, goal_map=gm)
    assert np.isfinite(val) and val > 0.0  # D's presence suppressed team-2 threat

    pcp = SpearmanParams(lambda_gk=3.0)
    t = compute_threat_pc(actual, attacking_team_id=ATTACKING_TEAM_ID, xt=xt, goal_map=gm, params=pcp)
    assert (t - t) == 0.0  # identical surfaces -> exactly 0 (the ADR-043 collapse shape)

    with pytest.raises(ValueError, match="exactly one row"):
        arm_a_threat_suppressed(actual, actual, attacking_team_id=ATTACKING_TEAM_ID, xt=xt, goal_map=gm)


def test_batch_matches_single():
    from silly_kicks.tracking import resolve_defended_goals

    actual = make_rich_frame()
    xt = make_fitted_xt()
    gm = resolve_defended_goals(actual)
    cf = remove_player_row(actual, player_pos=_dpos(actual))

    single = arm_a_threat_suppressed(actual, cf, attacking_team_id=ATTACKING_TEAM_ID, xt=xt, goal_map=gm)
    batch = arm_a_threat_suppressed_batch(actual, cf, attacking_team_id_by_frame=ATTACKING_TEAM_ID, xt=xt, goal_map=gm)
    assert len(batch) == 1
    assert abs(float(batch.iloc[0]) - single) < 1e-12


def test_arm_b_nearest_to_target_picks_and_removes_the_contesting_defender():
    from silly_kicks.tracking import resolve_defended_goals

    frame = make_rich_frame()
    xt = make_fitted_xt()
    gm = resolve_defended_goals(frame)
    target = (5.0, 34.0)  # deep centre, next to D (#102 at 4,34) -- D is the nearest team-1 defender
    d_idx = int(frame.index[frame["player_id"] == ACTOR_PLAYER_ID][0])
    assert nearest_defender_pos(frame, target, defending_team_id=DEFENDING_TEAM_ID) == d_idx

    val, removed = arm_b_threat_suppressed(
        frame,
        target_xy=target,
        defending_team_id=DEFENDING_TEAM_ID,
        attacking_team_id=ATTACKING_TEAM_ID,
        xt=xt,
        goal_map=gm,
    )
    assert removed == d_idx  # the contesting defender chosen by position IS D here
    assert val > 0.0  # removing the contesting defender raises team-2 threat


def test_arm_b_attribution_is_three_valued_measurable_and_honest_nan():
    # MEASURABLE both-sided on a FULL-TRACKING frame (no is_actor -> all ids real): nearest-to-target is
    # D for a deep-centre target (1.0), but a KNOWN non-D (#103 at 26,20) for a target beside it (0.0).
    ft = make_rich_frame().drop(columns=["is_actor"])
    assert (
        contesting_defender_is_d(ft, (5.0, 34.0), defending_team_id=DEFENDING_TEAM_ID, d_player_id=ACTOR_PLAYER_ID)
        == 1.0
    )
    assert (
        contesting_defender_is_d(ft, (26.0, 20.0), defending_team_id=DEFENDING_TEAM_ID, d_player_id=ACTOR_PLAYER_ID)
        == 0.0
    )
    # HONEST-NaN on an SB360 frame (has is_actor): the contesting defender #103 is a NON-actor, so its
    # snapshot-numbered id is anonymous -> un-measurable NaN, NEVER a fabricated 0.0 (ADR-027 / IMPL-01).
    sb = make_rich_frame()
    assert np.isnan(
        contesting_defender_is_d(sb, (26.0, 20.0), defending_team_id=DEFENDING_TEAM_ID, d_player_id=ACTOR_PLAYER_ID)
    )
    # ...but D itself, when it IS the actor (real id stamped by the bridge), stays measurable (1.0).
    assert (
        contesting_defender_is_d(sb, (5.0, 34.0), defending_team_id=DEFENDING_TEAM_ID, d_player_id=ACTOR_PLAYER_ID)
        == 1.0
    )
