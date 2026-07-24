import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.defensive_credit._params import DefensiveCreditParams
from silly_kicks.tracking.defensive_credit._rules import RULE_REGISTRY, RuleContext
from tests.tracking._defensive_credit_fixtures import frame_with_defender, one_action


def _shot_ctx(
    *, result_name, shot_blocked: object = pd.NA, xg=0.2, defender_x=96.0, fitted_xt=None, on_target: object = pd.NA
):
    """on_target: nullable-boolean the orchestrator normally computes (goal / provider / TF-48).
    Set it explicitly in unit tests since synthetic frames can't derive shot_on_target_derived."""
    actions = one_action(type_name="shot", result_name=result_name, start_x=95.0, start_y=34.0)
    actions["shot_blocked"] = pd.array([shot_blocked], dtype="boolean")  # type: ignore[reportCallIssue, reportArgumentType]
    actions["_on_target"] = pd.array([on_target], dtype="boolean")  # type: ignore[reportCallIssue, reportArgumentType]
    actions["xg"] = [xg]
    frames = frame_with_defender(defender_x=defender_x, defender_y=34.0)
    return RuleContext.build_single(
        actions,
        frames,
        idx=0,
        xg_column="xg",
        xt=fitted_xt,
        blocked_column="shot_blocked",
        params=DefensiveCreditParams(),
    )


def test_pressure_on_missed_shot_credits_nearest_defender(fitted_xt):
    # OFF-target (a genuine miss): result fail, not blocked, on_target definitively False -> +credit
    ctx = _shot_ctx(result_name="fail", shot_blocked=False, xg=0.2, on_target=False, fitted_xt=fitted_xt)
    rows = RULE_REGISTRY["pressure_on_missed_shot"](ctx)
    assert len(rows) == 1
    assert rows[0].rule == "pressure_on_missed_shot"
    assert rows[0].signed_value == pytest.approx(0.2)  # +xG, off-target
    assert rows[0].player_id == 900
    assert rows[0].sizing == "xg"


def test_failed_pressure_shot_on_target_debits_defender_on_goal(fitted_xt):
    # a GOAL is on-target (result success -> on_target True by construction)
    ctx = _shot_ctx(result_name="success", shot_blocked=False, xg=0.3, on_target=True, fitted_xt=fitted_xt)
    rows = RULE_REGISTRY["failed_pressure_shot_on_target"](ctx)
    assert len(rows) == 1
    assert rows[0].signed_value == pytest.approx(-0.3)  # -xG, on-target


def test_pressured_saved_shot_is_negative(fitted_xt):
    # THE P-1 regression: a SAVED shot is result=fail but ON-target -> failed_pressure (NEGATIVE),
    # NOT pressure_on_missed_shot (+credit). SPADL result can't tell saved from off-target; _on_target does.
    ctx = _shot_ctx(result_name="fail", shot_blocked=False, xg=0.3, on_target=True, fitted_xt=fitted_xt)
    assert RULE_REGISTRY["failed_pressure_shot_on_target"](ctx)[0].signed_value == pytest.approx(-0.3)
    assert RULE_REGISTRY["pressure_on_missed_shot"](ctx) == []  # must NOT +credit a saved shot


def test_unknown_on_target_fires_neither_pressure_rule(fitted_xt):
    # on_target unknown (NA) -> we do NOT fabricate a sign; neither pressure rule fires (no row).
    ctx = _shot_ctx(result_name="fail", shot_blocked=False, xg=0.2, on_target=pd.NA, fitted_xt=fitted_xt)
    assert RULE_REGISTRY["pressure_on_missed_shot"](ctx) == []
    assert RULE_REGISTRY["failed_pressure_shot_on_target"](ctx) == []


def test_shot_block_credits_the_blocker(fitted_xt):
    ctx = _shot_ctx(result_name="fail", shot_blocked=True, xg=0.25, on_target=pd.NA, fitted_xt=fitted_xt)
    rows = RULE_REGISTRY["shot_block"](ctx)
    assert len(rows) == 1
    assert rows[0].rule == "shot_block"
    assert rows[0].signed_value == pytest.approx(0.25)  # +xG to the blocker


def test_shot_rules_are_mutually_exclusive(fitted_xt):
    # a blocked shot fires ONLY shot_block, not the two pressure rules (blocked precedence, on_target moot)
    ctx = _shot_ctx(result_name="fail", shot_blocked=True, xg=0.25, on_target=True, fitted_xt=fitted_xt)
    assert RULE_REGISTRY["shot_block"](ctx)  # fires
    assert RULE_REGISTRY["pressure_on_missed_shot"](ctx) == []  # blocked precedence
    assert RULE_REGISTRY["failed_pressure_shot_on_target"](ctx) == []


def test_shot_rule_no_defender_no_row(fitted_xt):
    ctx = _shot_ctx(
        result_name="fail", shot_blocked=False, xg=0.2, defender_x=80.0, on_target=False, fitted_xt=fitted_xt
    )
    assert RULE_REGISTRY["pressure_on_missed_shot"](ctx) == []  # defender too far


def test_shot_rule_nan_xg_fires_but_unsizable(fitted_xt):
    ctx = _shot_ctx(result_name="fail", shot_blocked=False, xg=np.nan, on_target=False, fitted_xt=fitted_xt)
    rows = RULE_REGISTRY["pressure_on_missed_shot"](ctx)
    assert len(rows) == 1 and np.isnan(rows[0].signed_value)  # fired-but-unsizable


# --- Task 6: turnover rules ---
def _pass_ctx(
    *,
    result_name="fail",
    start_x=40.0,
    start_y=34.0,
    defender_x=41.0,
    team_id=10,
    player_id=5,
    fitted_xt=None,
    action_id=1,
):
    actions = one_action(
        type_name="pass",
        result_name=result_name,
        start_x=start_x,
        start_y=start_y,
        end_x=55.0,
        end_y=34.0,
        team_id=team_id,
        player_id=player_id,
        action_id=action_id,
    )
    frames = frame_with_defender(defender_x=defender_x, defender_y=start_y)
    return RuleContext.build_single(
        actions,
        frames,
        idx=0,
        xg_column="xg",
        xt=fitted_xt,
        blocked_column="shot_blocked",
        params=DefensiveCreditParams(),
    )


def test_pressure_pass_fail_emits_plus_presser_minus_passer(fitted_xt):
    ctx = _pass_ctx(start_x=40.0, defender_x=41.0, fitted_xt=fitted_xt)
    rows = RULE_REGISTRY["pressure_pass_fail"](ctx)
    assert len(rows) == 2
    plus = next(r for r in rows if r.signed_value > 0)
    minus = next(r for r in rows if r.signed_value < 0)
    assert plus.player_id == 900 and plus.team_id == 20  # presser (defender)
    assert minus.player_id == 5 and minus.team_id == 10  # passer (acting team)
    assert plus.signed_value == pytest.approx(-minus.signed_value)  # same origin -> equal magnitude
    assert plus.sizing == "xt"


def test_pressure_pass_fail_no_defender_no_rows(fitted_xt):
    ctx = _pass_ctx(start_x=40.0, defender_x=60.0, fitted_xt=fitted_xt)  # far
    assert RULE_REGISTRY["pressure_pass_fail"](ctx) == []


def test_forced_bad_touch_credits_presser(fitted_xt):
    actions = one_action(type_name="bad_touch", result_name="fail", start_x=45.0, start_y=34.0, team_id=10, player_id=5)
    frames = frame_with_defender(defender_x=46.0, defender_y=34.0)
    ctx = RuleContext.build_single(
        actions,
        frames,
        idx=0,
        xg_column="xg",
        xt=fitted_xt,
        blocked_column="shot_blocked",
        params=DefensiveCreditParams(),
    )
    rows = RULE_REGISTRY["forced_bad_touch"](ctx)
    assert len(rows) == 1 and rows[0].signed_value > 0 and rows[0].sizing == "xt"


def test_synchronized_fires_only_in_own_defensive_third(fitted_xt):
    # carrier's own defensive third = action-LTR x <= 35. Two defenders within threshold;
    # synchronized credits the one BEYOND nearest.
    actions = one_action(type_name="pass", result_name="fail", start_x=20.0, start_y=34.0, team_id=10, player_id=5)
    frames = frame_with_defender(defender_x=21.0, defender_y=34.0)  # 1 m -> 900
    extra = frames.iloc[[0]].copy()
    extra["player_id"] = 901
    extra["x"] = 22.0
    frames = pd.concat([frames, extra], ignore_index=True)
    ctx = RuleContext.build_single(
        actions,
        frames,
        idx=0,
        xg_column="xg",
        xt=fitted_xt,
        blocked_column="shot_blocked",
        params=DefensiveCreditParams(),
    )
    rows = RULE_REGISTRY["synchronized_final_third_pressure"](ctx)
    assert {r.player_id for r in rows} == {901}  # beyond-nearest


def test_synchronized_silent_outside_defensive_third(fitted_xt):
    actions = one_action(type_name="pass", result_name="fail", start_x=70.0, start_y=34.0, team_id=10, player_id=5)
    frames = frame_with_defender(defender_x=71.0, defender_y=34.0)
    extra = frames.iloc[[0]].copy()
    extra["player_id"] = 901
    extra["x"] = 72.0
    frames = pd.concat([frames, extra], ignore_index=True)
    ctx = RuleContext.build_single(
        actions,
        frames,
        idx=0,
        xg_column="xg",
        xt=fitted_xt,
        blocked_column="shot_blocked",
        params=DefensiveCreditParams(),
    )
    assert RULE_REGISTRY["synchronized_final_third_pressure"](ctx) == []


# --- Task 7: chained rules ---
def _stream_ctx(rows, *, idx, fitted_xt, frames):
    from silly_kicks.spadl import config as spadlconfig

    base: dict[str, object] = dict(game_id="g1", period_id=1, bodypart_id=spadlconfig.bodypart_id["foot"])
    recs = []
    for i, r in enumerate(rows):
        d: dict[str, object] = dict(base)
        d.update(action_id=i, time_seconds=float(i), **r)
        d["type_id"] = spadlconfig.actiontype_id[str(d.pop("type_name"))]
        d["result_id"] = spadlconfig.result_id[str(d.pop("result_name"))]
        d.setdefault("shot_blocked", pd.NA)
        d.setdefault("xg", np.nan)
        recs.append(d)
    actions = pd.DataFrame(recs)
    actions["shot_blocked"] = pd.array(actions["shot_blocked"].tolist(), dtype="boolean")
    return RuleContext.build_single(
        actions,
        frames,
        idx=idx,
        xg_column="xg",
        xt=fitted_xt,
        blocked_column="shot_blocked",
        params=DefensiveCreditParams(),
    )


def test_beaten_1v1_debits_defender_on_quality_resulting_shot(fitted_xt):
    frames = frame_with_defender(defender_x=51.0, defender_y=34.0, action_time=0.0)
    ctx = _stream_ctx(
        [
            dict(
                type_name="take_on",
                result_name="success",
                team_id=10,
                player_id=5,
                start_x=50.0,
                start_y=34.0,
                end_x=55.0,
                end_y=34.0,
            ),
            dict(
                type_name="shot",
                result_name="fail",
                team_id=10,
                player_id=6,
                start_x=95.0,
                start_y=34.0,
                end_x=105.0,
                end_y=34.0,
                xg=0.2,
            ),
        ],
        idx=0,
        fitted_xt=fitted_xt,
        frames=frames,
    )
    rows = RULE_REGISTRY["beaten_1v1"](ctx)
    assert len(rows) == 1 and rows[0].signed_value == pytest.approx(-0.2) and rows[0].team_id == 20


def test_beaten_1v1_no_quality_shot_no_row(fitted_xt):
    frames = frame_with_defender(defender_x=51.0, defender_y=34.0, action_time=0.0)
    ctx = _stream_ctx(
        [
            dict(
                type_name="take_on",
                result_name="success",
                team_id=10,
                player_id=5,
                start_x=50.0,
                start_y=34.0,
                end_x=55.0,
                end_y=34.0,
            ),
            dict(
                type_name="shot",
                result_name="fail",
                team_id=10,
                player_id=6,
                start_x=95.0,
                start_y=34.0,
                end_x=105.0,
                end_y=34.0,
                xg=0.01,
            ),  # below 0.05 floor
        ],
        idx=0,
        fitted_xt=fitted_xt,
        frames=frames,
    )
    assert RULE_REGISTRY["beaten_1v1"](ctx) == []


def test_failed_cross_block_pair(fitted_xt):
    frames = frame_with_defender(defender_x=101.0, defender_y=40.0, action_time=0.0)
    ctx = _stream_ctx(
        [
            dict(
                type_name="cross",
                result_name="success",
                team_id=10,
                player_id=5,
                start_x=90.0,
                start_y=5.0,
                end_x=100.0,
                end_y=40.0,
            ),
            dict(
                type_name="shot",
                result_name="fail",
                team_id=10,
                player_id=6,
                start_x=100.0,
                start_y=40.0,
                end_x=105.0,
                end_y=34.0,
                xg=0.3,
                shot_blocked=True,
            ),
        ],
        idx=0,
        fitted_xt=fitted_xt,
        frames=frames,
    )
    rows = RULE_REGISTRY["failed_cross_block"](ctx)
    signs = sorted(r.signed_value for r in rows)
    assert signs == pytest.approx([-0.3, 0.3])  # -def at receipt, +blocker


def test_failed_marking_through_ball(fitted_xt):
    frames = frame_with_defender(defender_x=61.0, defender_y=34.0, action_time=0.0)
    ctx = _stream_ctx(
        [
            dict(
                type_name="pass",
                result_name="success",
                team_id=10,
                player_id=5,
                start_x=60.0,
                start_y=34.0,
                end_x=95.0,
                end_y=34.0,
            ),  # big forward ΔxT
            dict(
                type_name="shot",
                result_name="fail",
                team_id=10,
                player_id=6,
                start_x=95.0,
                start_y=34.0,
                end_x=105.0,
                end_y=34.0,
                xg=0.2,
            ),
        ],
        idx=0,
        fitted_xt=fitted_xt,
        frames=frames,
    )
    rows = RULE_REGISTRY["failed_marking_through_ball"](ctx)
    assert len(rows) == 1 and rows[0].signed_value == pytest.approx(-0.2) and rows[0].team_id == 20


def test_registry_covers_every_rule():
    from silly_kicks.tracking.defensive_credit._params import DEFENSIVE_CREDIT_RULES

    assert set(RULE_REGISTRY) == set(DEFENSIVE_CREDIT_RULES)
