"""RuleContext + one pure function per rule + RULE_REGISTRY (TF-51).

See NOTICE for full bibliographic citations (Sumpter module 16.3; Bischofberger/Bauer/Baca
arXiv:2606.19931 for the xT(origin) turnover sizing).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking._action_orientation import acting_team_attacks_rtl

from ._chaining import recovery_after_pass, resulting_shot_in_possession, with_possessions
from ._params import (
    RULE_BEATEN_1V1,
    RULE_FAILED_CROSS_BLOCK,
    RULE_FAILED_MARKING_THROUGH_BALL,
    RULE_FAILED_PRESSURE_SHOT_ON_TARGET,
    RULE_FORCED_BAD_TOUCH,
    RULE_PRESSURE_ON_MISSED_SHOT,
    RULE_PRESSURE_PASS_FAIL,
    RULE_RECOVERY_DOUBLE_CREDIT,
    RULE_SHOT_BLOCK,
    RULE_SYNCHRONIZED_FINAL_THIRD_PRESSURE,
    SIZING_XG,
    SIZING_XT,
    DefensiveCreditParams,
)
from ._resolution import resolve_responsible_defenders
from ._sizing import extinguished_xt, xg_of_shot

_SHOT_TYPE = spadlconfig.actiontype_id["shot"]
_GOAL_RESULT = spadlconfig.result_id["success"]  # a scored shot is on-target by construction
_TAKE_ON = spadlconfig.actiontype_id["take_on"]
_CROSS = spadlconfig.actiontype_id["cross"]
_PASS = spadlconfig.actiontype_id["pass"]
_BAD_TOUCH = spadlconfig.actiontype_id["bad_touch"]
_SUCCESS = spadlconfig.result_id["success"]
_FAIL = spadlconfig.result_id["fail"]


def _is_failed_pass(a: pd.Series) -> bool:
    return a["type_id"] == _PASS and a["result_id"] == _FAIL


@dataclass
class CreditRow:
    game_id: object
    action_id: object
    player_id: object
    team_id: object
    rule: str
    signed_value: float
    anchor_type: str
    frame_id: object
    sizing: str


@dataclass
class RuleContext:
    actions: pd.DataFrame
    frames: pd.DataFrame
    idx: int
    xg_column: str
    xt: object
    blocked_column: str
    params: DefensiveCreditParams
    frame_id: int | None  # resolved linked frame for this action (None when unlinked)
    acting_team_id: object
    flip: bool  # precomputed action-LTR reprojection decision (ADR-028)

    @property
    def action(self) -> pd.Series:
        return self.actions.iloc[self.idx]

    def defenders(self, *, anchor_x, anchor_y, mode):
        return resolve_responsible_defenders(
            self.actions,
            self.frames,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            acting_team_id=self.acting_team_id,
            mode=mode,
            params=self.params,
            frame_id=self.frame_id,
            flip=self.flip,
        )

    @classmethod
    def build_single(cls, actions, frames, *, idx, xg_column, xt, blocked_column, params):
        """Convenience builder for unit tests: single action, single frame."""
        act = with_possessions(actions).reset_index(drop=True)
        fid = int(frames["frame_id"].iloc[0]) if "frame_id" in frames.columns and len(frames) else None
        flip = bool(acting_team_attacks_rtl(act, frames).iloc[idx])
        return cls(
            actions=act,
            frames=frames,
            idx=idx,
            xg_column=xg_column,
            xt=xt,
            blocked_column=blocked_column,
            params=params,
            frame_id=fid,
            acting_team_id=act.iloc[idx]["team_id"],
            flip=flip,
        )


def _is_true(val) -> bool:
    """Nullable-boolean -> plain bool: True only when definitively True (NA/False/absent -> False)."""
    if pd.isna(val):
        return False
    return bool(val)


def _is_blocked(ctx: RuleContext) -> bool:
    return _is_true(ctx.action.get(ctx.blocked_column, pd.NA))


def _on_target_state(ctx: RuleContext):
    """Tri-state on-target: True / False / None (unknown).

    A goal (result success) is on-target. Otherwise read the precomputed nullable-boolean
    ``_on_target`` column the ORCHESTRATOR attaches: provider outcome -> TF-48
    ``shot_on_target_derived`` fallback. Unknown (NA) -> None: the pressure rules DO NOT fire
    (we never fabricate a sign; a saved shot must not be mistaken for a miss -- P-1).
    """
    a = ctx.action
    if a["result_id"] == _GOAL_RESULT:
        return True
    val = a.get("_on_target", pd.NA)
    if pd.isna(val):
        return None
    return bool(val)


def _xt_at(ctx: RuleContext, x: float, y: float) -> float:
    return float(extinguished_xt([(x, y)], ctx.xt)[0])


def _shot_credit(ctx: RuleContext, *, rule: str, sign: float, mode: str) -> list[CreditRow]:
    a = ctx.action
    defs = ctx.defenders(anchor_x=a["start_x"], anchor_y=a["start_y"], mode=mode)
    if defs.empty:
        return []
    xg = xg_of_shot(a, xg_column=ctx.xg_column)
    rows = []
    for _, d in defs.iterrows():
        rows.append(
            CreditRow(
                game_id=a["game_id"],
                action_id=a["action_id"],
                player_id=d["player_id"],
                team_id=d["team_id"],
                rule=rule,
                signed_value=sign * xg,
                anchor_type="shot",
                frame_id=ctx.frame_id,
                sizing=SIZING_XG,
            )
        )
    return rows


# --- shot rules (mutually-exclusive partition of shot outcomes) ---
def rule_pressure_on_missed_shot(ctx: RuleContext) -> list[CreditRow]:
    if _is_blocked(ctx):
        return []
    if _on_target_state(ctx) is not False:  # fires ONLY when definitively OFF-target
        return []
    return _shot_credit(ctx, rule=RULE_PRESSURE_ON_MISSED_SHOT, sign=+1.0, mode="nearest")


def rule_failed_pressure_shot_on_target(ctx: RuleContext) -> list[CreditRow]:
    if _is_blocked(ctx):
        return []
    if _on_target_state(ctx) is not True:  # fires ONLY when definitively ON-target
        return []
    return _shot_credit(ctx, rule=RULE_FAILED_PRESSURE_SHOT_ON_TARGET, sign=-1.0, mode="nearest")


def rule_shot_block(ctx: RuleContext) -> list[CreditRow]:
    if not _is_blocked(ctx):
        return []
    return _shot_credit(ctx, rule=RULE_SHOT_BLOCK, sign=+1.0, mode="nearest")


# --- turnover rules (xT(origin)-sized) ---
def rule_pressure_pass_fail(ctx: RuleContext) -> list[CreditRow]:
    a = ctx.action
    if not _is_failed_pass(a):  # anchor: failed pass (spec section 5)
        return []
    defs = ctx.defenders(anchor_x=a["start_x"], anchor_y=a["start_y"], mode="nearest")
    if defs.empty:
        return []
    val = _xt_at(ctx, a["start_x"], a["start_y"])  # xT(origin) -- same point for both rows
    d = defs.iloc[0]
    return [
        CreditRow(
            a["game_id"],
            a["action_id"],
            d["player_id"],
            d["team_id"],
            RULE_PRESSURE_PASS_FAIL,
            +val,
            "pass",
            ctx.frame_id,
            SIZING_XT,
        ),
        CreditRow(
            a["game_id"],
            a["action_id"],
            a["player_id"],
            a["team_id"],
            RULE_PRESSURE_PASS_FAIL,
            -val,
            "pass",
            ctx.frame_id,
            SIZING_XT,
        ),
    ]


def rule_forced_bad_touch(ctx: RuleContext) -> list[CreditRow]:
    a = ctx.action
    if a["type_id"] != _BAD_TOUCH:  # anchor: bad_touch (spec section 5)
        return []
    defs = ctx.defenders(anchor_x=a["start_x"], anchor_y=a["start_y"], mode="nearest")
    if defs.empty:
        return []
    val = _xt_at(ctx, a["start_x"], a["start_y"])
    d = defs.iloc[0]
    return [
        CreditRow(
            a["game_id"],
            a["action_id"],
            d["player_id"],
            d["team_id"],
            RULE_FORCED_BAD_TOUCH,
            +val,
            "bad_touch",
            ctx.frame_id,
            SIZING_XT,
        )
    ]


def rule_synchronized_final_third_pressure(ctx: RuleContext) -> list[CreditRow]:
    a = ctx.action
    if not _is_failed_pass(a):  # anchor: failed pass (spec section 5)
        return []
    if a["start_x"] > ctx.params.synchronized_zone_boundary_x:  # not in carrier's own defensive third
        return []
    defs = ctx.defenders(anchor_x=a["start_x"], anchor_y=a["start_y"], mode="all_within_beyond_nearest")
    if defs.empty:
        return []
    val = _xt_at(ctx, a["start_x"], a["start_y"])
    return [
        CreditRow(
            a["game_id"],
            a["action_id"],
            d["player_id"],
            d["team_id"],
            RULE_SYNCHRONIZED_FINAL_THIRD_PRESSURE,
            +val,
            "pass",
            ctx.frame_id,
            SIZING_XT,
        )
        for _, d in defs.iterrows()
    ]


def rule_recovery_double_credit(ctx: RuleContext) -> list[CreditRow]:
    a = ctx.action
    if not _is_failed_pass(a):  # anchor: failed pass -> own recovery (spec section 5)
        return []
    rec = recovery_after_pass(
        ctx.actions, ctx.idx, max_actions=ctx.params.recovery_max_actions
    )  # single resolver (P-3)
    if rec is None:
        return []
    passer_val = _xt_at(ctx, a["start_x"], a["start_y"])  # -passer at the passer origin
    rec_val = _xt_at(ctx, float(rec["start_x"]), float(rec["start_y"]))  # +recoverer at the recovery location
    return [
        CreditRow(
            a["game_id"],
            a["action_id"],
            rec["player_id"],
            rec["team_id"],
            RULE_RECOVERY_DOUBLE_CREDIT,
            +rec_val,
            "pass",
            ctx.frame_id,
            SIZING_XT,
        ),
        CreditRow(
            a["game_id"],
            a["action_id"],
            a["player_id"],
            a["team_id"],
            RULE_RECOVERY_DOUBLE_CREDIT,
            -passer_val,
            "pass",
            ctx.frame_id,
            SIZING_XT,
        ),
    ]


# --- chained rules (resulting-shot xG-sized) ---
def _resulting_shot(ctx: RuleContext):
    return resulting_shot_in_possession(
        ctx.actions,
        ctx.idx,
        attacking_team_id=ctx.action["team_id"],
        max_actions=ctx.params.resulting_shot_max_actions,
    )


def rule_beaten_1v1(ctx: RuleContext) -> list[CreditRow]:
    a = ctx.action
    if a["type_id"] != _TAKE_ON or a["result_id"] != _SUCCESS:
        return []
    shot = _resulting_shot(ctx)
    if shot is None:
        return []
    xg = xg_of_shot(shot, xg_column=ctx.xg_column)
    if not (xg >= ctx.params.beaten_1v1_min_shot_xg):  # NaN-safe: NaN fails the floor -> no row
        return []
    defs = ctx.defenders(anchor_x=a["start_x"], anchor_y=a["start_y"], mode="nearest")
    if defs.empty:
        return []
    d = defs.iloc[0]
    return [
        CreditRow(
            a["game_id"],
            a["action_id"],
            d["player_id"],
            d["team_id"],
            RULE_BEATEN_1V1,
            -xg,
            "take_on",
            ctx.frame_id,
            SIZING_XG,
        )
    ]


def rule_failed_cross_block(ctx: RuleContext) -> list[CreditRow]:
    a = ctx.action
    if a["type_id"] != _CROSS or a["result_id"] != _SUCCESS:
        return []
    shot = _resulting_shot(ctx)
    if shot is None:
        return []
    xg = xg_of_shot(shot, xg_column=ctx.xg_column)
    rows: list[CreditRow] = []
    # -def at the receipt point (cross end)
    defs = ctx.defenders(anchor_x=a["end_x"], anchor_y=a["end_y"], mode="nearest")
    if not defs.empty:
        d = defs.iloc[0]
        rows.append(
            CreditRow(
                a["game_id"],
                a["action_id"],
                d["player_id"],
                d["team_id"],
                RULE_FAILED_CROSS_BLOCK,
                -xg,
                "cross",
                ctx.frame_id,
                SIZING_XG,
            )
        )
    # +blocker if the resulting shot was blocked (nearest opp to the shot origin)
    if _is_true(shot.get(ctx.blocked_column, pd.NA)):
        bdefs = ctx.defenders(anchor_x=shot["start_x"], anchor_y=shot["start_y"], mode="nearest")
        if not bdefs.empty:
            b = bdefs.iloc[0]
            rows.append(
                CreditRow(
                    a["game_id"],
                    a["action_id"],
                    b["player_id"],
                    b["team_id"],
                    RULE_FAILED_CROSS_BLOCK,
                    +xg,
                    "cross",
                    ctx.frame_id,
                    SIZING_XG,
                )
            )
    return rows


def rule_failed_marking_through_ball(ctx: RuleContext) -> list[CreditRow]:
    a = ctx.action
    if a["type_id"] != _PASS or a["result_id"] != _SUCCESS:
        return []
    dxt = _xt_at(ctx, a["end_x"], a["end_y"]) - _xt_at(ctx, a["start_x"], a["start_y"])
    if not (dxt >= ctx.params.through_ball_delta_xt_min):  # NaN-safe floor
        return []
    shot = _resulting_shot(ctx)
    if shot is None:
        return []
    xg = xg_of_shot(shot, xg_column=ctx.xg_column)
    defs = ctx.defenders(anchor_x=a["start_x"], anchor_y=a["start_y"], mode="nearest")
    if defs.empty:
        return []
    d = defs.iloc[0]
    return [
        CreditRow(
            a["game_id"],
            a["action_id"],
            d["player_id"],
            d["team_id"],
            RULE_FAILED_MARKING_THROUGH_BALL,
            -xg,
            "pass",
            ctx.frame_id,
            SIZING_XG,
        )
    ]


RULE_REGISTRY: dict[str, Callable[[RuleContext], list[CreditRow]]] = {
    RULE_PRESSURE_ON_MISSED_SHOT: rule_pressure_on_missed_shot,
    RULE_FAILED_PRESSURE_SHOT_ON_TARGET: rule_failed_pressure_shot_on_target,
    RULE_SHOT_BLOCK: rule_shot_block,
    RULE_PRESSURE_PASS_FAIL: rule_pressure_pass_fail,
    RULE_RECOVERY_DOUBLE_CREDIT: rule_recovery_double_credit,
    RULE_SYNCHRONIZED_FINAL_THIRD_PRESSURE: rule_synchronized_final_third_pressure,
    RULE_FORCED_BAD_TOUCH: rule_forced_bad_touch,
    RULE_FAILED_CROSS_BLOCK: rule_failed_cross_block,
    RULE_FAILED_MARKING_THROUGH_BALL: rule_failed_marking_through_ball,
    RULE_BEATEN_1V1: rule_beaten_1v1,
}
