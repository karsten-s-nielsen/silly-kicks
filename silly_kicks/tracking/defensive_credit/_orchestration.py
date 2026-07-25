"""Batch orchestration: actions -> long-form defensive-credit rows."""

from __future__ import annotations

import pandas as pd

from silly_kicks.id_compat import ids_differ
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking._action_orientation import acting_team_attacks_rtl
from silly_kicks.tracking.utils import link_actions_to_frames

from ._chaining import with_possessions
from ._line_break_signal import precompute_line_break_between_lines
from ._params import DEFENSIVE_CREDIT_RULES, RULE_FAILED_MARKING_THROUGH_BALL, DefensiveCreditParams
from ._rules import RULE_REGISTRY, CreditRow, RuleContext

_LONG_COLS = [
    "game_id",
    "period_id",
    "action_id",
    "player_id",
    "team_id",
    "rule",
    "signed_value",
    "anchor_type",
    "frame_id",
    "sizing",
    "resolution",
]

_SHOT_TYPE = spadlconfig.actiontype_id["shot"]
_GOAL_RESULT = spadlconfig.result_id["success"]


def _ensure_on_target(
    act: pd.DataFrame, frames: pd.DataFrame, pointers: pd.DataFrame, on_target_column: str
) -> pd.DataFrame:
    """Attach a nullable-boolean ``_on_target`` per action (P-1). Shots only; others NA.

    Goal (result success) -> True. Else the injected ``on_target_column`` if present, else the
    frame-based TF-48 ``shot_on_target_derived`` fallback (reuses the ONE link via ``links=pointers``).
    ``pd.NA`` stays NA (unknown) -> the pressure rules do not fire, so a saved shot is never mis-signed.
    """
    act = act.copy()
    if on_target_column in act.columns:
        base = pd.array(act[on_target_column], dtype="boolean")  # type: ignore[reportCallIssue, reportArgumentType]
    else:
        from silly_kicks.tracking.features import add_shot_goalmouth

        gm = add_shot_goalmouth(act, frames, links=pointers)
        col = (
            gm["shot_on_target_derived"]
            if "shot_on_target_derived" in gm.columns
            else pd.Series(pd.NA, index=act.index)
        )
        base = pd.array(col, dtype="boolean")  # type: ignore[reportCallIssue, reportArgumentType]
    is_shot = (act["type_id"] == _SHOT_TYPE).to_numpy()
    is_goal = ((act["type_id"] == _SHOT_TYPE) & (act["result_id"] == _GOAL_RESULT)).to_numpy()
    base[~is_shot] = pd.NA  # only meaningful for shots
    base[is_goal] = True  # a scored shot is on-target
    act["_on_target"] = base
    return act


def compute_defensive_credits(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    xg_column: str,
    xt,
    blocked_column: str = "shot_blocked",
    on_target_column: str = "shot_on_target_derived",
    links: pd.DataFrame | None = None,
    params: DefensiveCreditParams | None = None,
) -> pd.DataFrame:
    """Long-form: one row per (triggering action, credited player, rule). Pure.

    Examples
    --------
    Per-defender signed credit rows (needs a fitted ``ExpectedThreat`` and an injected per-shot xG)::

        credits = compute_defensive_credits(actions, frames, xg_column="xg", xt=xt)
        credits.groupby("player_id")["signed_value"].sum()  # per-player rollup
    """
    params = params or DefensiveCreditParams()
    act = with_possessions(actions).reset_index(drop=True)

    pointers = links if links is not None else link_actions_to_frames(act, frames)[0]
    act = _ensure_on_target(act, frames, pointers, on_target_column)  # P-1: attach _on_target (tri-state)
    fid_by_pos = (
        pointers.drop_duplicates("action_id")
        .set_index("action_id")["frame_id"]
        .reindex(act["action_id"].to_numpy())
        .to_numpy()
    )
    flip_series = acting_team_attacks_rtl(act, frames)  # ONE reprojection decision per action

    enabled = [r for r in DEFENSIVE_CREDIT_RULES if r in params.rules]
    # Item 3: precompute the between_lines line-break signal ONCE (threading the single link via
    # fid_by_pos); skip the Ward-clustering cost entirely when the rule is disabled (Q8).
    if RULE_FAILED_MARKING_THROUGH_BALL in enabled:
        line_break_between_lines = precompute_line_break_between_lines(
            act, frames, fid_by_pos=fid_by_pos, flip_by_pos=flip_series.to_numpy()
        )
    else:
        line_break_between_lines = pd.array([pd.NA] * len(act), dtype="boolean")
    rows: list[CreditRow] = []
    for idx in range(len(act)):
        fid = fid_by_pos[idx]
        fid = None if pd.isna(fid) else int(fid)
        ctx = RuleContext(
            actions=act,
            frames=frames,
            idx=idx,
            xg_column=xg_column,
            xt=xt,
            blocked_column=blocked_column,
            params=params,
            frame_id=fid,
            acting_team_id=act.iloc[idx]["team_id"],
            flip=bool(flip_series.iloc[idx]),
            line_break_between_lines=line_break_between_lines,
        )
        for rule_name in enabled:
            rows.extend(RULE_REGISTRY[rule_name](ctx))

    return _to_long_form(rows, act)


def _to_long_form(rows: list[CreditRow], act: pd.DataFrame) -> pd.DataFrame:
    if not rows:
        return _empty_long_form(act)
    df = pd.DataFrame([r.__dict__ for r in rows])  # 9 fields; period_id is NOT per-credit -> merge it
    pid = act[["action_id", "period_id"]].drop_duplicates("action_id")
    df = df.merge(pid, on="action_id", how="left")
    df = df[_LONG_COLS]  # reorder to the canonical schema (period_id now present via the merge)
    df["signed_value"] = df["signed_value"].astype("float64")
    df["frame_id"] = df["frame_id"].astype("Int64")
    df["period_id"] = df["period_id"].astype("int64")
    return df.reset_index(drop=True)


def _empty_long_form(act: pd.DataFrame) -> pd.DataFrame:
    empty = {c: pd.Series([], dtype="object") for c in _LONG_COLS}  # resolution stays object
    empty["signed_value"] = pd.Series([], dtype="float64")
    empty["frame_id"] = pd.Series([], dtype="Int64")
    empty["period_id"] = pd.Series([], dtype="int64")
    return pd.DataFrame(empty)


def _aggregate_defensive_credit(
    actions,
    frames,
    *,
    xg_column,
    xt,
    blocked_column="shot_blocked",
    on_target_column="shot_on_target_derived",
    links=None,
    params=None,
) -> pd.DataFrame:
    """actions + per-action aggregate columns (defending-team-scoped). Pure -- returns a NEW frame.

    No ``home_team_id`` (P-2): the defending/attacking split derives from ``team_id != acting-team``
    and reprojection uses ``acting_team_attacks_rtl``, so a home_team_id would be a dead required param.
    """
    long = compute_defensive_credits(
        actions,
        frames,
        xg_column=xg_column,
        xt=xt,
        blocked_column=blocked_column,
        on_target_column=on_target_column,
        links=links,
        params=params,
    )
    out = actions.copy()
    act_team = actions.set_index("action_id")["team_id"]
    if long.empty:
        defending = long
    else:
        long = long.copy()
        long["_acting_team"] = long["action_id"].map(act_team)
        keep = ids_differ(long["team_id"], long["_acting_team"])  # credited team != acting team -> defender
        defending = long[keep.to_numpy()]

    if defending.empty:
        net = plus = minus = pd.Series(dtype="float64")
        cnt = pd.Series(dtype="int64")
    else:
        net = defending.groupby("action_id")["signed_value"].sum(min_count=0)
        plus = defending[defending["signed_value"] > 0].groupby("action_id")["signed_value"].sum()
        minus = defending[defending["signed_value"] < 0].groupby("action_id")["signed_value"].sum()
        cnt = defending.groupby("action_id").size()

    aid = out["action_id"]
    out["defensive_credit_net"] = aid.map(net).fillna(0.0).astype("float64")
    out["defensive_credit_plus"] = aid.map(plus).fillna(0.0).astype("float64")
    out["defensive_credit_minus"] = aid.map(minus).fillna(0.0).astype("float64")
    out["n_defensive_credits"] = aid.map(cnt).fillna(0).astype("Int64")
    return out
