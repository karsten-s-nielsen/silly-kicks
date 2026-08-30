"""Action-grid sampling for rest defense (TF-60, ADR-080).

Rest defense is a property of the IN-POSSESSION team's play, so the sampling grain is that team's
on-ball actions (spec §5): each action carries a freeze-frame (SB360) or links to a tracking frame.
This module turns ``(actions, frames)`` into one row per input action, tagging each with its
possession, its linked frame, the ball's frame-x, the team's own/attacked goal ends (from the
``GoalMap``), a moment-of-loss flag, and -- for actions that fail a gate -- a drop reason. Dropping
is drop-AND-COUNT (every action gets a disposition), so the orchestrator's report conserves.

All orientation comes from the ``GoalMap`` (ADR-055), never team identity; ids compare via
``id_compat`` (ADR-019). Pure: the caller's ``actions``/``frames`` are never mutated.
"""

from __future__ import annotations

import pandas as pd

from silly_kicks.id_compat import align_join_keys, canonical_id_series, ids_equal
from silly_kicks.spadl import add_possessions
from silly_kicks.tracking import link_actions_to_frames

from ._columns import RD_FRAME_KEYS, RD_SAMPLE_KEYS
from ._config import RestDefenseParams

# Drop reasons, in the precedence they are applied (a closed vocabulary; a token that VARIES, never
# a constant column -- the das_source idiom).
_NOT_IN_POSSESSION = "not_in_possession"
_STRIDE_SKIP = "stride_skip"
_UNLINKED = "unlinked"
_DEAD_BALL = "dead_ball"
_GOAL_UNRESOLVED = "goal_end_unresolved"
_NOT_COMMITTED = "not_committed_forward"

#: Closed vocabulary of ``gate_drop_reason`` values (scored rows carry ``pd.NA``).
GATE_DROP_REASONS = (
    _NOT_IN_POSSESSION,
    _STRIDE_SKIP,
    _UNLINKED,
    _DEAD_BALL,
    _GOAL_UNRESOLVED,
    _NOT_COMMITTED,
)

_LINK_COLS = ["action_id", "frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]

_OUTPUT_COLS = [
    *RD_SAMPLE_KEYS,
    "possession_id",
    "frame_id",
    "ball_x",
    "own_goal_x",
    "attacked_goal_x",
    "is_possession_loss",
    "gate_drop_reason",
]

#: Shared frozen default (RestDefenseParams is immutable, so one singleton is safe; avoids a B008
#: function-call-in-default).
_DEFAULT_PARAMS = RestDefenseParams()


def _possession_owner(team_ids: pd.Series):
    """Modal (most-frequent) non-NA team id of a possession -- the team that owns the ball."""
    present = team_ids.dropna()
    if present.empty:
        return pd.NA
    modes = present.mode()
    return modes.iloc[0] if not modes.empty else pd.NA


def select_rest_defense_samples(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    goal_map,
    params: RestDefenseParams = _DEFAULT_PARAMS,
    links: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per input action, tagged with its rest-defense window disposition (see module docstring).

    Columns: ``game_id, period_id, team_id, action_id`` (``team_id`` is the actor's team, which is the
    in-possession team A on scored rows), ``possession_id``, ``frame_id``, ``ball_x`` (the ball's x in
    the linked frame), ``own_goal_x`` / ``attacked_goal_x`` (from ``goal_map``), ``is_possession_loss``
    (bool), and ``gate_drop_reason`` (``pd.NA`` on scored rows; else a :data:`GATE_DROP_REASONS` token).
    """
    acts = add_possessions(actions).reset_index(drop=True)

    # --- team in possession (the possession owner) + moment-of-loss flag ------------------------
    owner_by_poss = acts.groupby("possession_id")["team_id"].agg(_possession_owner)
    owner = acts["possession_id"].map(owner_by_poss)
    in_possession = ids_equal(acts["team_id"], owner)

    is_loss = pd.Series(False, index=acts.index)
    inposs_order = acts[in_possession.to_numpy()].sort_values(["possession_id", "time_seconds", "action_id"])
    if not inposs_order.empty:
        terminal_idx = inposs_order.groupby("possession_id").tail(1).index
        is_loss.loc[terminal_idx] = True

    # --- stride subsampling (cost control): keep every Nth in-possession action, but always keep
    #     the possession's terminal loss snapshot ------------------------------------------------
    stride_skip = pd.Series(False, index=acts.index)
    if params.possession_stride > 1 and not inposs_order.empty:
        ordinal = inposs_order.groupby("possession_id").cumcount()
        skip_idx = inposs_order.index[(ordinal % params.possession_stride) != 0]
        stride_skip.loc[skip_idx] = True
        stride_skip &= ~is_loss

    # --- link each action to its frame ----------------------------------------------------------
    if links is None:
        links = link_actions_to_frames(acts, frames, on_low_coverage="ignore")[0]
    la, lr = align_join_keys(acts, links[_LINK_COLS], ["action_id"])
    acts = la.merge(lr, on="action_id", how="left").reset_index(drop=True)
    # merge re-orders nothing (how="left" preserves left order) but recompute the row-aligned masks
    in_possession = in_possession.to_numpy()
    is_loss = is_loss.to_numpy()
    stride_skip = stride_skip.to_numpy()

    # --- ball x + ball_state from the linked frame ----------------------------------------------
    ball = (
        frames.loc[frames["is_ball"], ["game_id", "period_id", "frame_id", "x", "ball_state"]]
        .rename(columns={"x": "ball_x"})
        .drop_duplicates(subset=RD_FRAME_KEYS, keep="first")
    )
    la, rb = align_join_keys(acts, ball, RD_FRAME_KEYS)
    acts = la.merge(rb, on=RD_FRAME_KEYS, how="left").reset_index(drop=True)

    # --- own / attacked goal ends (GoalMap, built once per match; ADR-055) ----------------------
    g = canonical_id_series(acts["game_id"])
    p = canonical_id_series(acts["period_id"])
    t = canonical_id_series(acts["team_id"])
    ckeys = list(zip(g, p, t, strict=True))
    own_map: dict = {}
    att_map: dict = {}
    for key in set(ckeys):
        cg, cp, ct = key
        own_map[key] = goal_map.get(cg, cp, ct, allow_guess=True)
        att_map[key] = goal_map.attacked_goal(cg, cp, ct, allow_guess=True)
    acts["own_goal_x"] = pd.Series([own_map[k] for k in ckeys], index=acts.index, dtype="float64")
    acts["attacked_goal_x"] = pd.Series([att_map[k] for k in ckeys], index=acts.index, dtype="float64")

    # --- gate: assign the first applicable drop reason (precedence) ------------------------------
    reason = pd.Series(pd.NA, index=acts.index, dtype="object")
    ball_state = acts["ball_state"] if "ball_state" in acts.columns else pd.Series(pd.NA, index=acts.index)
    advance = (acts["ball_x"] - acts["own_goal_x"]).abs()
    ordered = [
        (~pd.Series(in_possession, index=acts.index), _NOT_IN_POSSESSION),
        (pd.Series(stride_skip, index=acts.index), _STRIDE_SKIP),
        (acts["frame_id"].isna(), _UNLINKED),
        (ball_state.eq("dead"), _DEAD_BALL),
        (acts["own_goal_x"].isna(), _GOAL_UNRESOLVED),
        (~(advance >= params.min_ball_advance_m).fillna(False), _NOT_COMMITTED),
    ]
    for mask, token in ordered:
        cond = reason.isna() & mask.fillna(False).to_numpy()
        reason = reason.mask(cond, token)

    acts["is_possession_loss"] = pd.Series(is_loss, index=acts.index).astype(bool)
    acts["gate_drop_reason"] = reason
    return acts[_OUTPUT_COLS].reset_index(drop=True)
