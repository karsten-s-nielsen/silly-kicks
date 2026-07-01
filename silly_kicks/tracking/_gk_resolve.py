"""Frame-based defending-GK resolution (TF-13).

Resolves the defending team's goalkeeper player_id from tracking frames
for every action. Standalone composable utility -- callers use for fillna
on events-based defending_gk_player_id or as direct lookup.

See spec: docs/superpowers/specs/2026-05-04-tf13-tf14-defensive-line-design.md s2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._id_compat import ids_match
from .utils import link_actions_to_frames


def _gk_from_frames_linked(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    same_team: bool,
    tolerance_seconds: float,
) -> pd.Series:
    """Shared body of the two public resolvers: per-action GK player_id from the LINKED frame.

    Links each action to the nearest frame (within tolerance) and returns the ``is_goalkeeper`` row
    whose team matches (``same_team=True``, the ACTING team's GK) or differs (``same_team=False``, the
    OPPOSING/defending GK) from the action's team. Deterministic lowest-player_id tiebreak; output cast
    to frames' ``player_id`` dtype. ``same_team=False`` reproduces the original
    ``defending_gk_from_frames`` byte-for-byte (only the team predicate is parameterized)."""
    pid_dtype = frames["player_id"].dtype

    n = len(actions)
    out = pd.Series(np.full(n, np.nan), index=actions.index, dtype="object")

    if n == 0 or len(frames) == 0:
        return out

    pointers, _report = link_actions_to_frames(actions, frames, tolerance_seconds=tolerance_seconds)

    gk_rows = frames[(frames["is_goalkeeper"] == True) & (~frames["is_ball"])].copy()  # noqa: E712
    if gk_rows.empty:
        return out

    ptr = pointers.merge(
        actions[["action_id", "team_id", "period_id"]],
        on="action_id",
        how="left",
    )
    linked = ptr[ptr["frame_id"].notna()].copy()
    if linked.empty:
        return out

    linked["frame_id_int"] = linked["frame_id"].astype("int64")
    gk_in_frame = linked.merge(
        gk_rows[["period_id", "frame_id", "team_id", "player_id"]].rename(
            columns={"team_id": "gk_team_id", "player_id": "gk_player_id"}
        ),
        left_on=["period_id", "frame_id_int"],
        right_on=["period_id", "frame_id"],
        how="inner",
    )

    # Team predicate: acting team (==) vs opposing team (!=). NaN action team_id -> comparison False -> dropped.
    match_team = gk_in_frame["gk_team_id"] == gk_in_frame["team_id"]
    picked = gk_in_frame[match_team if same_team else ~match_team]

    if picked.empty:
        return out

    # Deterministic tiebreak: lowest player_id per action
    best = picked.sort_values("gk_player_id").drop_duplicates("action_id", keep="first")

    action_to_idx = pd.Series(actions.index, index=actions["action_id"].to_numpy())
    for _, row in best.iterrows():
        aid = row["action_id"]
        if aid in action_to_idx.index:
            out.loc[action_to_idx.loc[aid]] = row["gk_player_id"]

    # Cast to match frames dtype if numeric
    if pid_dtype == np.dtype("int64") or str(pid_dtype) == "Int64":
        out = pd.to_numeric(out, errors="coerce")
        if str(pid_dtype) == "Int64":
            out = out.astype("Int64")

    return out


def defending_gk_from_frames(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    tolerance_seconds: float = 0.2,
) -> pd.Series:
    """Per-action defending-GK player_id resolved from tracking frames.

    For each action, links to the nearest frame (within tolerance), finds
    the opposing team's is_goalkeeper=True row, and returns that player_id.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions with action_id, period_id, time_seconds, team_id.
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS shape).
    tolerance_seconds : float, default 0.2
        Maximum |time_offset| for a valid link.

    Returns
    -------
    pd.Series
        Aligned with actions.index. dtype matches frames' player_id dtype
        (object for kloppy/sportec, int64/Int64 for Gradient Sports).
        NaN where action couldn't link, no opposing-team GK in linked frame,
        or action.team_id is NaN.

    Examples
    --------
    Fill NaN from events-based GK resolution::

        from silly_kicks.tracking.features import defending_gk_from_frames
        gk_series = defending_gk_from_frames(actions, frames)
        actions["defending_gk_player_id"] = (
            actions["defending_gk_player_id"].fillna(gk_series)
        )

    See NOTICE for full bibliographic citations.
    """
    return _gk_from_frames_linked(actions, frames, same_team=False, tolerance_seconds=tolerance_seconds)


def acting_gk_from_frames(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    tolerance_seconds: float = 0.2,
) -> pd.Series:
    """Per-action ACTING-team GK player_id resolved from tracking frames (mirror of
    :func:`defending_gk_from_frames` with the team predicate inverted, CR 2026-07-01).

    For each action, returns the acting team's goalkeeper ``player_id``. Unlike the pure per-frame link
    (which returns NaN whenever the keeper is not detected in the linked frame), this adds an **identity
    fallback**: the acting team's GK is resolved from the roster-stable ``is_goalkeeper`` identity for
    that ``(game_id, team_id)`` even when the keeper is undetected at the linked frame — essential for
    goal-kicks on broadcast tracking, where the keeper is missing at ~40% of event frames. When a
    ``(game, team)`` has more than one ``is_goalkeeper`` identity (a keeper substitution), the one whose
    frames are **nearest-in-time** to the action is chosen.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions with action_id, period_id, time_seconds, team_id (game_id used when present).
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS shape).
    tolerance_seconds : float, default 0.2
        Maximum |time_offset| for the per-frame link (the identity fallback is time-tolerance-free).

    Returns
    -------
    pd.Series
        Aligned with actions.index; dtype matches frames' player_id dtype. NaN only where the acting
        team has no ``is_goalkeeper`` identity anywhere in the frames, or ``team_id`` is NaN.

    Notes
    -----
    Pure resolver — it never mutates ``actions``. Deciding WHEN to apply it (e.g. overriding a
    goal-kick's NULL taker with the keeper) is the consumer's synthesis step, not this function's.

    See NOTICE for full bibliographic citations.
    """
    out = _gk_from_frames_linked(actions, frames, same_team=True, tolerance_seconds=tolerance_seconds)
    need = out.isna().to_numpy()
    if not need.any() or len(frames) == 0:
        return out

    gk_rows = frames[(frames["is_goalkeeper"] == True) & (~frames["is_ball"])]  # noqa: E712
    if gk_rows.empty:
        return out

    use_game = "game_id" in actions.columns and "game_id" in gk_rows.columns
    # game arrays are dummy (empty) when use_game is False -- only read inside the ``if use_game`` guard,
    # but kept as ndarrays (not None) so the game filter stays statically subscriptable.
    gk_team = gk_rows["team_id"].to_numpy()
    gk_game = gk_rows["game_id"].to_numpy() if use_game else np.zeros(len(gk_rows))
    gk_time = gk_rows["time_seconds"].to_numpy(float)
    gk_pid = gk_rows["player_id"].to_numpy()

    act_team = actions["team_id"].to_numpy()
    act_game = actions["game_id"].to_numpy() if use_game else np.zeros(len(actions))
    act_time = actions["time_seconds"].to_numpy(float)

    # Series views for ids_match, built ONCE (not per loop iteration).
    gk_team_s = pd.Series(gk_team)
    gk_game_s = pd.Series(gk_game)

    for i in np.where(need)[0]:
        t = act_team[i]
        if pd.isna(t):
            continue  # NaN team -> unresolvable (stays NaN)
        sel = ids_match(gk_team_s, t).to_numpy()
        if use_game:
            sel = sel & ids_match(gk_game_s, act_game[i]).to_numpy()
        if not sel.any():
            continue  # no acting-team GK identity anywhere
        cand_pid = gk_pid[sel]
        distinct = pd.unique(cand_pid)
        if len(distinct) == 1:
            out.iloc[i] = distinct[0]
        else:  # GK sub: nearest-in-time identity
            cand_time = gk_time[sel]
            out.iloc[i] = cand_pid[np.abs(cand_time - act_time[i]).argmin()]

    return out


def defended_goal_x(frames: pd.DataFrame) -> dict:
    """(game_id, period_id, team_id) -> defended goal_x (0 or 105).

    N1: GK identification quality is provider-variable (Metrica/SkillCorner were
    21-50% pre-fix). Prefer mean GK x; fall back to the team's mean outfield x
    when a (game, period, team) has no GK rows, so a mis-/missing-GK does not
    silently drop the team from the goal map.

    Extracted byte-identically from ``_xshot_occurrence._defended_goal_x``
    (TF-48, spec 2026-06-10-shot-goalmouth-psxg-design); xS re-imports via shim.

    Examples
    --------
    Resolve each team's defended goal end per (game, period)::

        from silly_kicks.tracking._gk_resolve import defended_goal_x
        goal_map = defended_goal_x(frames)
        # goal_map[(game_id, period_id, team_id)] in (0.0, 105.0)
    """
    players = frames[~frames["is_ball"].astype(bool)]
    out: dict = {}
    for key, grp in players.groupby(["game_id", "period_id", "team_id"], dropna=False):
        gk_rows = grp[grp["is_goalkeeper"].astype(bool)]
        ref = gk_rows if len(gk_rows) else grp  # fallback: whole-team mean-x
        out[key] = 0.0 if float(ref["x"].mean()) < 52.5 else 105.0
    return out
