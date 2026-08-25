"""Implements the label tranformers of the VAEP framework."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadl
from silly_kicks.id_compat import ids_differ, ids_equal, ids_match, same_id

# Single-source goal/own-goal predicates (ADR-0NN). Extracted so the definition lives in ONE place
# instead of being copy-pasted across the ~8 label functions (the copy-paste that hid the own-goal
# undercount). Kept on `type_name` to preserve the label input contract, with an explicit shot-type
# name-set rather than a fragile ``str.contains("shot")`` substring.
_SHOT_TYPE_NAMES = frozenset({"shot", "shot_penalty", "shot_freekick"})


def _is_goal(actions: pd.DataFrame) -> pd.Series:
    """A goal is a successful shot-class action (shot / shot_penalty / shot_freekick)."""
    return actions["type_name"].isin(_SHOT_TYPE_NAMES) & (actions["result_id"] == spadl.result_id["success"])


def _is_owngoal(actions: pd.DataFrame) -> pd.Series:
    """An own goal is unambiguous by RESULT (own goals are ``bad_touch``, not shots) — NO type gate.

    This is the fix for the codebase-wide undercount: own goals (every converter emits them as
    ``bad_touch``) never matched the former shot-type-substring gate, so they never registered in
    scores/concedes/xG labels.
    """
    return actions["result_id"] == spadl.result_id["owngoal"]


def _warn_if_nr_actions_ignored(nr_actions: int, window: str) -> None:
    if nr_actions != 10 and window != "action":
        warnings.warn(
            f"nr_actions={nr_actions} is ignored when window={window!r}; only window='action' uses nr_actions",
            UserWarning,
            stacklevel=3,
        )


def _require_column(actions: pd.DataFrame, col: str, window: str) -> None:
    if col not in actions.columns:
        raise ValueError(
            f"window={window!r} requires a '{col}' column. "
            + (
                "Call add_possessions() first."
                if col == "possession_id"
                else f"Ensure '{col}' is present in the actions DataFrame."
            )
        )


def scores(
    actions: pd.DataFrame,
    nr_actions: int = 10,
    xg_column: str | None = None,
    *,
    window: Literal["action", "possession", "time"] = "action",
    window_seconds: float = 15.0,
) -> pd.DataFrame:
    """Determine whether the team possessing the ball scored a goal within the next window.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    nr_actions : int, default=10  # noqa: DAR103
        Number of actions after the current action to consider.
        Only used when ``window="action"``.
    xg_column : str or None, default=None  # noqa: DAR103
        If provided, return xG-weighted scoring probability instead of boolean labels.
    window : {"action", "possession", "time"}, default="action"
        Lookahead window mode. ``"action"`` uses ``nr_actions`` (original VAEP).
        ``"possession"`` looks within the same ``possession_id`` chain (requires
        column; call ``add_possessions()`` first — use default params for DTAI-naive,
        or ``merge_brief_opposing_actions=2, brief_window_seconds=2.0,
        defensive_transition_types=("interception", "clearance")`` for DTAI-extended).
        ``"time"`` looks within ``window_seconds`` of the current action's
        ``time_seconds``, bounded by ``period_id``.
    window_seconds : float, default=15.0
        Time window in seconds. Only used when ``window="time"``.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'scores' and a row for each action set to
        True if a goal was scored by the team possessing the ball within the
        next window; otherwise False. If xg_column is provided, the column
        contains the maximum xG value instead of a boolean.

    Examples
    --------
    Compute "scores" labels for VAEP training::

        from silly_kicks.spadl import add_names
        from silly_kicks.vaep.labels import scores

        actions_with_names = add_names(actions)
        y_scores = scores(actions_with_names, nr_actions=10)

    Possession-based windowing::

        from silly_kicks.spadl import add_possessions
        actions = add_possessions(actions)
        y_scores = scores(actions, window="possession")

    Time-based windowing (15-second lookahead)::

        y_scores = scores(actions, window="time", window_seconds=15.0)
    """
    _warn_if_nr_actions_ignored(nr_actions, window)

    if window == "action":
        if xg_column is not None:
            return _scores_xg(actions, nr_actions, xg_column)
        return _scores_action(actions, nr_actions)
    elif window == "possession":
        _require_column(actions, "possession_id", window)
        return _scores_possession(actions, xg_column)
    elif window == "time":
        _require_column(actions, "time_seconds", window)
        return _scores_time(actions, window_seconds, xg_column)
    else:
        raise ValueError(f"Unknown window mode: {window!r}")


def _same_team_scalar(a, b) -> bool:
    """Scalar id equality for the groupby loops. NA is never equal to anything."""
    return same_id(a, b)


def _other_team_scalar(a, b) -> bool:
    """Scalar OPPONENT test: both ids present AND different -- the scalar sibling of `ids_differ`.

    `not same_id(a, b)` is NOT this. `same_id` is False when either id is NA, so negating it
    promotes every NULL-team row to "opponent", which is the silent defect this fix exists to
    remove: in a concedes label that charges an unknown-team row with the other side's goal.

    Deliberately a module-local helper rather than a new `id_compat` export: ADR-027 fixed the
    identical shape in `_line_breaking.py` with `same_id` plus an explicit `pd.isna` route at the
    call site, and following that precedent keeps the public id surface unchanged.
    """
    return not (pd.isna(a) or pd.isna(b)) and not same_id(a, b)


def _scores_action(actions: pd.DataFrame, nr_actions: int) -> pd.DataFrame:
    """Original VAEP action-count windowed scoring labels."""
    goal = _is_goal(actions)
    owngoal = _is_owngoal(actions)
    team_id = actions["team_id"]

    result = goal.copy()
    for i in range(1, nr_actions):
        shifted_goal = goal.shift(-i, fill_value=False)
        shifted_owngoal = owngoal.shift(-i, fill_value=False)
        shifted_team = team_id.shift(-i)
        same_team = pd.Series(ids_equal(team_id, shifted_team).to_numpy(), index=team_id.index)
        other_team = pd.Series(ids_differ(team_id, shifted_team).to_numpy(), index=team_id.index)
        result = result | (shifted_goal & same_team) | (shifted_owngoal & other_team)

    return pd.DataFrame(result, columns=["scores"])


def concedes(
    actions: pd.DataFrame,
    nr_actions: int = 10,
    xg_column: str | None = None,
    *,
    window: Literal["action", "possession", "time"] = "action",
    window_seconds: float = 15.0,
) -> pd.DataFrame:
    """Determine whether the team possessing the ball conceded a goal within the next window.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    nr_actions : int, default=10  # noqa: DAR103
        Number of actions after the current action to consider.
        Only used when ``window="action"``.
    xg_column : str or None, default=None  # noqa: DAR103
        If provided, return xG-weighted conceding probability.
    window : {"action", "possession", "time"}, default="action"
        Lookahead window mode. See :func:`scores` for details.
    window_seconds : float, default=15.0
        Time window in seconds. Only used when ``window="time"``.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'concedes' and a row for each action set to
        True if a goal was conceded by the team possessing the ball within the
        next window; otherwise False. If xg_column is provided, the column
        contains the maximum xG value instead of a boolean.

    Examples
    --------
    Compute "concedes" labels (the dual of ``scores``) for VAEP training::

        from silly_kicks.spadl import add_names
        from silly_kicks.vaep.labels import concedes

        actions_with_names = add_names(actions)
        y_concedes = concedes(actions_with_names, nr_actions=10)
        # y_concedes["concedes"] is bool: True iff the team in possession
        # concedes within the next 10 actions.
    """
    _warn_if_nr_actions_ignored(nr_actions, window)

    if window == "action":
        if xg_column is not None:
            return _concedes_xg(actions, nr_actions, xg_column)
        return _concedes_action(actions, nr_actions)
    elif window == "possession":
        _require_column(actions, "possession_id", window)
        return _concedes_possession(actions, xg_column)
    elif window == "time":
        _require_column(actions, "time_seconds", window)
        return _concedes_time(actions, window_seconds, xg_column)
    else:
        raise ValueError(f"Unknown window mode: {window!r}")


def _concedes_action(actions: pd.DataFrame, nr_actions: int) -> pd.DataFrame:
    """Original VAEP action-count windowed conceding labels."""
    goal = _is_goal(actions)
    owngoal = _is_owngoal(actions)
    team_id = actions["team_id"]

    result = owngoal.copy()
    for i in range(1, nr_actions):
        shifted_goal = goal.shift(-i, fill_value=False)
        shifted_owngoal = owngoal.shift(-i, fill_value=False)
        shifted_team = team_id.shift(-i)
        same_team = pd.Series(ids_equal(team_id, shifted_team).to_numpy(), index=team_id.index)
        other_team = pd.Series(ids_differ(team_id, shifted_team).to_numpy(), index=team_id.index)
        result = result | (shifted_goal & other_team) | (shifted_owngoal & same_team)

    return pd.DataFrame(result, columns=["concedes"])


def _scores_xg(actions: pd.DataFrame, nr_actions: int, xg_column: str) -> pd.DataFrame:
    """Compute xG-weighted scoring labels using shift-based vectorization."""
    goal = _is_goal(actions)
    owngoal = _is_owngoal(actions)
    xg = actions.get(xg_column, pd.Series(0.0, index=actions.index)).fillna(0.0)  # type: ignore[reportOptionalMemberAccess]
    team_id = actions["team_id"]

    result = pd.Series(0.0, index=actions.index)
    for i in range(1, nr_actions):
        shifted_goal = goal.shift(-i, fill_value=False)
        shifted_owngoal = owngoal.shift(-i, fill_value=False)
        shifted_xg = xg.shift(-i).fillna(0.0)
        shifted_team = team_id.shift(-i)
        same_team = pd.Series(ids_equal(team_id, shifted_team).to_numpy(), index=team_id.index)
        other_team = pd.Series(ids_differ(team_id, shifted_team).to_numpy(), index=team_id.index)
        score_xg = shifted_xg.where(shifted_goal & same_team, 0.0)
        owngoal_xg = shifted_xg.where(shifted_owngoal & other_team, 0.0)
        result = result.combine(score_xg + owngoal_xg, max, fill_value=0.0)  # type: ignore[reportArgumentType]
    return pd.DataFrame({"scores": result})


def _concedes_xg(actions: pd.DataFrame, nr_actions: int, xg_column: str) -> pd.DataFrame:
    """Compute xG-weighted conceding labels using shift-based vectorization."""
    goal = _is_goal(actions)
    owngoal = _is_owngoal(actions)
    xg = actions.get(xg_column, pd.Series(0.0, index=actions.index)).fillna(0.0)  # type: ignore[reportOptionalMemberAccess]
    team_id = actions["team_id"]

    result = pd.Series(0.0, index=actions.index)
    for i in range(1, nr_actions):
        shifted_goal = goal.shift(-i, fill_value=False)
        shifted_owngoal = owngoal.shift(-i, fill_value=False)
        shifted_xg = xg.shift(-i).fillna(0.0)
        shifted_team = team_id.shift(-i)
        same_team = pd.Series(ids_equal(team_id, shifted_team).to_numpy(), index=team_id.index)
        other_team = pd.Series(ids_differ(team_id, shifted_team).to_numpy(), index=team_id.index)
        concede_xg = shifted_xg.where(shifted_goal & other_team, 0.0)
        owngoal_xg = shifted_xg.where(shifted_owngoal & same_team, 0.0)
        result = result.combine(concede_xg + owngoal_xg, max, fill_value=0.0)  # type: ignore[reportArgumentType]
    return pd.DataFrame({"concedes": result})


def _suffix_after(ev: np.ndarray) -> np.ndarray:
    """``rev[i]`` = aggregate of ``ev[i+1:]`` (max for float, OR for bool); 0/False past the end.

    ``np.maximum.accumulate`` is OR on bool and max on float, so one implementation serves both the
    boolean (any downstream eligible event) and the xG (max downstream eligible xG) label paths.
    """
    rev = np.zeros_like(ev)
    if len(ev) > 1:
        suffix = np.maximum.accumulate(ev[::-1])[::-1]  # suffix[i] = agg(ev[i:])
        rev[:-1] = suffix[1:]  # rev[i] = agg(ev[i+1:])  (strictly downstream)
    return rev


def _possession_labels(
    actions: pd.DataFrame,
    xg_column: str | None,
    *,
    col_name: str,
    same_event: np.ndarray,
    other_event: np.ndarray,
    self_event: np.ndarray,
) -> pd.DataFrame:
    """Vectorized possession-chain windowed label (ADR-068; replaces the O(k^2) scalar-``.loc`` loop).

    A position labels 1 iff a downstream ``same_event`` by its OWN team, or a downstream
    ``other_event`` by the opposing team, occurs within the same possession -- plus a ``self_event``
    at the position itself. **Team-aware:** a possession holds at most two teams (carve-outs can add a
    second to a mostly-one-team chain), so per team present we take the reverse-cumulative aggregate
    of that team's eligible downstream events and index each position by its OWN team. Semantics are
    byte-identical to the prior nested loop: ``same_id`` for same-team (``ids_match``), both-present-
    and-different for opponent (``pd.notna & ~ids_match``); xG path takes the MAX over eligible
    downstream events (the old loop's no-break ``max``), boolean path the OR.
    """
    team = actions["team_id"].to_numpy()
    group_cols = ["game_id", "possession_id"] if "game_id" in actions.columns else ["possession_id"]
    # A row with a NaN group key is dropped by groupby(dropna=True), so the prior nested loop never
    # visited it and its self-event never applied. Gate the GLOBAL self-event init the same way, or a
    # NaN-key goal/owngoal would score itself where the old loop left it 0.0/False (byte-identity).
    valid_key = actions[group_cols].notna().all(axis=1).to_numpy()
    if xg_column is not None:
        xg = actions.get(xg_column, pd.Series(0.0, index=actions.index)).fillna(0.0).to_numpy(dtype=float)
        out = np.where(self_event & valid_key, xg, 0.0)  # self-scoring / self-conceding pass
    else:
        xg = None
        out = self_event & valid_key  # bool
    for pos_idx in actions.groupby(group_cols, sort=False).indices.values():
        if len(pos_idx) < 2:
            continue  # no strictly-downstream position exists
        g_team = team[pos_idx]
        g_same = same_event[pos_idx]
        g_other = other_event[pos_idx]
        for tteam in pd.unique(g_team):
            if pd.isna(tteam):
                continue  # a NULL-team position matches no team (same_id NA-never-equal) -> self only
            # ndarray (not Series) so the boolean-index reads below stay positional
            same_team = np.asarray(ids_match(g_team, tteam))  # team[j] == tteam (NA-never-equal)
            other_team = pd.notna(g_team) & ~same_team  # opponent: both present AND different
            eligible = (g_same & same_team) | (g_other & other_team)
            ev = np.where(eligible, xg[pos_idx], 0.0) if xg is not None else eligible
            rev = _suffix_after(ev)
            local = pos_idx[same_team]  # global positions of this team
            if xg_column is not None:
                out[local] = np.maximum(out[local], rev[same_team])
            else:
                out[local] = out[local] | rev[same_team]

    return pd.DataFrame({col_name: out}, index=actions.index)


def _scores_possession(actions: pd.DataFrame, xg_column: str | None) -> pd.DataFrame:
    """Possession-chain windowed scoring labels."""
    goal = _is_goal(actions).to_numpy()
    owngoal = _is_owngoal(actions).to_numpy()
    # scores: downstream same-team GOAL, or downstream opponent OWNGOAL; self = a goal action.
    return _possession_labels(
        actions, xg_column, col_name="scores", same_event=goal, other_event=owngoal, self_event=goal
    )


def _concedes_possession(actions: pd.DataFrame, xg_column: str | None) -> pd.DataFrame:
    """Possession-chain windowed conceding labels."""
    goal = _is_goal(actions).to_numpy()
    owngoal = _is_owngoal(actions).to_numpy()
    # concedes: downstream opponent GOAL, or downstream same-team OWNGOAL; self = an owngoal action.
    return _possession_labels(
        actions, xg_column, col_name="concedes", same_event=owngoal, other_event=goal, self_event=owngoal
    )


def _scores_time(
    actions: pd.DataFrame,
    window_seconds: float,
    xg_column: str | None,
) -> pd.DataFrame:
    """Time-windowed scoring labels using searchsorted for strict inequality."""
    goal = _is_goal(actions)
    owngoal = _is_owngoal(actions)
    team_id = np.asarray(actions["team_id"].values)
    time_s = np.asarray(actions["time_seconds"].values, dtype=np.float64)

    if xg_column is not None:
        xg = np.asarray(
            actions.get(xg_column, pd.Series(0.0, index=actions.index)).fillna(0.0).values, dtype=np.float64
        )  # type: ignore[reportOptionalMemberAccess]

    result = np.zeros(len(actions), dtype=float) if xg_column else np.zeros(len(actions), dtype=bool)

    # Group by (game_id, period_id) for boundary isolation
    group_keys: list[str] = []
    if "game_id" in actions.columns:
        group_keys.append("game_id")
    if "period_id" in actions.columns:
        group_keys.append("period_id")

    if group_keys:
        groups = actions.groupby(group_keys)
    else:
        groups = [(None, actions)]  # type: ignore[assignment]

    for _key, grp in groups:
        idx = np.asarray(grp.index)
        t = time_s[idx]
        g = goal.values[idx]
        og = owngoal.values[idx]
        tid = team_id[idx]

        # Precondition: time_seconds must be non-decreasing within each period
        if len(t) > 1 and not (np.diff(t) >= -1e-9).all():
            raise ValueError(
                "time_seconds must be non-decreasing within each (game_id, period_id) group. "
                "Sort actions by (game_id, period_id, time_seconds) before calling."
            )

        # searchsorted with side="left": boundary = first index where t >= t[i] + window_seconds
        # This gives strict inequality: only actions j where t[j] < t[i] + window_seconds
        boundaries = np.searchsorted(t, t + window_seconds, side="left")

        for local_i in range(len(idx)):
            global_i = idx[local_i]
            end = boundaries[local_i]
            for local_j in range(local_i + 1, min(end, len(idx))):
                if g[local_j]:
                    if _same_team_scalar(tid[local_i], tid[local_j]):
                        if xg_column:
                            result[global_i] = max(float(result[global_i]), float(xg[idx[local_j]]))
                        else:
                            result[global_i] = True
                            break
                if og[local_j]:
                    if _other_team_scalar(tid[local_i], tid[local_j]):
                        if xg_column:
                            result[global_i] = max(float(result[global_i]), float(xg[idx[local_j]]))
                        else:
                            result[global_i] = True
                            break

            # The goal action itself
            if g[local_i]:
                if xg_column:
                    result[global_i] = max(float(result[global_i]), float(xg[idx[local_i]]))
                else:
                    result[global_i] = True

    return pd.DataFrame({"scores": result})


def _concedes_time(
    actions: pd.DataFrame,
    window_seconds: float,
    xg_column: str | None,
) -> pd.DataFrame:
    """Time-windowed conceding labels using searchsorted for strict inequality."""
    goal = _is_goal(actions)
    owngoal = _is_owngoal(actions)
    team_id = np.asarray(actions["team_id"].values)
    time_s = np.asarray(actions["time_seconds"].values, dtype=np.float64)

    if xg_column is not None:
        xg = np.asarray(
            actions.get(xg_column, pd.Series(0.0, index=actions.index)).fillna(0.0).values, dtype=np.float64
        )  # type: ignore[reportOptionalMemberAccess]

    result = np.zeros(len(actions), dtype=float) if xg_column else np.zeros(len(actions), dtype=bool)

    group_keys: list[str] = []
    if "game_id" in actions.columns:
        group_keys.append("game_id")
    if "period_id" in actions.columns:
        group_keys.append("period_id")

    if group_keys:
        groups = actions.groupby(group_keys)
    else:
        groups = [(None, actions)]  # type: ignore[assignment]

    for _key, grp in groups:
        idx = np.asarray(grp.index)
        t = time_s[idx]
        g = goal.values[idx]
        og = owngoal.values[idx]
        tid = team_id[idx]

        if len(t) > 1 and not (np.diff(t) >= -1e-9).all():
            raise ValueError(
                "time_seconds must be non-decreasing within each (game_id, period_id) group. "
                "Sort actions by (game_id, period_id, time_seconds) before calling."
            )

        boundaries = np.searchsorted(t, t + window_seconds, side="left")

        for local_i in range(len(idx)):
            global_i = idx[local_i]
            end = boundaries[local_i]
            for local_j in range(local_i + 1, min(end, len(idx))):
                if g[local_j]:
                    if _other_team_scalar(tid[local_i], tid[local_j]):
                        if xg_column:
                            result[global_i] = max(float(result[global_i]), float(xg[idx[local_j]]))
                        else:
                            result[global_i] = True
                            break
                if og[local_j]:
                    if _same_team_scalar(tid[local_i], tid[local_j]):
                        if xg_column:
                            result[global_i] = max(float(result[global_i]), float(xg[idx[local_j]]))
                        else:
                            result[global_i] = True
                            break

            # The owngoal action itself concedes
            if og[local_i]:
                if xg_column:
                    result[global_i] = max(float(result[global_i]), float(xg[idx[local_i]]))
                else:
                    result[global_i] = True

    return pd.DataFrame({"concedes": result})


def goal_from_shot(actions: pd.DataFrame) -> pd.DataFrame:
    """Determine whether a goal was scored from the current action.

    This label can be use to train an xG model.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'goal' and a row for each action set to
        True if a goal was scored from the current action; otherwise False.

    Examples
    --------
    Build per-action goal labels for an xG model::

        from silly_kicks.spadl import add_names
        from silly_kicks.vaep.labels import goal_from_shot

        actions_with_names = add_names(actions)
        y = goal_from_shot(actions_with_names)
        # y["goal_from_shot"] is True only on shot rows that resulted in a goal.
    """
    goals = _is_goal(actions)

    return pd.DataFrame(goals, columns=["goal_from_shot"])


def save_from_shot(actions: pd.DataFrame) -> pd.DataFrame:
    """Determine whether the goalkeeper saved the current shot.

    This label can be used to train an Expected Saves (xS) model.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'save_from_shot' and a row for each action
        set to True if the action is a keeper save; otherwise False.

    Examples
    --------
    Build per-action keeper-save labels for an Expected Saves (xS) model::

        from silly_kicks.spadl import add_names
        from silly_kicks.vaep.labels import save_from_shot

        actions_with_names = add_names(actions)
        y = save_from_shot(actions_with_names)
        # y["save_from_shot"] is True only on successful keeper_save rows.
    """
    saves = actions["type_name"].str.contains("keeper_save") & (actions["result_id"] == spadl.result_id["success"])
    return pd.DataFrame(saves, columns=["save_from_shot"])


def claim_from_cross(actions: pd.DataFrame) -> pd.DataFrame:
    """Determine whether the goalkeeper claimed the current cross.

    This label can be used to train an Expected Claims (xC) model.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'claim_from_cross' and a row for each action
        set to True if the action is a keeper claim; otherwise False.

    Examples
    --------
    Build per-action keeper-claim labels for an Expected Claims (xC) model::

        from silly_kicks.spadl import add_names
        from silly_kicks.vaep.labels import claim_from_cross

        actions_with_names = add_names(actions)
        y = claim_from_cross(actions_with_names)
        # y["claim_from_cross"] is True only on successful keeper_claim rows.
    """
    claims = actions["type_name"].str.contains("keeper_claim") & (actions["result_id"] == spadl.result_id["success"])
    return pd.DataFrame(claims, columns=["claim_from_cross"])
