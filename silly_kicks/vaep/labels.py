"""Implements the label tranformers of the VAEP framework."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadl

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
        same_team = team_id == shifted_team
        result = result | (shifted_goal & same_team) | (shifted_owngoal & ~same_team)

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
        same_team = team_id == shifted_team
        result = result | (shifted_goal & ~same_team) | (shifted_owngoal & same_team)

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
        same_team = team_id == shifted_team
        score_xg = shifted_xg.where(shifted_goal & same_team, 0.0)
        owngoal_xg = shifted_xg.where(shifted_owngoal & ~same_team, 0.0)
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
        same_team = team_id == shifted_team
        concede_xg = shifted_xg.where(shifted_goal & ~same_team, 0.0)
        owngoal_xg = shifted_xg.where(shifted_owngoal & same_team, 0.0)
        result = result.combine(concede_xg + owngoal_xg, max, fill_value=0.0)  # type: ignore[reportArgumentType]
    return pd.DataFrame({"concedes": result})


def _scores_possession(actions: pd.DataFrame, xg_column: str | None) -> pd.DataFrame:
    """Possession-chain windowed scoring labels."""
    goal = _is_goal(actions)
    owngoal = _is_owngoal(actions)
    team_id = actions["team_id"]

    if xg_column is not None:
        xg = actions.get(xg_column, pd.Series(0.0, index=actions.index)).fillna(0.0)  # type: ignore[reportOptionalMemberAccess]

    group_cols = ["game_id", "possession_id"] if "game_id" in actions.columns else ["possession_id"]
    result: pd.Series = (  # type: ignore[type-arg]
        pd.Series(0.0, index=actions.index) if xg_column is not None else pd.Series(False, index=actions.index)
    )

    for _key, grp in actions.groupby(group_cols):
        idx = grp.index
        for i, pos in enumerate(idx):
            for j_pos in idx[i + 1 :]:
                if goal.loc[j_pos]:
                    same_team = team_id.loc[pos] == team_id.loc[j_pos]
                    if same_team:
                        if xg_column is not None:
                            result.loc[pos] = max(result.loc[pos], xg.loc[j_pos])
                        else:
                            result.loc[pos] = True
                            break
                if owngoal.loc[j_pos]:
                    same_team = team_id.loc[pos] == team_id.loc[j_pos]
                    if not same_team:
                        if xg_column is not None:
                            result.loc[pos] = max(result.loc[pos], xg.loc[j_pos])
                        else:
                            result.loc[pos] = True
                            break
        # The goal action itself scores
        for pos in idx:
            if goal.loc[pos]:
                if xg_column is not None:
                    result.loc[pos] = max(result.loc[pos], xg.loc[pos])
                else:
                    result.loc[pos] = True

    return pd.DataFrame(result, columns=["scores"])


def _concedes_possession(actions: pd.DataFrame, xg_column: str | None) -> pd.DataFrame:
    """Possession-chain windowed conceding labels."""
    goal = _is_goal(actions)
    owngoal = _is_owngoal(actions)
    team_id = actions["team_id"]

    if xg_column is not None:
        xg = actions.get(xg_column, pd.Series(0.0, index=actions.index)).fillna(0.0)  # type: ignore[reportOptionalMemberAccess]

    group_cols = ["game_id", "possession_id"] if "game_id" in actions.columns else ["possession_id"]
    result: pd.Series = (  # type: ignore[type-arg]
        pd.Series(0.0, index=actions.index) if xg_column is not None else pd.Series(False, index=actions.index)
    )

    for _key, grp in actions.groupby(group_cols):
        idx = grp.index
        for i, pos in enumerate(idx):
            for j_pos in idx[i + 1 :]:
                if goal.loc[j_pos]:
                    same_team = team_id.loc[pos] == team_id.loc[j_pos]
                    if not same_team:
                        if xg_column is not None:
                            result.loc[pos] = max(result.loc[pos], xg.loc[j_pos])
                        else:
                            result.loc[pos] = True
                            break
                if owngoal.loc[j_pos]:
                    same_team = team_id.loc[pos] == team_id.loc[j_pos]
                    if same_team:
                        if xg_column is not None:
                            result.loc[pos] = max(result.loc[pos], xg.loc[j_pos])
                        else:
                            result.loc[pos] = True
                            break
        # The owngoal action itself concedes
        for pos in idx:
            if owngoal.loc[pos]:
                if xg_column is not None:
                    result.loc[pos] = max(result.loc[pos], xg.loc[pos])
                else:
                    result.loc[pos] = True

    return pd.DataFrame(result, columns=["concedes"])


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
                    if tid[local_i] == tid[local_j]:
                        if xg_column:
                            result[global_i] = max(float(result[global_i]), float(xg[idx[local_j]]))
                        else:
                            result[global_i] = True
                            break
                if og[local_j]:
                    if tid[local_i] != tid[local_j]:
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
                    if tid[local_i] != tid[local_j]:
                        if xg_column:
                            result[global_i] = max(float(result[global_i]), float(xg[idx[local_j]]))
                        else:
                            result[global_i] = True
                            break
                if og[local_j]:
                    if tid[local_i] == tid[local_j]:
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
