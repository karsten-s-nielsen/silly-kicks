"""Canonical per-action re-projection of frame-sampled positions into SPADL action-LTR.

``convert_to_frames`` emits home-attacks-right coordinates (the home team attacks
x=105 in every period); ``to_spadl_ltr`` emits per-acting-team-LTR action
coordinates (the *acting* team attacks x=105). The two conventions agree for
home-team actions and are a 180-degree point reflection (``x->105-x, y->68-y``)
apart for away-team actions.

Every emitted per-action tracking-geometry POSITION column must be expressed in
the action-LTR frame of the action it annotates. This module is the single
source of truth for that re-projection. Decision: ADR-028.

The per-action flip is derived from the frame's ``team_attacking_direction``
column (ground truth of "which way does this team attack in these
coordinates"), so the helper is robust to ANY frame orientation and needs no
``home_team_id``.
"""

from __future__ import annotations

import pandas as pd

from ._id_compat import align_join_keys

FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0

__all__ = ["FIELD_LENGTH", "FIELD_WIDTH", "acting_team_attacks_rtl", "reproject_to_action_ltr"]


def acting_team_attacks_rtl(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
) -> pd.Series:
    """Per-action boolean: True iff the acting team attacks RIGHT-TO-LEFT in the frames.

    A True row means the action's LTR frame is the 180-degree mirror of the frame
    coordinate system, so frame-sampled positions for that action must be flipped
    (``x->105-x, y->68-y``) to land in the action-LTR frame.

    Derivation: build a ``(game_id, period_id, team_id) -> attacking_direction``
    lookup from non-ball frame rows, then map each action's
    ``(game_id, period_id, team_id)``. Actions whose acting team has no resolvable
    direction (absent from the frame, or NaN/None direction) default to False (no
    flip); such actions produce NaN geometry anyway because they cannot link to a
    usable position.

    Returns
    -------
    pd.Series
        Boolean Series index-aligned to ``actions``.
    """
    flip = pd.Series(False, index=actions.index)
    if len(actions) == 0 or len(frames) == 0:
        return flip
    if "team_attacking_direction" not in frames.columns:
        return flip

    # Adapt the join keys to whatever team-direction identity is present on BOTH frames
    # and actions. period_id + team_id are always present (schema); game_id is included
    # only when both carry it (a minimal single-game fixture / the context path may omit
    # it -- the linker itself keys on (period_id, frame_id), not game_id).
    keys = [k for k in ("game_id", "period_id", "team_id") if k in actions.columns and k in frames.columns]
    if "team_id" not in keys or "period_id" not in keys:
        return flip

    players = frames[~frames["is_ball"].astype(bool)]
    players = players[players["team_attacking_direction"].notna()]
    if players.empty:
        return flip

    # One direction per key tuple: first non-null (constant within a period).
    lookup = (
        players.groupby(keys)["team_attacking_direction"]
        .first()
        .reset_index()
        .rename(columns={"team_attacking_direction": "_dir"})
    )

    # Dtype-safe id join (ADR-019): a numeric action team_id vs object-string frame team_id
    # would silently mis-match and compute the wrong flip. align_join_keys reconciles the
    # id-valued keys (no-op for already-matching dtypes).
    left = actions[keys].copy()
    left, lookup = align_join_keys(left, lookup, keys)
    keyed = left.merge(lookup, on=keys, how="left")
    keyed.index = actions.index
    return (keyed["_dir"] == "rtl").fillna(False)


def reproject_to_action_ltr(
    df: pd.DataFrame,
    flip_mask: pd.Series,
    *,
    x_cols: list[str],
    y_cols: list[str],
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``x_cols``/``y_cols`` mirrored where ``flip_mask``.

    ``x -> 105 - x`` and ``y -> 68 - y`` on flipped rows; NaN is preserved (NaN
    arithmetic). ``flip_mask`` is reindexed to ``df`` (missing -> False).
    """
    out = df.copy()
    mask = flip_mask.reindex(out.index, fill_value=False).to_numpy(dtype=bool)
    if not mask.any():
        return out
    for col in x_cols:
        if col in out.columns:
            vals = out[col].to_numpy(dtype="float64")
            out.loc[mask, col] = FIELD_LENGTH - vals[mask]
    for col in y_cols:
        if col in out.columns:
            vals = out[col].to_numpy(dtype="float64")
            out.loc[mask, col] = FIELD_WIDTH - vals[mask]
    return out
