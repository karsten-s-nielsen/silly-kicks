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

from silly_kicks.id_compat import align_join_keys

FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0

__all__ = [
    "FIELD_LENGTH",
    "FIELD_WIDTH",
    "acting_team_attacks_rtl",
    "reproject_to_action_ltr",
    "validate_period_directions",
]


#: Periods whose attacking direction is defined. Mirrors ``direction.py:29``
#: (``_LTR_KNOWN_PERIODS``); period 5 is a penalty shootout, played at a single end, so
#: orientation is genuinely undefined -- ``orient_frames_to_ltr_by_geometry`` never flips
#: it either. Duplicated deliberately rather than imported: importing ``direction`` here
#: would add an edge from this leaf module to a much heavier one.
_ORIENTED_PERIODS: tuple[int, ...] = (1, 2, 3, 4)


def validate_period_directions(frames: pd.DataFrame, *, caller: str) -> None:
    """Raise if a single team CONTRADICTS ITSELF about its attacking direction.

    Scope is deliberately narrow: the only genuinely impossible state is one team resolving
    to BOTH ``"ltr"`` and ``"rtl"`` within one ``(game_id, period_id)``. Everything else that
    might look wrong is a legitimate, in-library convention:

    * **Unoriented** -- the frame makes no orientation claim (``team_attacking_direction``
      absent or all-null). ``skillcorner.py:282`` / ``metrica.py:180`` emit exactly this for
      ``output_convention="absolute_frame"`` (documented at ``skillcorner.py:180``), and it
      is what ``scripts/_loader_pining.py`` feeds the training corpora.
    * **A different convention** -- ``snapshot_to_tracking_frames`` (``_snapshot.py:92``)
      labels every player ``"ltr"`` because snapshot frames are ALREADY in SPADL action-LTR,
      so "never flip" is the correct reading, not a contradiction.
    * **Undefined by nature** -- period 5 (PSO); see ``_ORIENTED_PERIODS``.

    For all three, ``acting_team_attacks_rtl`` returning False ("no orientation asserted, so
    no flip") is the CONTRACT, not a gap -- an earlier draft of this guard treated a missing
    direction as "mislabelled" and consequently rejected all three shapes, regressing paths
    that had always worked.

    **Promotion into ``acting_team_attacks_rtl`` itself is REJECTED ON EVIDENCE, not
    deferred.** Even this narrowed rule stays at the single consumer that needs it, because
    the three shapes above are produced by the library itself, so no amount of consumer-side
    data could establish the precondition a blanket guard would need. Recorded in ADR-041.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames.
    caller : str
        Public name used in the error message.

    Raises
    ------
    ValueError
        If one team carries BOTH directions within a single period.

    Examples
    --------
    Guard a consumer's input::

        from silly_kicks.tracking._action_orientation import validate_period_directions

        validate_period_directions(frames, caller="my_feature")
    """
    required = {"team_attacking_direction", "team_id", "is_ball", "period_id"}
    if not required.issubset(frames.columns) or frames.empty:
        return

    players = frames[~frames["is_ball"].astype(bool)]
    # Period 5 (PSO) is excluded even though the narrowed rule makes it redundant today
    # (an all-null period cannot self-contradict). Kept as an explicit, documented
    # invariant so that a future widening of this guard cannot silently re-break shootout
    # frames -- which is precisely how the earlier draft went wrong.
    players = players[players["period_id"].isin(_ORIENTED_PERIODS)]
    if players.empty:
        return

    keys = ["game_id", "period_id"] if "game_id" in players.columns else ["period_id"]
    for key, grp in players.groupby(keys, sort=False):
        per_team = grp.groupby("team_id", sort=False)["team_attacking_direction"].agg(
            lambda s: set(s.dropna().unique())
        )
        contradictory = sorted(str(t) for t, dirs in per_team.items() if len(dirs) > 1)
        if contradictory:
            raise ValueError(
                f"{caller}: team(s) {contradictory} carry BOTH 'ltr' and 'rtl' in period "
                f"{key!r}. A team attacks one way per period, so these frames contradict "
                "themselves. Re-orient them -- see silly_kicks.tracking.orient_frames_to_ltr "
                "/ orient_frames_to_ltr_by_geometry (ADR-029). (An all-null or uniformly "
                "labelled column is NOT an error: it means unoriented, or a different "
                "convention such as snapshot frames, and is accepted.)"
            )


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
    vx_cols: list[str] | None = None,
    vy_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with the named columns re-projected where ``flip_mask``.

    Positions map ``x -> 105 - x`` / ``y -> 68 - y``; velocity columns are NEGATED
    (a 180-degree point reflection reverses a vector). NaN is preserved.
    ``flip_mask`` is reindexed to ``df`` (missing -> False).

    ``vx_cols``/``vy_cols`` exist because a positions-only re-projection silently
    produced velocity that contradicted its own positions (ADR-045 D1).
    """
    from silly_kicks.reflection import reflect_columns

    return reflect_columns(
        df,
        flip_mask,
        point_x=x_cols,
        point_y=y_cols,
        vector_x=vx_cols or [],
        vector_y=vy_cols or [],
    )
