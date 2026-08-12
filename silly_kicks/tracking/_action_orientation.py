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

    # Function-local import, matching _shot_goalmouth.py's use of the same helper: this module is
    # a low-level orientation primitive and _gk_geometry is a high-level restart resolver, so a
    # module-level import here would invert the layering.
    from ._gk_geometry import _truthy_bool

    # ADR-019: NEVER `.astype(bool)` a provider string qualifier -- `pd.Series(["False"])
    # .astype(bool)` is True, so `~` selected NO player rows and this guard silently inspected an
    # EMPTY frame for every provider emitting an object/string `is_ball` -- i.e. it could not have
    # rejected a contradictory labelling on exactly those providers.
    players = frames[~_truthy_bool(frames["is_ball"])]
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


def _is_shootout_only(actions: pd.DataFrame) -> bool:
    """True iff EVERY action is period 5 (PSO), where orientation is undefined by design.

    ``direction.py``'s ``_LTR_KNOWN_PERIODS = (1, 2, 3, 4)`` already excludes period 5 because
    shootout direction has no meaning, and ``test_off_ball_runs_orientation.py`` pins period-5
    frames as an ACCEPTED unoriented shape. Warning about it would be noise, not signal.

    Requires EVERY action to be period 5: a call mixing a shootout with real play still has
    resolvable rows, so it must still warn.
    """
    if "period_id" not in actions.columns or len(actions) == 0:
        return False
    return bool((actions["period_id"] == 5).all())


def _warn_unresolved(reason: str) -> None:
    """One message for every silent-failure exit (ADR-028 D2).

    Specified by OUTCOME, not by enumerated condition: any wholly-unresolved return that is not
    "there were no actions to flip" warns. An enumerated fix rots the next time a branch is
    added -- which is exactly how the join-key branch was missed when this fix was first
    specified as "absent or all-null".

    The outcome it names is now all-``<NA>`` rather than all-False (4.80.0). The warning is
    still worth emitting: a consumer that answers ``<NA>`` with ``.fillna(False)`` lands exactly
    where the old contract put it, and this message is what tells the caller the frames, not the
    consumer, are the thing to fix.
    """
    import warnings

    from ._warnings import OrientationUnresolvedWarning

    warnings.warn(
        f"acting_team_attacks_rtl: no action's direction could be resolved ({reason}), so the "
        "returned flip is entirely <NA>. Consumers that treat <NA> as 'no flip' will apply no "
        "ADR-028 re-projection, mixing coordinate conventions for away-team geometry. Orient "
        "the frames first -- convert_to_frames(output_convention='ltr') "
        "or tracking.orient_frames_to_ltr().",
        OrientationUnresolvedWarning,
        stacklevel=3,
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
    ``(game_id, period_id, team_id)``.

    **An action whose direction cannot be resolved yields <NA>, never False.** This is a
    NULLABLE boolean, and that is the whole point: the previous contract returned a bare
    ``bool`` with ``.fillna(False)``, so a consumer could not distinguish a RESOLVED
    left-to-right team from one whose direction was simply unknown. Twenty-one call sites
    inherited that guess, and every one of them silently re-projected nothing while looking
    like it had decided.

    The old docstring justified the default with *"such actions produce NaN geometry anyway
    because they cannot link to a usable position"*. That argument was made for off-ball runs
    and MEASURED not to transfer: an xT grid (``_player_influence``) exists whether or not any
    single action links, so a defaulted row emitted a real number computed against a guessed
    orientation.

    **Consumers must now decide explicitly.** ``.fillna(False)`` is still the right answer at
    many sites -- a positional metric that is symmetric under the flip, or a path that already
    NaNs unlinkable rows -- but it has to be WRITTEN, with a reason. That is the difference
    between a considered default and an inherited one. See ADR-055's amendment (4.80.0).

    Returns
    -------
    pd.Series
        Nullable boolean (``dtype="boolean"``) index-aligned to ``actions``. ``<NA>`` marks an
        action whose acting team has no resolvable attacking direction in ``frames``.
    """
    # <NA>, not False: every early exit below is an UNRESOLVED case, and the whole contract
    # is that unresolved is distinguishable from "resolved, does not flip".
    flip = pd.Series(pd.NA, index=actions.index, dtype="boolean")
    if len(actions) == 0:
        return flip  # nothing to resolve -- the ONE legitimate silent no-op
    # Period-5 (PSO) orientation is undefined by design, so an unresolved direction there is
    # expected rather than a defect. Suppresses the WARNING only; the all-<NA> return below is
    # unchanged, so no behaviour moves.
    _quiet = _is_shootout_only(actions)
    if len(frames) == 0:
        if not _quiet:
            _warn_unresolved("frames is empty")
        return flip
    if "team_attacking_direction" not in frames.columns:
        if not _quiet:
            _warn_unresolved("frames has no team_attacking_direction column")
        return flip

    # Adapt the join keys to whatever team-direction identity is present on BOTH frames
    # and actions. period_id + team_id are always present (schema); game_id is included
    # only when both carry it (a minimal single-game fixture / the context path may omit
    # it -- the linker itself keys on (period_id, frame_id), not game_id).
    keys = [k for k in ("game_id", "period_id", "team_id") if k in actions.columns and k in frames.columns]
    if "team_id" not in keys or "period_id" not in keys:
        if not _quiet:
            _warn_unresolved("actions and frames do not share the team_id + period_id join keys")
        return flip

    # See the note in validate_period_directions on why this import is function-local.
    from ._gk_geometry import _truthy_bool

    # ADR-019: NEVER `.astype(bool)` a provider string qualifier -- `pd.Series(["False"])
    # .astype(bool)` is True, so `~` selected NO player rows and this resolver fell through to
    # its "nothing resolved" exit for every provider emitting an object/string `is_ball`. That
    # is the ADR-028 defect firing on a whole input class, and it stayed invisible until 4.80.0
    # because the fall-through returned all-False -- indistinguishable from a legitimately
    # all-home action set. Found by the <NA> contract, not by the warning: the warning DID fire,
    # but "unoriented frames" is a routine condition, so it read as expected noise.
    players = frames[~_truthy_bool(frames["is_ball"])]
    players = players[players["team_attacking_direction"].notna()]
    if players.empty:
        if not _quiet:
            _warn_unresolved("team_attacking_direction is present but entirely null")
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

    # THE ESCAPE ROUTE the early-exit branches cannot see (ADR-028 D2). Frames that are labelled
    # and keyed can still resolve NOTHING: the acting team may simply be absent from the frames,
    # or the join keys may carry different id spellings (measured during the D2 sweep --
    # actions keyed `game_id="idsse_J03WMX"` against frames keyed `"J03WMX"`). Both looked
    # oriented and were not. Since 4.80.0 the return is all-<NA> rather than all-False, so this
    # case is now VISIBLE in the value as well as in the warning -- but the warning stays,
    # because a consumer is free to fillna(False) and land back in the silent no-op.
    #
    # The signal is NOTHING RESOLVED, not nothing flipped: a legitimately all-home action set
    # yields an all-False (resolved!) flip and must stay silent. A PARTIAL miss is silent too --
    # ADR-027 NaN-team rows never resolve, and warning on those would fire on healthy GS data
    # every call.
    if not _quiet and not keyed["_dir"].notna().any():
        _warn_unresolved("no action's join key matched any frame row (team absent, or id spellings differ)")
    # NA where the direction did not resolve -- NOT `.fillna(False)`, which is the defect this
    # contract removes. A row that resolved to "ltr" is False; a row that resolved to nothing is
    # <NA>, and the consumer has to say which it wants.
    resolved = keyed["_dir"].notna()
    out = pd.Series(pd.NA, index=actions.index, dtype="boolean")
    out[resolved] = keyed.loc[resolved, "_dir"] == "rtl"
    return out


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
