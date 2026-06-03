"""Dangerous Accessible Space adapter (TF-28).

Thin wrapper over the ``accessible-space`` PyPI package (MIT), mapping
silly-kicks 20-column tracking schema to the library's API.

See docs/superpowers/specs/2026-05-06-tf28-tf29-das-vaep-variants-design.md
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import contextlib
import warnings

import pandas as pd

# Coordinate transform constants: silly-kicks [0,105]x[0,68] <-> DAS [-52.5,52.5]x[-34,34]
_X_OFFSET = 52.5
_Y_OFFSET = 34.0
_X_PITCH_MIN = -52.5
_X_PITCH_MAX = 52.5
_Y_PITCH_MIN = -34
_Y_PITCH_MAX = 34

# Column-name mapping: silly-kicks schema -> accessible-space parameter names.
_COLUMN_MAP = {
    "x_col": "x",
    "y_col": "y",
    "vx_col": "vx",
    "vy_col": "vy",
    "player_col": "player_id",
    "team_col": "team_id",
    "frame_col": "frame_id",
    "period_col": "period_id",
    "team_in_possession_col": "team_in_possession",
}

#: Canonical per-frame ball-carrier (passer) column, produced by derive_team_in_possession.
#: Forwarded to accessible-space as player_in_possession_col so respect_offside (the DAS
#: default) excludes the passer from the offside mask (Phase 0c).
_DEFAULT_PLAYER_IN_POSSESSION_COL = "ball_carrier_player_id"

#: Module-level one-time guard so the no-carrier guidance isn't emitted per call.
_OFFSIDE_WARNED = False

#: Emitted (stacklevel=2 at each DAS entry point) when a frame subset has no frame
#: containing both the ball and players -- see _has_simulatable_frame.
_NO_SIMULATABLE_FRAME_MSG = (
    "DAS has no simulatable frame (no frame contains both the ball and players "
    "with a resolved team_in_possession); returning NaN DAS. accessible-space would "
    "otherwise build a zero-frame simulation and crash on a None dereference."
)

#: Emitted (stacklevel=2) when no pass references a frame containing both the
#: ball and players -- the xC analogue of _NO_SIMULATABLE_FRAME_MSG.
_NO_SIMULATABLE_XC_FRAME_MSG = (
    "xC has no simulatable pass (no pass references a frame containing both the ball "
    "and players); returning NaN xC. accessible-space would otherwise build a "
    "zero-frame simulation and crash."
)


def _resolve_player_in_possession_col(frames: pd.DataFrame, player_in_possession_col: str | None) -> str | None:
    """Resolve the carrier column to forward to accessible-space.

    - ``None``: caller opted out; do not forward.
    - present on ``frames``: forward it (correct offside -- passer excluded).
    - explicitly named (non-default) but missing: ``ValueError`` (caller contract violation).
    - default name missing (e.g. old frames): ``None`` (degrade to prior behavior).
    """
    if player_in_possession_col is None:
        return None
    if player_in_possession_col in frames.columns:
        return player_in_possession_col
    if player_in_possession_col != _DEFAULT_PLAYER_IN_POSSESSION_COL:
        raise ValueError(f"player_in_possession_col={player_in_possession_col!r} not found in frames columns")
    return None


@contextlib.contextmanager
def _suppress_offside_warning():
    """Suppress accessible-space's per-call offside warning; silly-kicks owns this UX."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*player_in_possession_col.*", category=UserWarning)
        yield


def _warn_no_carrier_once() -> None:
    """Emit silly-kicks' own offside-no-carrier guidance once per process (stacklevel=2)."""
    global _OFFSIDE_WARNED
    if not _OFFSIDE_WARNED:
        _OFFSIDE_WARNED = True
        warnings.warn(
            "DAS respect_offside is on but no ball-carrier column was available to exclude "
            "the passer from the offside mask. Pass player_in_possession_col, or run "
            "derive_team_in_possession (which now preserves ball_carrier_player_id). "
            "Proceeding without passer exclusion.",
            UserWarning,
            stacklevel=2,
        )


def _import_accessible_space():  # type: ignore[return]
    """Lazy import guard for the optional accessible-space package."""
    try:
        import accessible_space  # type: ignore[import-not-found]

        return accessible_space
    except ImportError as e:
        raise ImportError(
            "accessible-space is required for DAS features. Install with: pip install 'silly-kicks[das]'"
        ) from e


def _to_das_coords(frames: pd.DataFrame) -> pd.DataFrame:
    """Shift silly-kicks [0,105]x[0,68] to DAS [-52.5,52.5]x[-34,34]."""
    out = frames.copy()
    out["x"] = out["x"] - _X_OFFSET
    out["y"] = out["y"] - _Y_OFFSET
    return out


def _validate_das_inputs(frames: pd.DataFrame) -> None:
    """Validate required columns, raising with actionable messages."""
    if "vx" not in frames.columns or "vy" not in frames.columns:
        raise ValueError(
            "DAS requires velocity columns ('vx', 'vy'). Call derive_velocities() or smooth_frames() first."
        )
    if "team_in_possession" not in frames.columns:
        raise ValueError(
            "DAS requires a 'team_in_possession' column. Call derive_team_in_possession(frames, carrier_df) to add it."
        )


def _prepare_frames(frames: pd.DataFrame) -> pd.DataFrame:
    """Validate, transform coordinates, normalise ball rows.

    Casts nullable pandas dtypes (Int64, boolean) to numpy equivalents
    because the accessible-space library cannot handle nullable arrays
    (e.g. BooleanArray comparisons produce 2-D structures that crash).
    Gradient Sports is the primary provider affected (Int64 player_id/team_id/team_in_possession).
    """
    _validate_das_inputs(frames)
    out = _to_das_coords(frames)
    # Downcast nullable pandas dtypes to numpy equivalents for accessible-space compat.
    for col in out.columns:
        dtype_name = str(out[col].dtype)
        if dtype_name == "Int64":
            out[col] = out[col].astype(object)
        elif dtype_name == "boolean":
            out[col] = out[col].astype(bool)
    # accessible-space indexes the team / player columns 2-D (e.g. ``passer_teams[:, None]``
    # in the offside path). pandas StringDtype / pyarrow-backed arrays (the default for string
    # columns on newer pandas) reject 2-D indexing -> "IndexError: too many indices for array".
    # Force numpy ``object`` so the library always sees a plain ndarray. (Idempotent for the
    # object/int64 columns it already handled.)
    for col in ("team_id", "team_in_possession", "player_id"):
        if col in out.columns:
            out[col] = out[col].astype(object)
    ball_mask = out["is_ball"] == True  # noqa: E712
    out.loc[ball_mask, "player_id"] = "ball"
    return out


def _frames_with_ball_and_players(frames: pd.DataFrame) -> set:
    """Frame ids that contain BOTH a ball row and a player (non-ball) row.

    accessible-space restricts every simulation to exactly these frames —
    ``transform_into_arrays`` computes ``frames_to_consider = ball_frames &
    player_frames`` and drops the rest. When the result is empty it builds a
    zero-frame ``PLAYER_POS`` (``F == 0``); ``simulate_passes_chunked`` then
    returns ``None`` (or trips a matrix-consistency assertion), and the caller
    dereferences the ``None`` simulation result — a hard crash
    (``AttributeError`` on the DAS path, ``AssertionError`` on the xC path)
    rather than an honest NaN. accessible-space's own "is the data empty?"
    guards run *before* this intersection, so they do not catch the disjoint
    case. ``frames`` rows carry ``is_ball`` (``_prepare_frames`` convention).
    """
    is_ball = frames["is_ball"] == True  # noqa: E712
    ball_frames = set(frames.loc[is_ball, "frame_id"].unique())
    player_frames = set(frames.loc[~is_ball, "frame_id"].unique())
    return ball_frames & player_frames


def _has_simulatable_frame(prepared: pd.DataFrame) -> bool:
    """True iff DAS has at least one simulatable frame (ball + players present).

    Mirrors accessible-space's DAS selection exactly: ``get_dangerous_accessible_space``
    drops rows whose ``team_in_possession`` is NaN *first*, then
    ``transform_into_arrays`` intersects ball/player frame sets. A link-restricted
    subset whose ball frames and player frames are disjoint (e.g. one action batch
    whose linked frames lost their ball or their player rows) collapses to
    ``F == 0``; detecting it here lets DAS degrade to NaN — consistent with
    silly-kicks' "undefined case -> NaN DAS" contract — instead of crashing.
    See ``_frames_with_ball_and_players``. ``prepared`` is ``_prepare_frames`` output.
    """
    poss = prepared[prepared["team_in_possession"].notna()]
    if poss.empty:
        return False
    return len(_frames_with_ball_and_players(poss)) > 0


def _nan_das_result(frames: pd.DataFrame) -> pd.DataFrame:
    """A copy of ``frames`` with all-NaN ``AS``/``DAS`` columns (degenerate DAS)."""
    result = frames.copy()
    result["AS"] = float("nan")
    result["DAS"] = float("nan")
    return result


def get_das(
    frames: pd.DataFrame,
    *,
    use_progress_bar: bool = False,
    player_in_possession_col: str | None = _DEFAULT_PLAYER_IN_POSSESSION_COL,
    **kwargs,
) -> pd.DataFrame:
    """Team-level Accessible Space and Dangerous Accessible Space per frame.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames. Must contain ``vx``, ``vy``, and
        ``team_in_possession`` columns.
    use_progress_bar : bool, default False
        Show progress bar during simulation.
    **kwargs
        Passthrough to ``accessible_space.get_dangerous_accessible_space``.

    Returns
    -------
    pd.DataFrame
        Input frames with added ``AS`` and ``DAS`` columns (float64).

    Examples
    --------
    Full pipeline from raw tracking::

        from silly_kicks.tracking import (
            derive_velocities, infer_ball_carrier, derive_team_in_possession,
        )
        from silly_kicks.tracking._das import get_das

        frames = derive_velocities(raw_frames)
        carrier = infer_ball_carrier(frames)
        frames = derive_team_in_possession(frames, carrier)
        result = get_das(frames)

    See NOTICE for full bibliographic citations.
    """
    asmod = _import_accessible_space()
    ppc = _resolve_player_in_possession_col(frames, player_in_possession_col)
    prepared = _prepare_frames(frames)

    if not _has_simulatable_frame(prepared):
        warnings.warn(_NO_SIMULATABLE_FRAME_MSG, UserWarning, stacklevel=2)
        return _nan_das_result(frames)

    with _suppress_offside_warning():
        ret = asmod.get_dangerous_accessible_space(
            prepared,
            ball_player_id="ball",
            x_pitch_min=_X_PITCH_MIN,
            x_pitch_max=_X_PITCH_MAX,
            y_pitch_min=_Y_PITCH_MIN,
            y_pitch_max=_Y_PITCH_MAX,
            infer_attacking_direction=True,
            player_in_possession_col=ppc,
            use_progress_bar=use_progress_bar,
            **_COLUMN_MAP,
            **kwargs,
        )
    if ppc is None:
        _warn_no_carrier_once()

    if len(ret.acc_space) != len(prepared):
        warnings.warn(
            f"accessible-space returned {len(ret.acc_space)} values for "
            f"{len(prepared)} input rows; output may be misaligned",
            UserWarning,
            stacklevel=2,
        )

    result = frames.copy()
    result["AS"] = ret.acc_space
    result["DAS"] = ret.das
    return result


def _pin_attacking_direction(frames: pd.DataFrame) -> pd.DataFrame:
    """Attach an ``attacking_direction`` column inferred from the FULL frames.

    accessible-space infers playing direction per ``(period, team_in_possession)``
    from the mean x-position over the frames it is *given*. When the frame set is
    later restricted to a handful of action-linked frames, that subset can infer a
    flipped direction and silently change DAS. Pinning the direction here — reusing
    the library's own ``infer_playing_direction`` for identical semantics — lets
    callers restrict frames while keeping the full-frame sign, so per-frame DAS
    stays bit-identical. Direction inference is shift-invariant (mean-x ordering),
    so running it on un-shifted silly-kicks coordinates matches what the library
    would compute internally on its shifted coordinates.

    Returns a copy of ``frames`` with an added ``attacking_direction`` column.
    """
    _import_accessible_space()  # fail fast with the actionable install message
    # Raise the canonical ValueError (which DAS consumers already catch and
    # degrade to NaN) on missing vx/vy/team_in_possession, instead of letting
    # accessible-space's infer_playing_direction raise an uncaught KeyError.
    _validate_das_inputs(frames)
    # Dead-ball window: when team_in_possession is all-NaN (e.g. the ball is out of
    # play and infer_ball_carrier found no carrier), infer_playing_direction asserts
    # ("no non-ball teams in common") — an AssertionError that escapes add_das's
    # except. Raise the canonical ValueError instead so DAS degrades to NaN here.
    # Attacking direction is genuinely undefined without a possessing team; silly-kicks
    # does NOT fabricate possession (ADR / PR-S67 invariant) — supply
    # attacking_direction_col=... to bypass inference when the direction is known.
    if not frames["team_in_possession"].notna().any():
        raise ValueError(
            "team_in_possession is all-NaN (dead-ball window): attacking direction is "
            "undefined, so DAS is undefined here. add_das degrades these actions to NaN."
        )
    from accessible_space.interface import infer_playing_direction

    out = frames.copy()
    # Mirror the library's internal handling: ball rows carry no team.
    ball_mask = out["is_ball"] == True  # noqa: E712
    out.loc[ball_mask, "team_id"] = None
    direction = infer_playing_direction(
        out,
        team_col="team_id",
        period_col="period_id",
        team_in_possession_col="team_in_possession",
        x_col="x",
        ball_team=None,
        frame_col="frame_id",
    )
    out["attacking_direction"] = direction.to_numpy()
    # infer_playing_direction adds its own 'playing_direction' column in place.
    return out.drop(columns=["playing_direction"], errors="ignore")


def get_individual_das(
    frames: pd.DataFrame,
    *,
    use_progress_bar: bool = False,
    attacking_direction_col: str | None = None,
    player_in_possession_col: str | None = _DEFAULT_PLAYER_IN_POSSESSION_COL,
    **kwargs,
) -> pd.DataFrame:
    """Per-player Accessible Space and Dangerous Accessible Space per frame.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames with ``vx``, ``vy``, ``team_in_possession``.
    use_progress_bar : bool, default False
        Show progress bar.
    attacking_direction_col : str or None, default None
        When given, the column on ``frames`` holding a precomputed per-frame
        attacking direction; the library uses it verbatim instead of inferring
        direction from the input frames. When None, direction is inferred (the
        default). See ``_pin_attacking_direction``.
    **kwargs
        Passthrough to ``accessible_space.get_individual_dangerous_accessible_space``.

    Returns
    -------
    pd.DataFrame
        Input frames with added ``AS`` and ``DAS`` columns (float64, per-player).

    Examples
    --------
    Per-player DAS decomposition::

        from silly_kicks.tracking._das import get_individual_das
        result = get_individual_das(frames)

    See NOTICE for full bibliographic citations.
    """
    asmod = _import_accessible_space()
    ppc = _resolve_player_in_possession_col(frames, player_in_possession_col)
    prepared = _prepare_frames(frames)

    if not _has_simulatable_frame(prepared):
        warnings.warn(_NO_SIMULATABLE_FRAME_MSG, UserWarning, stacklevel=2)
        return _nan_das_result(frames)

    with _suppress_offside_warning():
        ret = asmod.get_individual_dangerous_accessible_space(
            prepared,
            ball_player_id="ball",
            x_pitch_min=_X_PITCH_MIN,
            x_pitch_max=_X_PITCH_MAX,
            y_pitch_min=_Y_PITCH_MIN,
            y_pitch_max=_Y_PITCH_MAX,
            infer_attacking_direction=attacking_direction_col is None,
            attacking_direction_col=attacking_direction_col,
            player_in_possession_col=ppc,
            use_progress_bar=use_progress_bar,
            **_COLUMN_MAP,
            **kwargs,
        )
    if ppc is None:
        _warn_no_carrier_once()

    if len(ret.player_acc_space) != len(prepared):
        warnings.warn(
            f"accessible-space returned {len(ret.player_acc_space)} values for "
            f"{len(prepared)} input rows; output may be misaligned",
            UserWarning,
            stacklevel=2,
        )

    result = frames.copy()
    result["AS"] = ret.player_acc_space
    result["DAS"] = ret.player_das
    return result


def get_xc(
    passes: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    use_progress_bar: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Expected pass completion (xC) for each pass using tracking context.

    Parameters
    ----------
    passes : pd.DataFrame
        SPADL actions filtered to passes. Must contain ``start_x``, ``start_y``,
        ``end_x``, ``end_y``, ``team_id``, ``player_id``.
    frames : pd.DataFrame
        Long-form tracking frames with ``vx``, ``vy``, ``team_in_possession``.
    use_progress_bar : bool, default False
        Show progress bar.
    **kwargs
        Passthrough to ``accessible_space.get_expected_pass_completion``.

    Returns
    -------
    pd.DataFrame
        Copy of ``passes`` with added ``xC`` column (float64, probability).

    Examples
    --------
    Compute xC for all passes in a match::

        from silly_kicks.tracking._das import get_xc
        pass_actions = actions[actions["type_name"] == "pass"]
        result = get_xc(pass_actions, frames)

    See NOTICE for full bibliographic citations.
    """
    asmod = _import_accessible_space()
    # Use the canonical frame prep (validates inputs, shifts coords, labels the ball,
    # downcasts Int64/boolean, and coerces team/team_in_possession/player_id to numpy
    # object). The object coercion of the team columns is required: accessible-space's
    # offside path 2-D-indexes the team arrays (passer_teams[:, np.newaxis]), which a
    # pyarrow-backed StringDtype column (the default on newer pandas / py3.11+) rejects
    # with "IndexError: too many indices for array". The old lighter prep only coerced
    # player_id, so xC crashed there on the CI 3.11/3.12 legs. Mirrors get_das.
    prepared_frames = _prepare_frames(frames)

    prepared_passes = passes.copy()
    prepared_passes["start_x"] = prepared_passes["start_x"] - _X_OFFSET
    prepared_passes["start_y"] = prepared_passes["start_y"] - _Y_OFFSET
    prepared_passes["end_x"] = prepared_passes["end_x"] - _X_OFFSET
    prepared_passes["end_y"] = prepared_passes["end_y"] - _Y_OFFSET
    # Coerce the pass identifier columns to numpy object for the same reason as the
    # tracking team columns above: with use_event_team_as_team_in_possession (the
    # default) accessible-space derives passer_teams from the event team_id, then
    # 2-D-indexes it (passer_teams[:, np.newaxis]) -- a pyarrow StringDtype rejects it.
    for _col in ("team_id", "player_id"):
        if _col in prepared_passes.columns:
            prepared_passes[_col] = prepared_passes[_col].astype(object)

    # Same degenerate-frame fragility as the DAS path: accessible-space simulates one
    # frame per pass (its event frame) and keeps only frames with BOTH a ball row and
    # player rows. If no pass references such a frame the intersection is empty -> F==0
    # -> the simulation result is None and gets dereferenced (here an AssertionError on
    # the matrix-consistency check). Degrade to NaN xC instead. (Only when the pass
    # frame column is present; otherwise let accessible-space raise its own column error.)
    if "frame_id" in prepared_passes.columns:
        pass_frame_ids = set(prepared_passes["frame_id"].dropna().unique())
        relevant_frames = prepared_frames[prepared_frames["frame_id"].isin(pass_frame_ids)]
        if not _frames_with_ball_and_players(relevant_frames):
            warnings.warn(_NO_SIMULATABLE_XC_FRAME_MSG, UserWarning, stacklevel=2)
            result = passes.copy()
            result["xC"] = float("nan")
            return result

    ret = asmod.get_expected_pass_completion(
        prepared_passes,
        prepared_frames,
        event_frame_col="frame_id",
        event_player_col="player_id",
        event_team_col="team_id",
        event_start_x_col="start_x",
        event_start_y_col="start_y",
        event_end_x_col="end_x",
        event_end_y_col="end_y",
        tracking_frame_col=_COLUMN_MAP["frame_col"],
        tracking_player_col=_COLUMN_MAP["player_col"],
        tracking_team_col=_COLUMN_MAP["team_col"],
        tracking_x_col=_COLUMN_MAP["x_col"],
        tracking_y_col=_COLUMN_MAP["y_col"],
        tracking_vx_col=_COLUMN_MAP["vx_col"],
        tracking_vy_col=_COLUMN_MAP["vy_col"],
        ball_tracking_player_id="ball",
        x_pitch_min=_X_PITCH_MIN,
        x_pitch_max=_X_PITCH_MAX,
        y_pitch_min=_Y_PITCH_MIN,
        y_pitch_max=_Y_PITCH_MAX,
        tracking_period_col=_COLUMN_MAP["period_col"],
        infer_attacking_direction=True,
        use_progress_bar=use_progress_bar,
        **kwargs,
    )

    result = passes.copy()
    result["xC"] = ret.xc
    return result
