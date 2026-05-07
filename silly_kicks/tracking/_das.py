"""Dangerous Accessible Space adapter (TF-28).

Thin wrapper over the ``accessible-space`` PyPI package (MIT), mapping
silly-kicks 20-column tracking schema to the library's API.

See docs/superpowers/specs/2026-05-06-tf28-tf29-das-vaep-variants-design.md
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

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
    PFF is the primary provider affected (Int64 player_id/team_id/team_in_possession).
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
    ball_mask = out["is_ball"] == True  # noqa: E712
    out.loc[ball_mask, "player_id"] = "ball"
    return out


def get_das(
    frames: pd.DataFrame,
    *,
    use_progress_bar: bool = False,
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
    prepared = _prepare_frames(frames)

    ret = asmod.get_dangerous_accessible_space(
        prepared,
        ball_player_id="ball",
        x_pitch_min=_X_PITCH_MIN,
        x_pitch_max=_X_PITCH_MAX,
        y_pitch_min=_Y_PITCH_MIN,
        y_pitch_max=_Y_PITCH_MAX,
        infer_attacking_direction=True,
        use_progress_bar=use_progress_bar,
        **_COLUMN_MAP,
        **kwargs,
    )

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


def get_individual_das(
    frames: pd.DataFrame,
    *,
    use_progress_bar: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Per-player Accessible Space and Dangerous Accessible Space per frame.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames with ``vx``, ``vy``, ``team_in_possession``.
    use_progress_bar : bool, default False
        Show progress bar.
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
    prepared = _prepare_frames(frames)

    ret = asmod.get_individual_dangerous_accessible_space(
        prepared,
        ball_player_id="ball",
        x_pitch_min=_X_PITCH_MIN,
        x_pitch_max=_X_PITCH_MAX,
        y_pitch_min=_Y_PITCH_MIN,
        y_pitch_max=_Y_PITCH_MAX,
        infer_attacking_direction=True,
        use_progress_bar=use_progress_bar,
        **_COLUMN_MAP,
        **kwargs,
    )

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
    _validate_das_inputs(frames)

    prepared_frames = _to_das_coords(frames)
    prepared_frames["player_id"] = prepared_frames["player_id"].astype(object)
    ball_mask = prepared_frames["is_ball"] == True  # noqa: E712
    prepared_frames.loc[ball_mask, "player_id"] = "ball"

    prepared_passes = passes.copy()
    prepared_passes["start_x"] = prepared_passes["start_x"] - _X_OFFSET
    prepared_passes["start_y"] = prepared_passes["start_y"] - _Y_OFFSET
    prepared_passes["end_x"] = prepared_passes["end_x"] - _X_OFFSET
    prepared_passes["end_y"] = prepared_passes["end_y"] - _Y_OFFSET

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
