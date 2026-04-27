"""Utility functions for working with SPADL dataframes."""

import warnings

import pandas as pd

from . import config as spadlconfig
from .schema import SPADL_COLUMNS


def add_names(actions: pd.DataFrame) -> pd.DataFrame:
    """Add the type name, result name and bodypart name to a SPADL dataframe.

    All columns not in the SPADL schema are preserved unchanged.

    Parameters
    ----------
    actions : pd.DataFrame
        A SPADL dataframe.

    Returns
    -------
    pd.DataFrame
        The original dataframe with 'type_name', 'result_name' and
        'bodypart_name' appended.

    .. note::
        All caller-added columns (e.g. ``match_id``, ``competition_id``,
        ``data_source``) are preserved in the returned DataFrame alongside
        the three appended name columns.
    """
    return (
        actions.drop(columns=["type_name", "result_name", "bodypart_name"], errors="ignore")
        .merge(spadlconfig.actiontypes_df(), how="left")  # type: ignore[reportOptionalMemberAccess]
        .merge(spadlconfig.results_df(), how="left")
        .merge(spadlconfig.bodyparts_df(), how="left")
        .set_index(actions.index)
    )


def play_left_to_right(actions: pd.DataFrame, home_team_id: int) -> pd.DataFrame:
    """Perform all actions in the same playing direction.

    This changes the start and end location of each action, such that all actions
    are performed as if the team that executes the action plays from left to
    right.

    Parameters
    ----------
    actions : pd.DataFrame
        The SPADL actions of a game.
    home_team_id : int
        The ID of the home team.

    Returns
    -------
    pd.DataFrame
        All actions performed left to right.

    See Also
    --------
    silly_kicks.vaep.features.play_left_to_right : For transforming gamestates.
    """
    ltr_actions = actions.copy()
    away_idx = actions.team_id != home_team_id
    for col in ["start_x", "end_x"]:
        ltr_actions.loc[away_idx, col] = spadlconfig.field_length - actions[away_idx][col].values  # type: ignore[reportAttributeAccessIssue]
    for col in ["start_y", "end_y"]:
        ltr_actions.loc[away_idx, col] = spadlconfig.field_width - actions[away_idx][col].values  # type: ignore[reportAttributeAccessIssue]
    return ltr_actions


def _finalize_output(
    df: pd.DataFrame,
    schema: dict[str, str] | None = None,
    *,
    extra_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Project to declared columns and enforce dtypes, optionally preserving extras.

    Parameters
    ----------
    df : pd.DataFrame
        The raw converter output DataFrame.
    schema : dict[str, str], optional
        Column name to dtype mapping. Defaults to SPADL_COLUMNS.
    extra_columns : list[str], optional
        Additional columns to preserve in the output, appended after the
        schema columns in the order specified. Each must be present in
        *df* and must NOT overlap with *schema*. Their existing dtypes
        are preserved (no coercion).

        Powers the public ``preserve_native`` parameter on each provider's
        ``convert_to_actions`` for surfacing provider-native fields (e.g.
        StatsBomb ``possession``) alongside the canonical SPADL columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with exactly the declared schema columns (with enforced
        dtypes) plus any *extra_columns* (with their input dtypes), in
        ``[*schema, *extra_columns]`` order.

    Raises
    ------
    ValueError
        If any *extra_columns* entry is missing from *df* or overlaps with
        *schema*.

    The returned DataFrame is guaranteed to contain exactly the columns
    in *schema* with the specified dtypes plus *extra_columns* with their
    input dtypes. Callers do not need defensive column-existence checks
    after calling a converter.
    """
    if schema is None:
        schema = SPADL_COLUMNS
    extras = list(extra_columns) if extra_columns else []

    if extras:
        missing = [c for c in extras if c not in df.columns]
        if missing:
            raise ValueError(
                f"_finalize_output: extra_columns missing from df: {sorted(missing)}. Available: {sorted(df.columns)}"
            )
        overlap = [c for c in extras if c in schema]
        if overlap:
            raise ValueError(
                f"_finalize_output: extra_columns overlap with schema: {sorted(overlap)}. "
                f"These names are already part of the canonical schema; remove them from extra_columns."
            )

    cols = [*schema.keys(), *extras]
    result = df[cols].copy()
    for col, dtype in schema.items():
        result[col] = result[col].astype(dtype)
    return result


def _validate_preserve_native(
    df: pd.DataFrame,
    preserve_native: list[str] | None,
    provider: str,
    *,
    schema: dict[str, str] | None = None,
) -> None:
    """Validate that ``preserve_native`` fields are present in the input df
    AND do not overlap with the canonical output schema.

    Powers the public ``preserve_native`` parameter on each provider's
    ``convert_to_actions``. Raises early — before the conversion pipeline
    starts — so callers see a clear error rather than a deep-stack
    duplicate-column failure later in the pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        The input event DataFrame.
    preserve_native : list[str] or None
        Provider-native fields the caller wants to preserve through
        conversion. ``None`` or empty list is a no-op.
    provider : str
        Provider name for error messages.
    schema : dict[str, str], optional
        Output schema to check against for overlap. Defaults to
        ``SPADL_COLUMNS``. Pass a different schema (e.g.
        ``KLOPPY_SPADL_COLUMNS``) when relevant.

    Raises
    ------
    ValueError
        If any field in ``preserve_native`` is missing from ``df``, OR
        if any field overlaps with ``schema``'s column names.
    """
    if not preserve_native:
        return
    if schema is None:
        schema = SPADL_COLUMNS
    missing = [c for c in preserve_native if c not in df.columns]
    if missing:
        raise ValueError(
            f"{provider} convert_to_actions: preserve_native fields missing from events: "
            f"{sorted(missing)}. Available columns: {sorted(df.columns)}"
        )
    overlap = [c for c in preserve_native if c in schema]
    if overlap:
        raise ValueError(
            f"{provider} convert_to_actions: preserve_native fields overlap with the SPADL schema: "
            f"{sorted(overlap)}. These are already canonical SPADL columns; remove them from preserve_native."
        )


def _validate_input_columns(
    df: pd.DataFrame,
    expected: set[str],
    provider: str,
) -> None:
    """Validate that a DataFrame has all expected columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to validate.
    expected : set[str]
        Set of required column names.
    provider : str
        Provider name for error messages.

    Raises
    ------
    ValueError
        If any expected columns are missing.
    """
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"{provider} convert_to_actions: missing required columns: {sorted(missing)}. Got: {sorted(df.columns)}"
        )


def validate_spadl(
    df: pd.DataFrame,
    schema: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Validate that a DataFrame conforms to the SPADL schema.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    schema : dict[str, str], optional
        Column name to dtype mapping. Defaults to SPADL_COLUMNS.

    Returns
    -------
    pd.DataFrame
        The input DataFrame unchanged (for chaining).

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    if schema is None:
        schema = SPADL_COLUMNS
    missing = set(schema.keys()) - set(df.columns)
    if missing:
        raise ValueError(f"Missing SPADL columns: {sorted(missing)}")
    for col, expected_dtype in schema.items():
        actual_dtype = str(df[col].dtype)
        if actual_dtype != expected_dtype:
            warnings.warn(
                f"Column '{col}' has dtype '{actual_dtype}', expected '{expected_dtype}'",
                stacklevel=2,
            )
    return df
