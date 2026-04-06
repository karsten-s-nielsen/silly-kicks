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
    """
    return (
        actions.drop(columns=["type_name", "result_name", "bodypart_name"], errors="ignore")
        .merge(spadlconfig.actiontypes_df(), how="left")
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
) -> pd.DataFrame:
    """Project to declared columns and enforce dtypes.

    Parameters
    ----------
    df : pd.DataFrame
        The raw converter output DataFrame.
    schema : dict[str, str], optional
        Column name to dtype mapping. Defaults to SPADL_COLUMNS.

    Returns
    -------
    pd.DataFrame
        DataFrame with exactly the declared columns and dtypes.
    """
    if schema is None:
        schema = SPADL_COLUMNS
    result = df[list(schema.keys())].copy()
    for col, dtype in schema.items():
        result[col] = result[col].astype(dtype)
    return result


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
