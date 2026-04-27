"""Utility functions for working with SPADL dataframes."""

import warnings
from typing import Final

import numpy as np
import pandas as pd

from . import config as spadlconfig
from .schema import SPADL_COLUMNS

_SET_PIECE_RESTART_TYPE_NAMES: Final[frozenset[str]] = frozenset(
    {"freekick_short", "freekick_crossed", "corner_short", "corner_crossed", "throw_in", "goalkick"}
)
"""SPADL action types that represent a stoppage-restart taken by the team
that won possession (or had possession before the stoppage). Recognised by
``add_possessions`` for the set-piece carve-out."""

_ADD_POSSESSIONS_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "period_id",
    "action_id",
    "time_seconds",
    "team_id",
    "type_id",
)


def add_possessions(
    actions: pd.DataFrame,
    *,
    max_gap_seconds: float = 5.0,
    retain_on_set_pieces: bool = True,
) -> pd.DataFrame:
    """Assign a per-match possession-sequence integer to each SPADL action.

    Provider-agnostic possession reconstruction. Adds a ``possession_id``
    column (int64) to a copy of ``actions`` based on a team-change-with-
    carve-outs heuristic. The counter is per-match: it resets to 0 at each
    new ``game_id`` and is monotonically non-decreasing within a game.

    Algorithm
    ---------
    Actions are sorted by ``(game_id, period_id, action_id)``. A possession
    boundary is emitted when ANY of:

    1. ``game_id`` changes (counter resets to 0).
    2. ``period_id`` changes within the same game (counter increments).
    3. The time gap to the previous action is ``>= max_gap_seconds``.
    4. ``team_id`` changes from the previous action, EXCEPT when
       ``retain_on_set_pieces=True`` AND the current action is a set-piece
       restart (``freekick_short/_crossed``, ``corner_short/_crossed``,
       ``throw_in``, ``goalkick``) AND the previous action was a foul by
       the opposing team. In that case the team-change is NOT a possession
       boundary — the team that won the foul resumes its possession.

    The carve-out is approximate (StatsBomb's proprietary possession rules
    capture additional context), but matches typical published heuristics
    at boundary-F1 ~0.90 against StatsBomb's native possession_id.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL action stream. Must contain ``game_id``, ``period_id``,
        ``action_id``, ``time_seconds``, ``team_id``, ``type_id``. Other
        columns are preserved unchanged.
    max_gap_seconds : float, default 5.0
        Time-gap threshold (seconds) above which a new possession starts
        even if the team hasn't changed. Set to ``float("inf")`` to disable.
    retain_on_set_pieces : bool, default True
        Whether to apply the foul-then-set-piece carve-out (see Algorithm).

    Returns
    -------
    pd.DataFrame
        A copy of ``actions`` with an additional ``possession_id: int64``
        column. The returned DataFrame is sorted by
        ``(game_id, period_id, action_id)``.

    Raises
    ------
    ValueError
        If any required column is missing or if ``max_gap_seconds`` is
        negative.

    Examples
    --------
    Reconstruct possessions for any provider's SPADL output::

        actions, _ = statsbomb.convert_to_actions(events, home_team_id=100)
        actions = add_possessions(actions)
        # actions["possession_id"] now contains a per-match possession sequence.

    Validate a heuristic against a provider's native possession_id by
    preserving the source field through conversion (silly-kicks 1.1.0+)::

        actions, _ = statsbomb.convert_to_actions(
            events, home_team_id=100, preserve_native=["possession"]
        )
        actions = add_possessions(actions)
        # Compare actions["possession_id"] (heuristic) to actions["possession"] (native).
    """
    missing = [c for c in _ADD_POSSESSIONS_REQUIRED_COLUMNS if c not in actions.columns]
    if missing:
        raise ValueError(
            f"add_possessions: actions missing required columns: {sorted(missing)}. Got: {sorted(actions.columns)}"
        )
    if max_gap_seconds < 0:
        raise ValueError(f"add_possessions: max_gap_seconds must be >= 0, got {max_gap_seconds}")

    # Sort by canonical SPADL order. Stable sort preserves original-row order on ties.
    sorted_actions = actions.sort_values(["game_id", "period_id", "action_id"], kind="mergesort").reset_index(drop=True)

    n = len(sorted_actions)
    if n == 0:
        sorted_actions["possession_id"] = pd.Series([], dtype=np.int64)
        return sorted_actions

    # Vectorised boundary detection.
    game_id = sorted_actions["game_id"].to_numpy()
    period_id = sorted_actions["period_id"].to_numpy()
    team_id = sorted_actions["team_id"].to_numpy()
    time_seconds = sorted_actions["time_seconds"].to_numpy()
    type_id = sorted_actions["type_id"].to_numpy()

    prev_game = np.empty(n, dtype=game_id.dtype)
    prev_period = np.empty(n, dtype=period_id.dtype)
    prev_team = np.empty(n, dtype=team_id.dtype)
    prev_time = np.empty(n, dtype=time_seconds.dtype)
    prev_type = np.empty(n, dtype=type_id.dtype)
    # Sentinel values for the first row; the game_change check on row 0 hits regardless
    # because we explicitly mark row 0 as a game_change boundary below.
    prev_game[0] = game_id[0]
    prev_period[0] = period_id[0]
    prev_team[0] = team_id[0]
    prev_time[0] = time_seconds[0]
    prev_type[0] = type_id[0]
    prev_game[1:] = game_id[:-1]
    prev_period[1:] = period_id[:-1]
    prev_team[1:] = team_id[:-1]
    prev_time[1:] = time_seconds[:-1]
    prev_type[1:] = type_id[:-1]

    game_change = np.empty(n, dtype=bool)
    game_change[0] = True  # first row is always the start of a possession in its game.
    game_change[1:] = game_id[1:] != game_id[:-1]

    period_change_within_game = (~game_change) & (period_id != prev_period)
    gap_timeout = (~game_change) & (~period_change_within_game) & ((time_seconds - prev_time) >= max_gap_seconds)
    team_change = (~game_change) & (team_id != prev_team)

    # Set-piece carve-out: current is a set-piece restart AND previous was a foul by other team.
    # Carve-out only matters when the team has changed (otherwise no team change → no boundary anyway).
    set_piece_ids = {spadlconfig.actiontype_id[name] for name in _SET_PIECE_RESTART_TYPE_NAMES}
    foul_id = spadlconfig.actiontype_id["foul"]
    is_set_piece = np.isin(type_id, list(set_piece_ids))
    prev_is_foul = prev_type == foul_id
    set_piece_carve_out = retain_on_set_pieces & team_change & is_set_piece & prev_is_foul

    new_possession_mask = game_change | period_change_within_game | gap_timeout | (team_change & ~set_piece_carve_out)

    # Per-game cumulative count of new-possession events; subtract 1 so the first row of each
    # game is possession_id=0 (game_change is True there → cumsum starts at 1 → -1 = 0).
    sorted_actions["_new_possession"] = new_possession_mask.astype(np.int64)
    sorted_actions["possession_id"] = (
        sorted_actions.groupby("game_id", sort=False)["_new_possession"].cumsum() - 1
    ).astype(np.int64)
    sorted_actions = sorted_actions.drop(columns=["_new_possession"])

    return sorted_actions


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
        # ``np.dtype(dtype)`` narrows the schema's str dtype name to a typed
        # ``DtypeObj`` so pandas-stubs's ``astype`` overload set accepts it.
        result[col] = result[col].astype(np.dtype(dtype))
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
