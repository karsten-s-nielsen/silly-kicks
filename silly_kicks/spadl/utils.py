"""Utility functions for working with SPADL dataframes."""

import os
import warnings
from typing import TYPE_CHECKING, Final, TypedDict

import numpy as np
import pandas as pd

from silly_kicks._nan_safety import nan_safe_enrichment

from . import config as spadlconfig
from .schema import SPADL_COLUMNS

if TYPE_CHECKING:
    # Type-only import for pyright fidelity. Runtime call site lazy-imports per ADR-005 § 5.
    from silly_kicks.tracking.features import (  # noqa: F401
        add_pre_shot_gk_position as _tracking_add_pre_shot_gk_position,
    )

_SET_PIECE_RESTART_TYPE_NAMES: Final[frozenset[str]] = frozenset(
    {"freekick_short", "freekick_crossed", "corner_short", "corner_crossed", "throw_in", "goalkick"}
)
"""SPADL action types that represent a stoppage-restart taken by the team
that won possession (or had possession before the stoppage). Recognised by
``add_possessions`` for the set-piece carve-out."""

_PANDAS_EXTENSION_DTYPES: Final[frozenset[str]] = frozenset(
    {
        "Int64",
        "Int32",
        "Int16",
        "Int8",
        "UInt64",
        "UInt32",
        "UInt16",
        "UInt8",
        "Float64",
        "Float32",
        "boolean",
        "string",
    }
)
"""Pandas extension dtypes that must be passed as string names to ``astype``
rather than wrapped in ``np.dtype(...)``. Used by :func:`_finalize_output`
to support nullable / pandas-extension columns in provider output schemas
(e.g. ``PFF_SPADL_COLUMNS``'s ``Int64`` tackle-passthrough columns)."""

_ADD_POSSESSIONS_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "period_id",
    "action_id",
    "time_seconds",
    "team_id",
    "type_id",
)

_ADD_GK_ROLE_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "period_id",
    "action_id",
    "team_id",
    "player_id",
    "type_id",
    "start_x",
)

_GK_ROLE_CATEGORIES: Final[tuple[str, ...]] = (
    "shot_stopping",
    "cross_collection",
    "sweeping",
    "pick_up",
    "distribution",
)


@nan_safe_enrichment
def add_gk_role(
    actions: pd.DataFrame,
    *,
    penalty_area_x_threshold: float = 16.5,
    distribution_lookback_actions: int = 1,
    goalkeeper_ids: set | None = None,
) -> pd.DataFrame:
    """Tag each action with the goalkeeper's role context.

    Adds a ``gk_role`` categorical column with five categories
    (``shot_stopping`` / ``cross_collection`` / ``sweeping`` / ``pick_up`` /
    ``distribution``) plus null for non-GK actions. Provider-agnostic
    enrichment that mirrors the post-conversion shape of
    :func:`add_possessions` and :func:`add_names`.

    Categorisation rules
    --------------------
    For each row, ``gk_role`` is assigned by precedence (first match wins):

    1. ``sweeping`` — any ``keeper_*`` action with
       ``start_x > penalty_area_x_threshold``. Overrides the type-specific
       role: a ``keeper_save`` taken at ``start_x = 20`` (a sweeper-style
       rush-out save) is tagged ``sweeping``, not ``shot_stopping``.

       .. note::

           ``keeper_claim`` / ``keeper_punch`` / ``keeper_pick_up`` outside
           the penalty area are illegal handball offences in regulation
           football and effectively non-existent in clean event data; if
           one appears, treating it as ``sweeping`` is a pragmatic
           position-based tag rather than a semantically rigorous
           classification.
    2. ``shot_stopping`` — ``keeper_save`` inside the box.
    3. ``cross_collection`` — ``keeper_claim`` or ``keeper_punch`` inside the box.
    4. ``pick_up`` — ``keeper_pick_up`` inside the box.
    5. ``distribution`` — non-keeper action by the same player whose
       previous action (within ``distribution_lookback_actions`` steps in
       the same game) was a keeper action.
    6. ``None`` — everything else.

    GK identity is inferred from ``keeper_*`` action history; canonical SPADL
    has no ``position_group`` flag, so consumers who want to filter by
    registered GK position must do so via their own ``dim_players`` join.

    NaN values in caller-supplied identifier columns (e.g. ``player_id``)
    are treated as "not identifiable" for that row's enrichment lookup;
    downstream rows behave as if no identifier were present. See ADR-003.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL action stream. Must contain ``game_id``, ``period_id``,
        ``action_id``, ``team_id``, ``player_id``, ``type_id``, ``start_x``.
        Other columns are preserved unchanged.
    penalty_area_x_threshold : float, default 16.5
        SPADL x-coordinate (metres from own goal line) above which a
        keeper action is classified as ``sweeping`` rather than the
        type-specific role. Default matches the regulation 16.5m
        penalty-area depth.
    distribution_lookback_actions : int, default 1
        Number of preceding actions to consider when detecting a
        non-keeper action by the same player as ``distribution``. Default
        of 1 (immediately-preceding action only) avoids false positives
        from ball-possession returns; larger values widen the window.
    goalkeeper_ids : set, optional
        When provided, distribution-detection extends beyond strict
        ``same_player`` matching to also tag rows where:

        - The current row's ``player_id`` is in ``goalkeeper_ids`` AND the
          preceding action (within ``distribution_lookback_actions`` steps,
          same ``team_id`` and ``game_id``) was a keeper-type action.
          (Use case: caller knows the GK player_ids; clean-attribution data.)
        - Both the current row's and the preceding action's ``player_id``
          are NaN AND the ``team_id`` matches AND the preceding action was
          keeper-type. (Use case: caller's data has NaN player_id but the
          team/sequence implies the GK distributed the ball.)

        Opting in via this parameter signals that the caller accepts the
        coarser heuristic (the second rule may over-tag if multiple
        NaN-player_id non-keeper actions follow a keeper action by the
        same team within the lookback window).

        When ``None`` (default), only strict ``same_player`` matching applies
        — byte-for-byte compatible with pre-2.5.0 behavior.

    Returns
    -------
    pd.DataFrame
        A copy of ``actions`` sorted by ``(game_id, period_id, action_id)``
        with an additional ``gk_role`` Categorical column.

    Raises
    ------
    ValueError
        If a required column is missing, ``penalty_area_x_threshold`` is
        negative, or ``distribution_lookback_actions < 1``.

    References
    ----------
    Yam, "A Data-Driven Goalkeeper Evaluation Framework"
    (MIT Sloan Sports Analytics Conference). Four-pillar GK taxonomy
    (shot stopping / cross collection / distribution / defensive activity)
    inspires the categories used here.

    Examples
    --------
    Tag GK roles after conversion::

        actions, _ = statsbomb.convert_to_actions(events, home_team_id=100)
        actions = add_gk_role(actions)
        # Filter to distribution-from-the-back passes:
        gk_distribution = actions[actions["gk_role"] == "distribution"]
    """
    missing = [c for c in _ADD_GK_ROLE_REQUIRED_COLUMNS if c not in actions.columns]
    if missing:
        raise ValueError(
            f"add_gk_role: actions missing required columns: {sorted(missing)}. Got: {sorted(actions.columns)}"
        )
    if penalty_area_x_threshold < 0:
        raise ValueError(f"add_gk_role: penalty_area_x_threshold must be >= 0, got {penalty_area_x_threshold}")
    if distribution_lookback_actions < 1:
        raise ValueError(
            f"add_gk_role: distribution_lookback_actions must be >= 1, got {distribution_lookback_actions}"
        )

    sorted_actions = actions.sort_values(["game_id", "period_id", "action_id"], kind="mergesort").reset_index(drop=True)

    n = len(sorted_actions)
    if n == 0:
        sorted_actions["gk_role"] = pd.Categorical([], categories=list(_GK_ROLE_CATEGORIES))
        return sorted_actions

    type_id = sorted_actions["type_id"].to_numpy()
    start_x = sorted_actions["start_x"].to_numpy(dtype=np.float64)
    player_id = sorted_actions["player_id"]
    game_id = sorted_actions["game_id"]

    # Resolve keeper action type IDs at fit time (defensive against out-of-spec configs).
    save_id = spadlconfig.actiontype_id["keeper_save"]
    claim_id = spadlconfig.actiontype_id["keeper_claim"]
    punch_id = spadlconfig.actiontype_id["keeper_punch"]
    pickup_id = spadlconfig.actiontype_id["keeper_pick_up"]

    is_save = type_id == save_id
    is_claim = type_id == claim_id
    is_punch = type_id == punch_id
    is_pickup = type_id == pickup_id
    is_keeper = is_save | is_claim | is_punch | is_pickup
    is_outside_box = start_x > penalty_area_x_threshold
    is_sweeping = is_keeper & is_outside_box

    # Distribution detection: vectorised k-step lookback. For each k in
    # [1..distribution_lookback_actions], OR in a (prev_is_keeper & match & same_game) mask
    # where ``match`` is same_player by default, extended via goalkeeper_ids.
    prev_keeper_within_k = np.zeros(n, dtype=bool)
    is_keeper_series = pd.Series(is_keeper)
    team_id = sorted_actions["team_id"]
    for k in range(1, distribution_lookback_actions + 1):
        shifted_keeper = is_keeper_series.shift(k, fill_value=False).to_numpy(dtype=bool)
        shifted_player = player_id.shift(k).to_numpy()
        shifted_game = game_id.shift(k).to_numpy()
        cur_player_arr = player_id.to_numpy()
        cur_game_arr = game_id.to_numpy()
        same_player = cur_player_arr == shifted_player
        same_game = cur_game_arr == shifted_game

        match = same_player

        if goalkeeper_ids is not None:
            shifted_team = team_id.shift(k).to_numpy()
            cur_team_arr = team_id.to_numpy()
            same_team = cur_team_arr == shifted_team

            # Rule (a) — known-GK match: caller declared a GK player_id set.
            cur_is_known_gk = pd.Series(cur_player_arr).isin(goalkeeper_ids).to_numpy()
            match = match | (cur_is_known_gk & same_team)

            # Rule (b) — NaN-team fallback: both player_ids unidentifiable
            # but same team + prev was keeper. Coarse heuristic; caller
            # opts in by passing goalkeeper_ids (signals willingness to
            # accept over-counting risk on dense NaN data).
            cur_player_na = pd.isna(cur_player_arr)
            shifted_player_na = pd.isna(shifted_player)
            match = match | (cur_player_na & shifted_player_na & same_team)

        prev_keeper_within_k |= shifted_keeper & match & same_game

    is_distribution = (~is_keeper) & prev_keeper_within_k

    # Assign roles by precedence — sweeping overrides type-specific role.
    role = np.full(n, None, dtype=object)
    role[is_save & ~is_sweeping] = "shot_stopping"
    role[(is_claim | is_punch) & ~is_sweeping] = "cross_collection"
    role[is_pickup & ~is_sweeping] = "pick_up"
    role[is_sweeping] = "sweeping"
    role[is_distribution] = "distribution"

    sorted_actions["gk_role"] = pd.Categorical(role, categories=list(_GK_ROLE_CATEGORIES))
    return sorted_actions


_ADD_GK_DISTRIBUTION_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "period_id",
    "action_id",
    "team_id",
    "player_id",
    "type_id",
    "result_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
)

_GK_LAUNCH_PASS_TYPE_NAMES: Final[frozenset[str]] = frozenset(
    {"pass", "goalkick", "freekick_short", "freekick_crossed"}
)
"""Action types that qualify as GK launches when the distance exceeds
``long_threshold``. Excludes ``clearance`` (not deliberate distribution per
Lamberts GVM) and ``throw_in`` / ``cross`` (rarely taken by GKs)."""

_GK_PASS_LENGTH_CATEGORIES: Final[tuple[str, ...]] = ("short", "medium", "long")

_PITCH_LENGTH_M: Final[float] = 105.0
_PITCH_WIDTH_M: Final[float] = 68.0


@nan_safe_enrichment
def add_gk_distribution_metrics(
    actions: pd.DataFrame,
    *,
    xt_grid: np.ndarray | None = None,
    short_threshold: float = 32.0,
    long_threshold: float = 60.0,
    require_gk_role: bool = True,
) -> pd.DataFrame:
    """Add length classification + optional xT delta to GK distribution actions.

    Operates only on rows where ``gk_role == "distribution"`` (per
    :func:`add_gk_role`). Adds four columns:

    - ``gk_pass_length_m`` — Euclidean ``(start, end)`` distance in metres
      (NaN on non-distribution rows).
    - ``gk_pass_length_class`` — Categorical ``{"short", "medium", "long"}``
      keyed off ``short_threshold`` / ``long_threshold``.
    - ``is_launch`` — bool. True iff length > ``long_threshold`` AND action
      is a deliberate-distribution pass type (``pass``, ``goalkick``,
      ``freekick_short``, ``freekick_crossed``). False everywhere else.
    - ``gk_xt_delta`` — float xT-grid delta from start zone to end zone,
      computed only when ``xt_grid`` is provided AND the pass succeeded
      (``result_id == "success"``). NaN otherwise.

    NaN values in caller-supplied identifier columns (e.g. ``player_id``)
    are treated as "not identifiable" for that row's enrichment lookup.
    Rows with NaN coordinates are excluded from xT-delta zone-binning
    (``gk_xt_delta`` is NaN for those rows). See ADR-003.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL action stream. Required columns: ``game_id``, ``period_id``,
        ``action_id``, ``team_id``, ``player_id``, ``type_id``, ``result_id``,
        ``start_x``, ``start_y``, ``end_x``, ``end_y``. ``gk_role`` is
        either present (used directly) or computed via :func:`add_gk_role`.
    xt_grid : np.ndarray, optional
        12x8 Expected Threat grid (SPADL coordinate system, rows = x-bins,
        cols = y-bins). When omitted, ``gk_xt_delta`` is NaN for all rows.
    short_threshold : float, default 32.0
        Pass length (metres) strictly below which a distribution is "short".
    long_threshold : float, default 60.0
        Pass length (metres) strictly above which a distribution is "long".
    require_gk_role : bool, default True
        When True (default) and ``gk_role`` column is absent, calls
        :func:`add_gk_role` internally with default kwargs. When False,
        skips distribution detection if ``gk_role`` is absent (all four
        added columns will be NaN/False).

    Returns
    -------
    pd.DataFrame
        Sorted copy of ``actions`` with the four columns appended. ``gk_role``
        is also present in output (either pre-existing or added internally).

    Raises
    ------
    ValueError
        If a required column is missing, ``short_threshold`` or
        ``long_threshold`` is negative, ``short_threshold >= long_threshold``,
        or ``xt_grid`` shape is not ``(12, 8)``.

    References
    ----------
    Lamberts (2025), "Introducing the Goalkeeper Value Model (GVM)" —
    short / medium / long distribution categorisation and launch-pass
    distinction.

    Examples
    --------
    Compute GK distribution metrics with xT valuation::

        from silly_kicks import xthreat
        actions, _ = statsbomb.convert_to_actions(events, home_team_id=100)
        xt = xthreat.fit(actions)  # produces 12x8 grid
        actions = add_gk_distribution_metrics(actions, xt_grid=xt.value_grid)
        # Filter to launches:
        launches = actions[actions["is_launch"]]
    """
    missing = [c for c in _ADD_GK_DISTRIBUTION_REQUIRED_COLUMNS if c not in actions.columns]
    if missing:
        raise ValueError(
            f"add_gk_distribution_metrics: actions missing required columns: {sorted(missing)}. "
            f"Got: {sorted(actions.columns)}"
        )
    if short_threshold < 0:
        raise ValueError(f"add_gk_distribution_metrics: short_threshold must be >= 0, got {short_threshold}")
    if long_threshold < 0:
        raise ValueError(f"add_gk_distribution_metrics: long_threshold must be >= 0, got {long_threshold}")
    if short_threshold >= long_threshold:
        raise ValueError(
            f"add_gk_distribution_metrics: short_threshold ({short_threshold}) must be strictly less "
            f"than long_threshold ({long_threshold})"
        )
    if xt_grid is not None and xt_grid.shape != (12, 8):
        raise ValueError(f"add_gk_distribution_metrics: xt_grid must have shape (12, 8), got {xt_grid.shape}")

    # Apply gk_role internally if needed.
    if "gk_role" not in actions.columns:
        if require_gk_role:
            actions = add_gk_role(actions)
        else:
            actions = actions.sort_values(["game_id", "period_id", "action_id"], kind="mergesort").reset_index(
                drop=True
            )
            actions["gk_role"] = pd.Categorical([None] * len(actions), categories=list(_GK_ROLE_CATEGORIES))

    n = len(actions)
    if n == 0:
        actions["gk_pass_length_m"] = pd.Series([], dtype=np.float64)
        actions["gk_pass_length_class"] = pd.Categorical([], categories=list(_GK_PASS_LENGTH_CATEGORIES))
        actions["is_launch"] = pd.Series([], dtype=bool)
        actions["gk_xt_delta"] = pd.Series([], dtype=np.float64)
        return actions

    # Identify distribution rows.
    is_distribution = (actions["gk_role"] == "distribution").to_numpy()

    # Vectorised length computation (defined for all rows; masked to NaN where not distribution).
    start_x = actions["start_x"].to_numpy(dtype=np.float64)
    start_y = actions["start_y"].to_numpy(dtype=np.float64)
    end_x = actions["end_x"].to_numpy(dtype=np.float64)
    end_y = actions["end_y"].to_numpy(dtype=np.float64)
    dx = end_x - start_x
    dy = end_y - start_y
    raw_length = np.sqrt(dx * dx + dy * dy)
    length_m = np.where(is_distribution, raw_length, np.nan)

    # Length classification.
    short_mask = is_distribution & (raw_length < short_threshold)
    long_mask = is_distribution & (raw_length > long_threshold)
    medium_mask = is_distribution & ~short_mask & ~long_mask

    length_class = np.full(n, None, dtype=object)
    length_class[short_mask] = "short"
    length_class[medium_mask] = "medium"
    length_class[long_mask] = "long"

    # is_launch — pass-type AND length > long_threshold.
    type_id = actions["type_id"].to_numpy()
    launch_type_ids = {spadlconfig.actiontype_id[name] for name in _GK_LAUNCH_PASS_TYPE_NAMES}
    is_launch_type = np.isin(type_id, list(launch_type_ids))
    is_launch = is_distribution & is_launch_type & (raw_length > long_threshold)

    # xT delta — only on successful distributions when grid provided.
    xt_delta = np.full(n, np.nan, dtype=np.float64)
    if xt_grid is not None:
        success_id = spadlconfig.result_id["success"]
        result_id_arr = actions["result_id"].to_numpy()
        # NaN coordinates would crash the .astype(int) zone-binning below.
        # Filter to rows where all four coords are finite (guards against
        # caller data with sparse spatial information). Non-finite rows
        # leave xt_delta at NaN (default initialization).
        coords_finite = np.isfinite(start_x) & np.isfinite(start_y) & np.isfinite(end_x) & np.isfinite(end_y)
        eligible = is_distribution & (result_id_arr == success_id) & coords_finite
        if eligible.any():
            zone_x_start = np.clip((start_x[eligible] / (_PITCH_LENGTH_M / 12.0)).astype(int), 0, 11)
            zone_y_start = np.clip((start_y[eligible] / (_PITCH_WIDTH_M / 8.0)).astype(int), 0, 7)
            zone_x_end = np.clip((end_x[eligible] / (_PITCH_LENGTH_M / 12.0)).astype(int), 0, 11)
            zone_y_end = np.clip((end_y[eligible] / (_PITCH_WIDTH_M / 8.0)).astype(int), 0, 7)
            xt_delta[eligible] = xt_grid[zone_x_end, zone_y_end] - xt_grid[zone_x_start, zone_y_start]

    actions["gk_pass_length_m"] = length_m
    actions["gk_pass_length_class"] = pd.Categorical(length_class, categories=list(_GK_PASS_LENGTH_CATEGORIES))
    actions["is_launch"] = is_launch
    actions["gk_xt_delta"] = xt_delta
    return actions


_ADD_PRE_SHOT_GK_CONTEXT_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "period_id",
    "action_id",
    "team_id",
    "player_id",
    "type_id",
    "time_seconds",
)

_GK_SHOT_TYPE_NAMES: Final[frozenset[str]] = frozenset({"shot", "shot_freekick", "shot_penalty"})
_GK_KEEPER_TYPE_NAMES: Final[frozenset[str]] = frozenset(
    {"keeper_save", "keeper_claim", "keeper_punch", "keeper_pick_up"}
)


@nan_safe_enrichment
def add_pre_shot_gk_context(
    actions: pd.DataFrame,
    *,
    frames: pd.DataFrame | None = None,
    lookback_seconds: float = 10.0,
    lookback_actions: int = 5,
) -> pd.DataFrame:
    """Tag each shot with the defending goalkeeper's recent activity (and optional position).

    For every shot row (``action_type`` in ``{"shot", "shot_freekick",
    "shot_penalty"}``), looks back up to ``lookback_actions`` rows OR up
    to ``lookback_seconds`` seconds in the same ``(game_id, period_id)``
    — whichever is smaller — to identify the defending GK and characterise
    their recent activity.

    Adds four columns:

    - ``gk_was_distributing`` (bool) — True iff the defending GK had a
      non-keeper action in the lookback window.
    - ``gk_was_engaged`` (bool) — True iff the defending GK had a
      ``keeper_*`` action in the lookback window.
    - ``gk_actions_in_possession`` (int) — count of ``keeper_*`` actions
      by the defending GK in the lookback window.
    - ``defending_gk_player_id`` (float, NaN-coded int) — ``player_id`` of
      the most recent defending-team ``keeper_*`` action in the window.
      NaN when no defending GK is identifiable.

    When ``frames`` is supplied, additionally emits 4 GK-position columns
    (``pre_shot_gk_x``, ``pre_shot_gk_y``, ``pre_shot_gk_distance_to_goal``,
    ``pre_shot_gk_distance_to_shot``) plus 4 linkage-provenance columns
    (``frame_id``, ``time_offset_seconds``, ``link_quality_score``,
    ``n_candidate_frames``) via the ``silly_kicks.tracking.features``
    canonical compute. When ``frames=None`` (default), behavior is bit-identical
    to silly-kicks 2.8.0 — no frames-related columns appear in the output.

    Non-shot rows receive default values (False / 0 / NaN). The defending
    GK is identified as the ``player_id`` of the most recent ``keeper_*``
    action by a team OTHER than the shooter's team; this is approximate
    but works on canonical SPADL streams without external GK metadata.

    .. note::

        This helper is genuinely novel — no published OSS / academic
        equivalent surfaces a goalkeeper's pre-shot activity context as
        explicit per-shot features. It is intended as feature input to
        downstream PSxG / xGOT models that may benefit from knowing the
        defending GK's recent engagement state. Validate empirically
        before drawing causal conclusions.

    NaN values in caller-supplied identifier columns (e.g. ``player_id``)
    are treated as "not identifiable" for that row's enrichment lookup;
    when the most-recent defending-keeper-action's ``player_id`` is NaN,
    the shot's GK-context columns receive their per-row default
    (``gk_was_engaged=False``, ``defending_gk_player_id=NaN``). See ADR-003.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL action stream. Required columns: ``game_id``, ``period_id``,
        ``action_id``, ``team_id``, ``player_id``, ``type_id``,
        ``time_seconds``.
    frames : pd.DataFrame | None, default None
        Long-form tracking frames matching ``TRACKING_FRAMES_COLUMNS``.
        When supplied, enables 4 GK-position + 4 linkage-provenance output
        columns. PR-S21 (silly-kicks 2.9.0+).
    lookback_seconds : float, default 10.0
        Time window (seconds) before each shot in which to consider
        defending-GK actions.
    lookback_actions : int, default 5
        Action-count window before each shot. The smaller of the two
        bounds wins.

    Returns
    -------
    pd.DataFrame
        Sorted copy of ``actions`` with the four context columns appended
        (and 4 GK-position + 4 provenance columns if ``frames`` supplied).

    Raises
    ------
    ValueError
        If a required column is missing, ``lookback_seconds`` is negative,
        or ``lookback_actions < 1``.

    References
    ----------
    Related work:

    - Butcher et al. (2025), "An Expected Goals On Target (xGOT) Model"
      (MDPI) — focuses on the shot moment; does not surface pre-shot GK
      engagement state.
    - Anzer, G., & Bauer, P. (2021), "A goal scoring probability model for
      shots based on synchronized positional and event data in football
      and futsal." Frontiers in Sports and Active Living, 3, 624475 —
      defending-GK position as xG feature; basis of the 4 GK-position
      columns when ``frames`` is supplied.

    Examples
    --------
    Tag pre-shot goalkeeper context for downstream PSxG / xGOT modeling
    (events-only path, silly-kicks 2.8.0 backward-compat)::

        actions, _ = statsbomb.convert_to_actions(events, home_team_id=100)
        actions = add_pre_shot_gk_context(actions, lookback_seconds=10.0)
        engaged_shots = actions[actions["gk_was_engaged"]]

    Events + tracking path (silly-kicks 2.9.0+)::

        from silly_kicks.tracking import sportec
        frames, _ = sportec.convert_to_frames(raw, home_team_id="DFL-CLU-A", home_team_start_left=True)
        actions = add_pre_shot_gk_context(actions, frames=frames)
        # Now also has pre_shot_gk_x/_y/_distance_to_{goal,shot} columns.
    """
    missing = [c for c in _ADD_PRE_SHOT_GK_CONTEXT_REQUIRED_COLUMNS if c not in actions.columns]
    if missing:
        raise ValueError(
            f"add_pre_shot_gk_context: actions missing required columns: {sorted(missing)}. "
            f"Got: {sorted(actions.columns)}"
        )
    if lookback_seconds < 0:
        raise ValueError(f"add_pre_shot_gk_context: lookback_seconds must be >= 0, got {lookback_seconds}")
    if lookback_actions < 1:
        raise ValueError(f"add_pre_shot_gk_context: lookback_actions must be >= 1, got {lookback_actions}")

    sorted_actions = actions.sort_values(["game_id", "period_id", "action_id"], kind="mergesort").reset_index(drop=True)

    n = len(sorted_actions)
    gk_was_distributing = np.zeros(n, dtype=bool)
    gk_was_engaged = np.zeros(n, dtype=bool)
    gk_actions_in_possession = np.zeros(n, dtype=np.int64)

    # defending_gk_player_id preserves the input ``player_id`` dtype: numeric input -> float64
    # NaN-coded; object/string input (Sportec / KLOPPY_SPADL_COLUMNS variant) -> object/None.
    # This makes the helper provider-agnostic without breaking the float64-output contract for
    # the canonical SPADL_COLUMNS case (PFF / StatsBomb / Opta / Wyscout / Metrica).
    _pid = sorted_actions["player_id"]
    player_id_is_object = _pid.dtype == object or pd.api.types.is_string_dtype(_pid)
    if player_id_is_object:
        defending_gk_player_id: np.ndarray = np.full(n, None, dtype=object)
    else:
        defending_gk_player_id = np.full(n, np.nan, dtype=np.float64)

    if n == 0:
        sorted_actions["gk_was_distributing"] = gk_was_distributing
        sorted_actions["gk_was_engaged"] = gk_was_engaged
        sorted_actions["gk_actions_in_possession"] = gk_actions_in_possession
        sorted_actions["defending_gk_player_id"] = defending_gk_player_id
        return sorted_actions

    type_id = sorted_actions["type_id"].to_numpy()
    team_id = sorted_actions["team_id"].to_numpy()
    player_id = sorted_actions["player_id"].to_numpy()
    time_seconds_arr = sorted_actions["time_seconds"].to_numpy(dtype=np.float64)
    game_id = sorted_actions["game_id"].to_numpy()
    period_id = sorted_actions["period_id"].to_numpy()

    shot_type_ids = {spadlconfig.actiontype_id[name] for name in _GK_SHOT_TYPE_NAMES}
    keeper_type_ids = {spadlconfig.actiontype_id[name] for name in _GK_KEEPER_TYPE_NAMES}
    is_shot = np.isin(type_id, list(shot_type_ids))
    is_keeper = np.isin(type_id, list(keeper_type_ids))

    shot_indices = np.where(is_shot)[0]
    for shot_idx in shot_indices:
        shooter_team = team_id[shot_idx]
        shot_time = time_seconds_arr[shot_idx]
        shot_game = game_id[shot_idx]
        shot_period = period_id[shot_idx]

        window_start = max(0, shot_idx - lookback_actions)
        win = slice(window_start, shot_idx)

        same_game_period = (game_id[win] == shot_game) & (period_id[win] == shot_period)
        within_time = (shot_time - time_seconds_arr[win]) <= lookback_seconds
        in_window = same_game_period & within_time
        defending_in_window = in_window & (team_id[win] != shooter_team)

        defending_keeper_in_window = defending_in_window & is_keeper[win]
        if not defending_keeper_in_window.any():
            continue

        # Most recent defending-keeper action's player_id defines the defending GK.
        relative_indices = np.where(defending_keeper_in_window)[0]
        gk_id_raw = player_id[window_start + relative_indices[-1]]
        if pd.isna(gk_id_raw):
            # Defending keeper action is identified (in window), but its
            # player_id is NaN — caller's data does not provide enough
            # information to identify the defending GK. Leave defaults
            # (gk_was_engaged stays False, defending_gk_player_id stays NaN).
            continue

        # Preserve native dtype: numeric -> float64 cast for backward-compat with the
        # canonical SPADL_COLUMNS schema; object/string -> pass through verbatim.
        if player_id_is_object:
            gk_id_for_match: object = gk_id_raw
            gk_id_for_assign: object = gk_id_raw
        else:
            gk_id_for_match = int(gk_id_raw)
            gk_id_for_assign = float(gk_id_for_match)

        same_gk_in_window = defending_in_window & (player_id[win] == gk_id_for_match)
        gk_keeper_actions_count = int((same_gk_in_window & is_keeper[win]).sum())
        gk_was_distributing_in_window = bool((same_gk_in_window & ~is_keeper[win]).any())

        gk_was_engaged[shot_idx] = True
        gk_was_distributing[shot_idx] = gk_was_distributing_in_window
        gk_actions_in_possession[shot_idx] = gk_keeper_actions_count
        defending_gk_player_id[shot_idx] = gk_id_for_assign

    sorted_actions["gk_was_distributing"] = gk_was_distributing
    sorted_actions["gk_was_engaged"] = gk_was_engaged
    sorted_actions["gk_actions_in_possession"] = gk_actions_in_possession
    sorted_actions["defending_gk_player_id"] = defending_gk_player_id

    # PR-S21: when tracking frames supplied, lazy-import + merge GK-position columns.
    # Lazy import preserves ADR-005 § 5 contract (no module-import-time spadl→tracking cycle).
    if frames is not None:
        from silly_kicks.tracking.features import add_pre_shot_gk_position

        sorted_actions = add_pre_shot_gk_position(sorted_actions, frames)

    return sorted_actions


def _compute_possession_boundaries(
    sorted_actions: pd.DataFrame,
    *,
    max_gap_seconds: float,
    retain_on_set_pieces: bool,
    defensive_transition_types: tuple[str, ...] = (),
    merge_brief_opposing_actions: int = 0,
    brief_window_seconds: float = 0.0,
) -> np.ndarray:
    """Return a boolean mask: ``True`` at rows that start a new possession.

    Operates on a pre-sorted ``(game_id, period_id, action_id)`` DataFrame.
    Vectorized; no row-level Python iteration. Shared logic for
    :func:`add_possessions` and any future variants.

    Parameters
    ----------
    sorted_actions : pd.DataFrame
        SPADL action stream, already sorted. Must contain ``game_id``,
        ``period_id``, ``team_id``, ``time_seconds``, ``type_id``.
    max_gap_seconds : float
        Time-gap threshold (seconds) above which a new possession starts
        regardless of team.
    retain_on_set_pieces : bool
        Whether to apply the foul-then-set-piece carve-out.

    Returns
    -------
    np.ndarray
        Shape ``(n,)`` boolean array. ``True`` at rows that begin a new
        possession (game change, period change, gap timeout, or team change
        without carve-out).
    """
    n = len(sorted_actions)
    if n == 0:
        return np.zeros(0, dtype=bool)

    game_id = sorted_actions["game_id"].to_numpy()
    period_id = sorted_actions["period_id"].to_numpy()
    team_id = sorted_actions["team_id"].to_numpy()
    time_seconds = sorted_actions["time_seconds"].to_numpy()
    type_id = sorted_actions["type_id"].to_numpy()

    prev_period = np.empty(n, dtype=period_id.dtype)
    prev_team = np.empty(n, dtype=team_id.dtype)
    prev_time = np.empty(n, dtype=time_seconds.dtype)
    prev_type = np.empty(n, dtype=type_id.dtype)
    prev_period[0] = period_id[0]
    prev_team[0] = team_id[0]
    prev_time[0] = time_seconds[0]
    prev_type[0] = type_id[0]
    prev_period[1:] = period_id[:-1]
    prev_team[1:] = team_id[:-1]
    prev_time[1:] = time_seconds[:-1]
    prev_type[1:] = type_id[:-1]

    game_change = np.empty(n, dtype=bool)
    game_change[0] = True
    game_change[1:] = game_id[1:] != game_id[:-1]

    period_change_within_game = (~game_change) & (period_id != prev_period)
    gap_timeout = (~game_change) & (~period_change_within_game) & ((time_seconds - prev_time) >= max_gap_seconds)
    team_change = (~game_change) & (team_id != prev_team)

    set_piece_ids = {spadlconfig.actiontype_id[name] for name in _SET_PIECE_RESTART_TYPE_NAMES}
    foul_id = spadlconfig.actiontype_id["foul"]
    is_set_piece = np.isin(type_id, list(set_piece_ids))
    prev_is_foul = prev_type == foul_id
    set_piece_carve_out = retain_on_set_pieces & team_change & is_set_piece & prev_is_foul

    boundary = team_change & ~set_piece_carve_out

    # Rule 2 (PR-S12, 2.1.0): defensive_transition_types — listed types do
    # not trigger team-change boundaries on their own.
    if defensive_transition_types:
        defensive_ids = {spadlconfig.actiontype_id[name] for name in defensive_transition_types}
        is_defensive = np.isin(type_id, list(defensive_ids))
        boundary = boundary & ~is_defensive

    # Rule 1 (PR-S12, 2.1.0): brief-opposing-action merge. For each surviving
    # team-change boundary at row i, look ahead k=1..N rows; if any row i+k
    # has team_id == prev_team[i] (the original team has come back) within
    # the time window AND same game_id/period_id, suppress both the boundary
    # at i AND the boundary at i+k (the team-flip-back).
    if merge_brief_opposing_actions > 0 and brief_window_seconds > 0:
        suppress_at_i = np.zeros(n, dtype=bool)
        suppress_at_k = np.zeros(n, dtype=bool)
        for k in range(1, merge_brief_opposing_actions + 1):
            # Aligned look-ahead: index i sees row i+k; sentinels for last k positions.
            team_at_k = np.empty(n, dtype=team_id.dtype)
            time_at_k = np.empty(n, dtype=time_seconds.dtype)
            game_at_k = np.empty(n, dtype=game_id.dtype)
            period_at_k = np.empty(n, dtype=period_id.dtype)
            if n > k:
                team_at_k[: n - k] = team_id[k:]
                time_at_k[: n - k] = time_seconds[k:]
                game_at_k[: n - k] = game_id[k:]
                period_at_k[: n - k] = period_id[k:]
            # Sentinels for last k positions: time=+inf forces window check to fail;
            # game/period sentinels also fail the same-game-period check.
            team_at_k[n - k :] = team_id[-1]  # arbitrary; never used due to sentinels below
            time_at_k[n - k :] = np.inf
            game_at_k[n - k :] = -1
            period_at_k[n - k :] = -1

            same_game_period = (game_at_k == game_id) & (period_at_k == period_id)
            within_time = (time_at_k - time_seconds) <= brief_window_seconds
            team_back = team_at_k == prev_team
            match = boundary & same_game_period & within_time & team_back

            suppress_at_i |= match
            # Shift match by k positions to mark the team-flip-back boundary for suppression.
            shifted = np.zeros(n, dtype=bool)
            if n > k:
                shifted[k:] = match[: n - k]
            suppress_at_k |= shifted

        boundary = boundary & ~suppress_at_i & ~suppress_at_k

    return game_change | period_change_within_game | gap_timeout | boundary


@nan_safe_enrichment
def add_possessions(
    actions: pd.DataFrame,
    *,
    max_gap_seconds: float = 7.0,
    retain_on_set_pieces: bool = True,
    merge_brief_opposing_actions: int = 0,
    brief_window_seconds: float = 0.0,
    defensive_transition_types: tuple[str, ...] = (),
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
    capture additional context like merging brief opposing-team actions
    back into the containing possession). At the default
    ``max_gap_seconds=7.0`` (PR-S12, 2.1.0) and all opt-in rules disabled,
    the heuristic empirically achieves on 64 StatsBomb WorldCup-2018 matches:

        - Boundary recall: ~0.94 (worst-match 0.85).
        - Boundary precision: ~0.44 (worst-match 0.35).
        - Boundary F1: ~0.60.

    Recall is the meaningful metric for downstream consumers — possessions
    detected by the heuristic correspond to real possession changes. The
    precision gap reflects the algorithm class, not a defect. Consumers
    needing strict StatsBomb-equivalent semantics should use the native
    possession_id where available; the heuristic is a possession proxy
    for sources without one (Wyscout, Sportec, Metrica, etc.).

    Opt-in precision-improvement rules (PR-S12, 2.1.0)
    --------------------------------------------------
    Three opt-in keyword-only parameters trade precision for recall on the
    same algorithm class. Measured on 64 WC-2018 matches:

    +--------------------------------------------------+--------+--------+------+--------+
    | Setting                                          | P_mean | R_mean | F1   | R_min  |
    +==================================================+========+========+======+========+
    | (default, all rules off)                         | 0.44   | 0.94   | 0.60 | 0.85   |
    +--------------------------------------------------+--------+--------+------+--------+
    | ``defensive_transition_types=("interception",    | 0.46   | 0.92   | 0.61 | 0.85   |
    |   "clearance")``                                 |        |        |      |        |
    +--------------------------------------------------+--------+--------+------+--------+
    | ``merge_brief_opposing_actions=2,                | 0.48   | 0.91   | 0.63 | 0.84   |
    |   brief_window_seconds=2.0``                     |        |        |      |        |
    +--------------------------------------------------+--------+--------+------+--------+
    | ``merge_brief_opposing_actions=3,                | 0.53   | 0.88   | 0.66 | 0.81   |
    |   brief_window_seconds=3.0`` (R_min < 0.85)      |        |        |      |        |
    +--------------------------------------------------+--------+--------+------+--------+

    See :func:`boundary_metrics` for downstream measurement.

    NaN values in caller-supplied identifier columns (e.g. ``team_id``)
    are treated as "not identifiable" for that row's enrichment lookup;
    boundaries are determined by team-change / time-gap / set-piece rules
    that are NaN-safe by construction. See ADR-003.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL action stream. Must contain ``game_id``, ``period_id``,
        ``action_id``, ``time_seconds``, ``team_id``, ``type_id``. Other
        columns are preserved unchanged.
    max_gap_seconds : float, default 7.0
        Time-gap threshold (seconds) above which a new possession starts
        even if the team hasn't changed. Set to ``float("inf")`` to disable.

        .. versionchanged:: 2.1.0
            Default changed from 5.0 to 7.0 — empirically Pareto-optimal at
            the per-match recall floor on 64 WC-2018 matches. To restore
            1.x-2.0.x behavior, pass ``max_gap_seconds=5.0`` explicitly.
    retain_on_set_pieces : bool, default True
        Whether to apply the foul-then-set-piece carve-out (see Algorithm).
    merge_brief_opposing_actions : int, default 0
        Maximum number of consecutive opposing-team actions to merge back
        into the containing possession. Both this AND ``brief_window_seconds``
        must be > 0 to enable; both 0 disables. Recommended:
        ``merge_brief_opposing_actions=2, brief_window_seconds=2.0``.

        .. versionadded:: 2.1.0
    brief_window_seconds : float, default 0.0
        Time window (seconds) for the brief-opposing-action merge rule.
        See ``merge_brief_opposing_actions`` for activation pairing.

        .. versionadded:: 2.1.0
    defensive_transition_types : tuple[str, ...], default ()
        Action type names that should NOT trigger team-change boundaries
        on their own. Must be a subset of
        :attr:`silly_kicks.spadl.config.actiontypes`. Empty tuple disables.
        Recommended: ``("interception", "clearance")``.

        .. versionadded:: 2.1.0

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
    if merge_brief_opposing_actions < 0:
        raise ValueError(
            f"add_possessions: merge_brief_opposing_actions must be >= 0, got {merge_brief_opposing_actions}"
        )
    if brief_window_seconds < 0:
        raise ValueError(f"add_possessions: brief_window_seconds must be >= 0, got {brief_window_seconds}")
    if (merge_brief_opposing_actions > 0) != (brief_window_seconds > 0):
        raise ValueError(
            "add_possessions: merge_brief_opposing_actions and brief_window_seconds must "
            "both be > 0 to enable the brief-opposing-merge rule, or both 0 to disable. "
            f"Got merge_brief_opposing_actions={merge_brief_opposing_actions}, "
            f"brief_window_seconds={brief_window_seconds}."
        )
    invalid_defensive = [t for t in defensive_transition_types if t not in spadlconfig.actiontype_id]
    if invalid_defensive:
        raise ValueError(
            f"add_possessions: defensive_transition_types contains unknown action types: "
            f"{sorted(invalid_defensive)}. Valid types: {sorted(spadlconfig.actiontype_id.keys())}"
        )

    # Sort by canonical SPADL order. Stable sort preserves original-row order on ties.
    sorted_actions = actions.sort_values(["game_id", "period_id", "action_id"], kind="mergesort").reset_index(drop=True)

    n = len(sorted_actions)
    if n == 0:
        sorted_actions["possession_id"] = pd.Series([], dtype=np.int64)
        return sorted_actions

    new_possession_mask = _compute_possession_boundaries(
        sorted_actions,
        max_gap_seconds=max_gap_seconds,
        retain_on_set_pieces=retain_on_set_pieces,
        defensive_transition_types=defensive_transition_types,
        merge_brief_opposing_actions=merge_brief_opposing_actions,
        brief_window_seconds=brief_window_seconds,
    )

    # Per-game cumulative count of new-possession events; subtract 1 so the first row of each
    # game is possession_id=0 (game_change is True there → cumsum starts at 1 → -1 = 0).
    sorted_actions["_new_possession"] = new_possession_mask.astype(np.int64)
    sorted_actions["possession_id"] = (
        sorted_actions.groupby("game_id", sort=False)["_new_possession"].cumsum() - 1
    ).astype(np.int64)
    sorted_actions = sorted_actions.drop(columns=["_new_possession"])

    return sorted_actions


class BoundaryMetrics(TypedDict):
    """Boundary-detection metrics returned by :func:`boundary_metrics`.

    All three values are floats in ``[0.0, 1.0]``. When a denominator is
    zero (no boundaries in either input, no boundaries in the heuristic,
    or no boundaries in the native), the corresponding metric is reported
    as ``0.0`` rather than raising — callers can compute on degenerate
    sequences (empty, single-row, all-constant) without guarding.
    """

    precision: float
    recall: float
    f1: float


def boundary_metrics(
    *,
    heuristic: pd.Series,
    native: pd.Series,
) -> BoundaryMetrics:
    """Boundary precision / recall / F1 between two possession-id sequences.

    Compares two integer possession-id sequences over identical row order
    (typically: :func:`add_possessions`'s heuristic output and a provider's
    native possession_id, both on the same SPADL action stream). Reports
    where the two sequences emit possession boundaries — invariant under
    counter relabeling, since boundaries are detected as consecutive-row
    inequality regardless of the absolute id values.

    Empirical baselines on StatsBomb open-data for silly-kicks's
    :func:`add_possessions` (3 matches across Women's World Cup,
    Champions League, Premier League):

    - Recall ~0.93 — every real possession boundary is detected.
    - Precision ~0.42 — the heuristic emits ~2x more boundaries than
      StatsBomb's native annotation (the precision gap reflects the
      team-change-with-carve-outs algorithm class, not a defect).
    - F1 ~0.58.

    Recall is the meaningful regression signal. F1 conflates two
    independent signals with very different magnitudes; consumers
    should report F1 alongside but should not treat it as a primary
    metric.

    Parameters
    ----------
    heuristic : pd.Series
        Possession-id sequence from ``add_possessions`` (or any other
        heuristic). Integer-typed.
    native : pd.Series
        Provider-native possession-id sequence (e.g. StatsBomb
        ``possession``). Same length and row order as ``heuristic``.

    Returns
    -------
    BoundaryMetrics
        ``{"precision": ..., "recall": ..., "f1": ...}``. Returns ``0.0``
        for any metric whose denominator is zero (empty / single-row /
        constant sequences, or no boundaries on one side).

    Raises
    ------
    ValueError
        If ``len(heuristic) != len(native)``.

    Examples
    --------
    Validate :func:`add_possessions` against StatsBomb native::

        actions, _ = statsbomb.convert_to_actions(
            events, home_team_id=100, preserve_native=["possession"]
        )
        actions = add_possessions(actions)
        m = boundary_metrics(
            heuristic=actions["possession_id"],
            native=actions["possession"].astype("int64"),
        )
        # m["recall"] ~0.93, m["precision"] ~0.42, m["f1"] ~0.58 on
        # typical StatsBomb open-data matches.
    """
    if len(heuristic) != len(native):
        raise ValueError(
            f"boundary_metrics: heuristic and native must have the same length. "
            f"Got len(heuristic)={len(heuristic)} vs len(native)={len(native)}."
        )

    # Degenerate sizes: 0 or 1 → no consecutive pairs → no boundaries.
    n = len(heuristic)
    if n < 2:
        return BoundaryMetrics(precision=0.0, recall=0.0, f1=0.0)

    h_changes = heuristic.ne(heuristic.shift(1)).iloc[1:].to_numpy()
    n_changes = native.ne(native.shift(1)).iloc[1:].to_numpy()

    tp = int((h_changes & n_changes).sum())
    fp = int((h_changes & ~n_changes).sum())
    fn = int((~h_changes & n_changes).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return BoundaryMetrics(precision=precision, recall=recall, f1=f1)


class CoverageMetrics(TypedDict):
    """Per-action-type coverage statistics for a SPADL action stream.

    Returned by :func:`coverage_metrics` for downstream coverage validation
    and silly-kicks's own cross-provider parity regression gate.

    Attributes
    ----------
    counts : dict[str, int]
        Maps action-type name to row count. Action-type names are resolved
        from ``type_id`` via :func:`silly_kicks.spadl.config.actiontypes_df`.
        ``type_id`` values not found in the canonical spadlconfig vocabulary
        are reported under the name ``"unknown"`` (no exception raised).
    missing : list[str]
        Action types that the caller passed via ``expected_action_types`` but
        which produced zero rows in ``actions``. Returned sorted (so test
        assertions are stable). Empty list when ``expected_action_types`` is
        ``None`` or every expected type was present.
    total_actions : int
        Row count of the input ``actions`` DataFrame.
    """

    counts: dict[str, int]
    missing: list[str]
    total_actions: int


def coverage_metrics(
    *,
    actions: pd.DataFrame,
    expected_action_types: set[str] | None = None,
) -> CoverageMetrics:
    """Compute SPADL action-type coverage for an action DataFrame.

    Resolves ``type_id`` to action-type name via
    :func:`silly_kicks.spadl.config.actiontypes_df` and counts each action
    type present. When ``expected_action_types`` is provided, returns any
    of those types with zero rows under ``missing``.

    Use cases:

    1. Test discipline — assert converter X emits action types Y on a
       fixture (used by silly-kicks 1.10.0's cross-provider parity
       regression gate).
    2. Downstream validation — consumers calling silly-kicks-converted
       bronze data can verify expected coverage before downstream
       aggregation.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL action stream. Must contain ``type_id``.
    expected_action_types : set[str] or None, default ``None``
        Action type names expected to be present. Returned (sorted) as
        ``missing`` if absent. ``None`` skips the expectation check; an
        empty list is then returned for ``missing``.

    Returns
    -------
    CoverageMetrics
        ``{"counts": {...}, "missing": [...], "total_actions": N}``.

    Raises
    ------
    ValueError
        If the ``type_id`` column is missing.

    Examples
    --------
    Validate IDSSE bronze→SPADL output covers all expected action types::

        from silly_kicks.spadl import sportec, coverage_metrics
        actions, _ = sportec.convert_to_actions(
            events, home_team_id="HOME", goalkeeper_ids={"DFL-OBJ-..."}
        )
        m = coverage_metrics(
            actions=actions,
            expected_action_types={"pass", "shot", "tackle", "keeper_pick_up"},
        )
        assert not m["missing"], f"Missing action types: {m['missing']}"
    """
    if "type_id" not in actions.columns:
        raise ValueError(f"coverage_metrics: actions missing required 'type_id' column. Got: {sorted(actions.columns)}")

    n = len(actions)
    counts: dict[str, int] = {}
    if n > 0:
        # Reverse map: id -> name. Built from spadlconfig.actiontypes (single
        # source of truth). Out-of-vocab ids report as "unknown".
        id_to_name = {i: name for i, name in enumerate(spadlconfig.actiontypes)}
        type_id_arr = actions["type_id"].to_numpy()
        # Tally with deterministic insertion order (numpy iteration order on
        # the input array). Equivalent to pd.value_counts but preserves
        # first-seen ordering for stable user-facing output.
        for tid in type_id_arr:
            if pd.isna(tid):
                name = "unknown"
            else:
                name = id_to_name.get(int(tid), "unknown")
            counts[name] = counts.get(name, 0) + 1

    expected = set(expected_action_types) if expected_action_types else set()
    missing = sorted(expected - set(counts.keys())) if expected else []

    return CoverageMetrics(counts=counts, missing=missing, total_actions=n)


@nan_safe_enrichment
def add_names(actions: pd.DataFrame) -> pd.DataFrame:
    """Add the type name, result name and bodypart name to a SPADL dataframe.

    All columns not in the SPADL schema are preserved unchanged.

    NaN values in caller-supplied identifier columns (e.g. ``type_id``)
    are treated as "not identifiable" for that row's enrichment lookup;
    NaN type/result/bodypart ids produce NaN name outputs via the
    underlying merge. See ADR-003.

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

    Examples
    --------
    Append name columns for human-readable diagnostics::

        actions, _ = statsbomb.convert_to_actions(events, home_team_id=100)
        actions = add_names(actions)
        # actions now has type_name / result_name / bodypart_name string columns:
        actions[["type_name", "result_name", "bodypart_name"]].head()
    """
    return (
        actions.drop(columns=["type_name", "result_name", "bodypart_name"], errors="ignore")
        .merge(spadlconfig.actiontypes_df(), how="left")  # type: ignore[reportOptionalMemberAccess]
        .merge(spadlconfig.results_df(), how="left")
        .merge(spadlconfig.bodyparts_df(), how="left")
        .set_index(actions.index)
    )


def play_left_to_right(actions: pd.DataFrame, home_team_id: int) -> pd.DataFrame:
    """Mirror away-team rows from absolute-frame to SPADL LTR convention.

    Public boundary helper for callers who hold actions in
    absolute-frame-home-right convention (home team plays LTR, away team plays
    RTL in absolute coordinates) and want to convert them to canonical SPADL
    "all teams attack left-to-right" convention. Mirrors
    ``(start_x, start_y) / (end_x, end_y)`` of every row whose
    ``team_id != home_team_id``.

    .. versionchanged:: 3.0.0
        ADR-006 / PR-S22: every silly-kicks SPADL converter now outputs
        canonical SPADL LTR directly via :func:`to_spadl_ltr` -- you should NOT
        call this function on output from ``statsbomb.convert_to_actions`` etc.
        The function is retained as a public utility for callers who load
        actions from outside silly-kicks (e.g. raw socceraction output).

    Parameters
    ----------
    actions : pd.DataFrame
        Actions in absolute-frame-home-right convention.
    home_team_id : int
        ID of the home team.

    Returns
    -------
    pd.DataFrame
        Actions with all rows oriented left-to-right.

    See Also
    --------
    silly_kicks.spadl.to_spadl_ltr : Canonical normalizer used inside converters
        (supports possession-perspective, absolute-frame, and per-period inputs).
    silly_kicks.vaep.features.play_left_to_right : Equivalent for gamestates.

    Examples
    --------
    Convert externally-loaded absolute-frame actions to SPADL LTR::

        from silly_kicks.spadl import play_left_to_right

        ltr = play_left_to_right(absolute_frame_actions, home_team_id=100)
        # All away-team actions now have flipped (start_x, start_y) / (end_x, end_y).
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
        if dtype in _PANDAS_EXTENSION_DTYPES:
            # pandas-stubs's ``astype`` overload set rejects bare extension-dtype
            # strings (e.g. ``"Int64"``); the runtime accepts them via dispatch
            # to ``ExtensionDtype.construct_from_string``.
            result[col] = result[col].astype(dtype)  # type: ignore[reportCallIssue]
        else:
            result[col] = result[col].astype(np.dtype(dtype))

    # Direction-of-play invariant assertion (PR-S22 / ADR-006). Off by default
    # for runtime cost; enabled in CI via SILLY_KICKS_ASSERT_INVARIANTS=1. Asserts
    # SPADL canonical convention -- both teams' shots cluster at high-x. Defers
    # silently on fixtures with fewer than 10 shots per team to avoid noise from
    # small synthetic event sets and from the documented kloppy metrica
    # fixture pairing issue (events from a different match than the metadata,
    # per tests/datasets/kloppy/README.md). Realistic match data has 10-25
    # shots per team per match. Atomic-SPADL output uses `x`/`y` not
    # `start_x`/`start_y`; the assertion is skipped for that schema -- atomic
    # actions inherit orientation from the SPADL convert_to_atomic input, which
    # was already validated upstream.
    if (
        os.environ.get("SILLY_KICKS_ASSERT_INVARIANTS") == "1"
        and "type_id" in result.columns
        and "start_x" in result.columns
    ):
        shots = result[result["type_id"] == spadlconfig.actiontype_id["shot"]]
        if len(shots) > 0 and shots["team_id"].nunique() >= 2:
            by_team = shots.groupby("team_id")["start_x"].agg(["count", "mean"])
            reliable = by_team[by_team["count"] >= 10]
            if len(reliable) >= 2 and not (reliable["mean"] > spadlconfig.field_length / 2).all():
                raise AssertionError(
                    f"Direction-of-play invariant violated: not all teams attacking high-x. "
                    f"Per-team shot stats (count, mean_x): {reliable.to_dict('index')}. "
                    f"Expected all mean_x > {spadlconfig.field_length / 2:.1f}. See ADR-006."
                )
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

    Examples
    --------
    Validate a converter's output conforms to the SPADL schema::

        actions, _ = statsbomb.convert_to_actions(events, home_team_id=100)
        validate_spadl(actions)  # raises ValueError on missing columns;
                                 # warns on dtype mismatches.
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
