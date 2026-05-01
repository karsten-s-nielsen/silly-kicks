"""Utility functions for working with Atomic-SPADL dataframes."""

import warnings
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from silly_kicks._nan_safety import nan_safe_enrichment
from silly_kicks.spadl.utils import CoverageMetrics

from . import config as spadlconfig
from .schema import ATOMIC_SPADL_COLUMNS

if TYPE_CHECKING:
    # Type-only import for pyright fidelity. Runtime call site lazy-imports per ADR-005 § 5.
    from silly_kicks.atomic.tracking.features import (  # noqa: F401
        add_pre_shot_gk_position as _atomic_tracking_add_pre_shot_gk_position,
    )

_ADD_POSSESSIONS_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "period_id",
    "action_id",
    "time_seconds",
    "team_id",
    "type_id",
)

_ATOMIC_SET_PIECE_RESTART_TYPE_NAMES: Final[frozenset[str]] = frozenset({"freekick", "corner", "throw_in", "goalkick"})
"""Atomic-SPADL action types that represent a stoppage-restart taken by the
team that won possession (or had possession before the stoppage). Note the
collapsed names: standard SPADL ``corner_short`` / ``corner_crossed`` →
``corner`` and ``freekick_short`` / ``freekick_crossed`` / ``shot_freekick`` →
``freekick`` (per the ``_simplify`` step in ``convert_to_atomic``)."""

_TRANSPARENT_TYPE_NAMES: Final[frozenset[str]] = frozenset({"yellow_card", "red_card"})
"""Atomic-only synthetic action types that are emitted as follow-ups to fouls
and do NOT represent the team in possession. They inherit the surrounding
possession state and never trigger a possession boundary by themselves."""

_ADD_GK_ROLE_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "period_id",
    "action_id",
    "team_id",
    "player_id",
    "type_id",
    "x",
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
    """Tag each Atomic-SPADL action with the goalkeeper's role context.

    Atomic-SPADL counterpart to :func:`silly_kicks.spadl.utils.add_gk_role`,
    with one atomic-specific adaptation: reads ``x`` (NOT ``start_x``) for the
    penalty-area threshold check. Same five role categories.

    Categorisation rules
    --------------------
    For each row, ``gk_role`` is assigned by precedence (first match wins):

    1. ``sweeping`` — any ``keeper_*`` action with
       ``x > penalty_area_x_threshold``.

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

    NaN values in caller-supplied identifier columns (e.g. ``player_id``)
    are treated as "not identifiable" for that row's enrichment lookup;
    downstream rows behave as if no identifier were present. See ADR-003.

    Parameters
    ----------
    actions : pd.DataFrame
        Atomic-SPADL action stream. Must contain ``game_id``, ``period_id``,
        ``action_id``, ``team_id``, ``player_id``, ``type_id``, ``x``.
        Other columns are preserved unchanged.
    penalty_area_x_threshold : float, default 16.5
        Atomic-SPADL x-coordinate (metres from own goal line) above which a
        keeper action is classified as ``sweeping`` rather than the
        type-specific role.
    distribution_lookback_actions : int, default 1
        Number of preceding actions to consider when detecting a
        non-keeper action by the same player as ``distribution``.
    goalkeeper_ids : set, optional
        When provided, distribution-detection extends beyond strict
        ``same_player`` matching to also tag rows where:

        - The current row's ``player_id`` is in ``goalkeeper_ids`` AND the
          preceding action (within ``distribution_lookback_actions`` steps,
          same ``team_id`` and ``game_id``) was a keeper-type action.
        - Both the current row's and the preceding action's ``player_id``
          are NaN AND the ``team_id`` matches AND the preceding action was
          keeper-type.

        Opting in via this parameter signals that the caller accepts the
        coarser heuristic. When ``None`` (default), only strict
        ``same_player`` matching applies — byte-for-byte compatible with
        pre-2.5.0 behavior. See standard-SPADL ``add_gk_role`` docstring
        and ADR-003 for the full semantics.

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

    Examples
    --------
    Tag GK roles after atomic conversion::

        from silly_kicks.atomic.spadl import convert_to_atomic
        from silly_kicks.atomic.spadl.utils import add_gk_role

        atomic = convert_to_atomic(actions)
        atomic = add_gk_role(atomic)
        # atomic["gk_role"] is now a Categorical with 5 categories + None.
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
    x = sorted_actions["x"].to_numpy(dtype=np.float64)
    player_id = sorted_actions["player_id"]
    game_id = sorted_actions["game_id"]

    save_id = spadlconfig.actiontype_id["keeper_save"]
    claim_id = spadlconfig.actiontype_id["keeper_claim"]
    punch_id = spadlconfig.actiontype_id["keeper_punch"]
    pickup_id = spadlconfig.actiontype_id["keeper_pick_up"]

    is_save = type_id == save_id
    is_claim = type_id == claim_id
    is_punch = type_id == punch_id
    is_pickup = type_id == pickup_id
    is_keeper = is_save | is_claim | is_punch | is_pickup
    is_outside_box = x > penalty_area_x_threshold
    is_sweeping = is_keeper & is_outside_box

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

    role = np.full(n, None, dtype=object)
    role[is_save & ~is_sweeping] = "shot_stopping"
    role[(is_claim | is_punch) & ~is_sweeping] = "cross_collection"
    role[is_pickup & ~is_sweeping] = "pick_up"
    role[is_sweeping] = "sweeping"
    role[is_distribution] = "distribution"

    sorted_actions["gk_role"] = pd.Categorical(role, categories=list(_GK_ROLE_CATEGORIES))
    return sorted_actions


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
    """Assign a per-match possession-sequence integer to each Atomic-SPADL action.

    Atomic-SPADL counterpart to :func:`silly_kicks.spadl.utils.add_possessions`,
    with two atomic-specific adaptations:

    1. Set-piece restart names match the post-collapse atomic types: ``corner``
       (NOT ``corner_short`` / ``corner_crossed``) and ``freekick`` (NOT
       ``freekick_short`` / ``freekick_crossed`` / ``shot_freekick``).
       ``throw_in`` and ``goalkick`` are not collapsed.
    2. ``yellow_card`` and ``red_card`` synthetic atomic rows are
       "transparent" to boundary detection — they never trigger a possession
       boundary on their own, and the carve-out for set-piece-after-foul fires
       across them (chain ``foul → yellow_card → freekick`` retains).

    Algorithm
    ---------
    Actions are sorted by ``(game_id, period_id, action_id)``. A possession
    boundary is emitted (on non-card actions) when ANY of:

    1. ``game_id`` changes (counter resets to 0).
    2. ``period_id`` changes within the same game (counter increments).
    3. The time gap to the previous non-card action is ``>= max_gap_seconds``.
    4. ``team_id`` changes from the previous non-card action, EXCEPT when
       ``retain_on_set_pieces=True`` AND the current action is a set-piece
       restart (``corner``, ``freekick``, ``throw_in``, ``goalkick``) AND
       the previous non-card action was a foul. In that case the team-change
       is NOT a possession boundary — the team that won the foul resumes its
       possession.

    Card rows (yellow_card / red_card) inherit their possession_id from the
    surrounding non-card context via forward-fill within ``game_id``.

    Empirical baselines on 64 StatsBomb WorldCup-2018 matches at default
    ``max_gap_seconds=7.0`` (rules off): boundary recall ~0.94, precision
    ~0.44, F1 ~0.60. Same algorithm class as the standard counterpart;
    expected behavior parity (atomic card transparency aside).

    Opt-in precision-improvement rules (PR-S12, 2.1.0) follow the same
    semantics as the standard counterpart's. Recommended settings carry
    over: ``defensive_transition_types=("interception", "clearance")`` for
    the +5pp-precision conservative case;
    ``merge_brief_opposing_actions=2, brief_window_seconds=2.0`` for the
    +7pp-precision moderate case (see standard ``add_possessions``
    docstring's tradeoff table).

    NaN values in caller-supplied identifier columns (e.g. ``team_id``)
    are treated as "not identifiable" for that row's enrichment lookup;
    boundaries are NaN-safe by construction. See ADR-003.

    Parameters
    ----------
    actions : pd.DataFrame
        Atomic-SPADL action stream. Must contain ``game_id``, ``period_id``,
        ``action_id``, ``time_seconds``, ``team_id``, ``type_id``. Other
        columns are preserved unchanged.
    max_gap_seconds : float, default 7.0
        Time-gap threshold (seconds) above which a new possession starts
        even if the team hasn't changed. Set to ``float("inf")`` to disable.

        .. versionchanged:: 2.1.0
            Default changed from 5.0 to 7.0 — mirrors the standard
            ``add_possessions`` change. To restore 1.x-2.0.x behavior,
            pass ``max_gap_seconds=5.0`` explicitly.
    retain_on_set_pieces : bool, default True
        Whether to apply the foul-then-set-piece carve-out (see Algorithm).
    merge_brief_opposing_actions : int, default 0
        Maximum number of consecutive opposing-team actions to merge back
        into the containing possession. Both this AND ``brief_window_seconds``
        must be > 0 to enable; both 0 disables. Atomic action vocabulary
        is used for type checks. Card rows (yellow/red) are transparent
        to the look-ahead — they inherit possession context.

        .. versionadded:: 2.1.0
    brief_window_seconds : float, default 0.0
        Time window (seconds) for the brief-opposing-action merge rule.
        See ``merge_brief_opposing_actions`` for activation pairing.

        .. versionadded:: 2.1.0
    defensive_transition_types : tuple[str, ...], default ()
        Action type names that should NOT trigger team-change boundaries
        on their own. Must be a subset of
        :attr:`silly_kicks.atomic.spadl.config.actiontypes` (atomic
        vocabulary; differs from standard SPADL on collapsed set-piece
        names but retains ``interception`` / ``clearance`` / ``tackle`` /
        ``bad_touch``).

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
    Reconstruct possessions for any provider's atomic-SPADL output::

        from silly_kicks.atomic.spadl import convert_to_atomic
        from silly_kicks.atomic.spadl.utils import add_possessions

        actions, _ = statsbomb.convert_to_actions(events, home_team_id=100)
        atomic = convert_to_atomic(actions)
        atomic = add_possessions(atomic)
        # atomic["possession_id"] now contains a per-match possession sequence.
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

    sorted_actions = actions.sort_values(["game_id", "period_id", "action_id"], kind="mergesort").reset_index(drop=True)

    n = len(sorted_actions)
    if n == 0:
        sorted_actions["possession_id"] = pd.Series([], dtype=np.int64)
        return sorted_actions

    type_id_arr = sorted_actions["type_id"].to_numpy()
    transparent_ids = {spadlconfig.actiontype_id[name] for name in _TRANSPARENT_TYPE_NAMES}
    is_card = np.isin(type_id_arr, list(transparent_ids))

    if not is_card.any():
        # Fast path: no cards → identical algorithm to standard SPADL.
        return _compute_possessions(
            sorted_actions,
            max_gap_seconds,
            retain_on_set_pieces,
            merge_brief_opposing_actions=merge_brief_opposing_actions,
            brief_window_seconds=brief_window_seconds,
            defensive_transition_types=defensive_transition_types,
        )

    # Slow path: drop cards, compute boundaries on the reduced subset,
    # then forward-fill card rows within game.
    non_card_idx = np.where(~is_card)[0]
    non_card_subset = sorted_actions.iloc[non_card_idx].reset_index(drop=True)
    non_card_with_pids = _compute_possessions(
        non_card_subset,
        max_gap_seconds,
        retain_on_set_pieces,
        merge_brief_opposing_actions=merge_brief_opposing_actions,
        brief_window_seconds=brief_window_seconds,
        defensive_transition_types=defensive_transition_types,
    )

    pids_full = pd.array([pd.NA] * n, dtype="Int64")
    pids_full[non_card_idx] = non_card_with_pids["possession_id"].to_numpy()
    pids_series = pd.Series(pids_full, index=sorted_actions.index)
    pids_series = pids_series.groupby(sorted_actions["game_id"], sort=False).ffill()
    pids_series = pids_series.groupby(sorted_actions["game_id"], sort=False).bfill()
    pids_series = pids_series.fillna(0).astype(np.int64)

    sorted_actions["possession_id"] = pids_series.to_numpy()
    return sorted_actions


def _compute_possessions(
    sorted_actions: pd.DataFrame,
    max_gap_seconds: float,
    retain_on_set_pieces: bool,
    *,
    merge_brief_opposing_actions: int = 0,
    brief_window_seconds: float = 0.0,
    defensive_transition_types: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Compute possession_id on a card-free, pre-sorted Atomic-SPADL frame.

    Mirrors the boundary logic of :func:`silly_kicks.spadl.utils.add_possessions`,
    using atomic-collapsed set-piece names + atomic action vocabulary for the
    PR-S12 opt-in rules. Mutates and returns *sorted_actions*.
    """
    n = len(sorted_actions)
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

    set_piece_ids = {spadlconfig.actiontype_id[name] for name in _ATOMIC_SET_PIECE_RESTART_TYPE_NAMES}
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

    # Rule 1 (PR-S12, 2.1.0): brief-opposing-action merge. Vectorized look-ahead.
    if merge_brief_opposing_actions > 0 and brief_window_seconds > 0:
        suppress_at_i = np.zeros(n, dtype=bool)
        suppress_at_k = np.zeros(n, dtype=bool)
        for k in range(1, merge_brief_opposing_actions + 1):
            team_at_k = np.empty(n, dtype=team_id.dtype)
            time_at_k = np.empty(n, dtype=time_seconds.dtype)
            game_at_k = np.empty(n, dtype=game_id.dtype)
            period_at_k = np.empty(n, dtype=period_id.dtype)
            if n > k:
                team_at_k[: n - k] = team_id[k:]
                time_at_k[: n - k] = time_seconds[k:]
                game_at_k[: n - k] = game_id[k:]
                period_at_k[: n - k] = period_id[k:]
            team_at_k[n - k :] = team_id[-1]
            time_at_k[n - k :] = np.inf
            game_at_k[n - k :] = -1
            period_at_k[n - k :] = -1

            same_game_period = (game_at_k == game_id) & (period_at_k == period_id)
            within_time = (time_at_k - time_seconds) <= brief_window_seconds
            team_back = team_at_k == prev_team
            match = boundary & same_game_period & within_time & team_back

            suppress_at_i |= match
            shifted = np.zeros(n, dtype=bool)
            if n > k:
                shifted[k:] = match[: n - k]
            suppress_at_k |= shifted

        boundary = boundary & ~suppress_at_i & ~suppress_at_k

    new_possession_mask = game_change | period_change_within_game | gap_timeout | boundary

    sorted_actions["_new_possession"] = new_possession_mask.astype(np.int64)
    sorted_actions["possession_id"] = (
        sorted_actions.groupby("game_id", sort=False)["_new_possession"].cumsum() - 1
    ).astype(np.int64)
    sorted_actions = sorted_actions.drop(columns=["_new_possession"])
    return sorted_actions


_ADD_GK_DISTRIBUTION_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "period_id",
    "action_id",
    "team_id",
    "player_id",
    "type_id",
    "x",
    "y",
    "dx",
    "dy",
)

_ATOMIC_GK_LAUNCH_PASS_TYPE_NAMES: Final[frozenset[str]] = frozenset({"pass", "goalkick", "freekick"})
"""Atomic action types that qualify as GK launches when the distance exceeds
``long_threshold``. Standard SPADL's ``freekick_short`` / ``freekick_crossed``
collapse to ``freekick`` in atomic — the helper treats the collapsed type as
a launch candidate. Excludes ``clearance``, ``throw_in``, ``cross``."""

_GK_PASS_LENGTH_CATEGORIES: Final[tuple[str, ...]] = ("short", "medium", "long")

_PITCH_LENGTH_M: Final[float] = 105.0
_PITCH_WIDTH_M: Final[float] = 68.0

_ATOMIC_PASS_FAIL_TYPE_NAMES: Final[frozenset[str]] = frozenset({"interception", "out", "offside"})
"""Atomic-only follow-up action types that mark a preceding pass as failed."""


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

    Atomic-SPADL counterpart to
    :func:`silly_kicks.spadl.utils.add_gk_distribution_metrics`. Three
    atomic-specific adaptations:

    - Length is ``sqrt(dx² + dy²)`` from atomic's ``(dx, dy)`` columns.
    - xT delta is from ``(x, y)`` (start) to ``(x + dx, y + dy)`` (end).
    - Pass success is detected from the FOLLOWING atomic action by row
      index: ``receival`` → success; ``interception`` / ``out`` /
      ``offside`` → failure; no following action (last row of
      game/period) → conservative failure (xT delta = NaN).

    Operates only on rows where ``gk_role == "distribution"`` (per
    :func:`add_gk_role`). Adds four columns:

    - ``gk_pass_length_m`` — Euclidean distance in metres
      (NaN on non-distribution rows).
    - ``gk_pass_length_class`` — Categorical ``{"short", "medium", "long"}``
      keyed off ``short_threshold`` / ``long_threshold``.
    - ``is_launch`` — bool. True iff length > ``long_threshold`` AND action
      is a deliberate-distribution pass type (``pass``, ``goalkick``,
      ``freekick``). False everywhere else.
    - ``gk_xt_delta`` — float xT-grid delta from start zone to end zone,
      computed only when ``xt_grid`` is provided AND the pass succeeded.
      NaN otherwise.

    NaN values in caller-supplied identifier columns (e.g. ``player_id``)
    are treated as "not identifiable" for that row's enrichment lookup.
    Rows with NaN coordinates are excluded from xT-delta zone-binning
    (``gk_xt_delta`` is NaN for those rows). See ADR-003.

    Parameters
    ----------
    actions : pd.DataFrame
        Atomic-SPADL action stream. Required columns: ``game_id``,
        ``period_id``, ``action_id``, ``team_id``, ``player_id``,
        ``type_id``, ``x``, ``y``, ``dx``, ``dy``. ``gk_role`` is either
        present (used directly) or computed via :func:`add_gk_role`.
    xt_grid : np.ndarray, optional
        12x8 Expected Threat grid (atomic coordinate system, rows = x-bins,
        cols = y-bins). When omitted, ``gk_xt_delta`` is NaN for all rows.
    short_threshold : float, default 32.0
    long_threshold : float, default 60.0
    require_gk_role : bool, default True

    Returns
    -------
    pd.DataFrame
        Sorted copy of ``actions`` with the four columns appended.

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
    Compute GK distribution metrics on an atomic-SPADL stream::

        from silly_kicks.atomic.spadl.utils import add_gk_distribution_metrics

        atomic = add_gk_distribution_metrics(atomic, long_threshold=60.0)
        # Filter to launches:
        launches = atomic[atomic["is_launch"]]
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

    is_distribution = (actions["gk_role"] == "distribution").to_numpy()

    x = actions["x"].to_numpy(dtype=np.float64)
    y = actions["y"].to_numpy(dtype=np.float64)
    dx = actions["dx"].to_numpy(dtype=np.float64)
    dy = actions["dy"].to_numpy(dtype=np.float64)
    raw_length = np.sqrt(dx * dx + dy * dy)
    length_m = np.where(is_distribution, raw_length, np.nan)

    short_mask = is_distribution & (raw_length < short_threshold)
    long_mask = is_distribution & (raw_length > long_threshold)
    medium_mask = is_distribution & ~short_mask & ~long_mask

    length_class = np.full(n, None, dtype=object)
    length_class[short_mask] = "short"
    length_class[medium_mask] = "medium"
    length_class[long_mask] = "long"

    type_id = actions["type_id"].to_numpy()
    launch_type_ids = {spadlconfig.actiontype_id[name] for name in _ATOMIC_GK_LAUNCH_PASS_TYPE_NAMES}
    is_launch_type = np.isin(type_id, list(launch_type_ids))
    is_launch = is_distribution & is_launch_type & (raw_length > long_threshold)

    # Pass-success detection: success iff next row is `receival` in the same
    # (game, period). Anything else — explicit failure follow-ups
    # (``_ATOMIC_PASS_FAIL_TYPE_NAMES``: interception / out / offside), no
    # following row (last row of game/period), or a cross-period boundary —
    # is treated as failure for xT purposes (``gk_xt_delta = NaN``).
    receival_id = spadlconfig.actiontype_id["receival"]
    next_type = np.full(n, -1, dtype=type_id.dtype)
    next_type[:-1] = type_id[1:]

    game_id = actions["game_id"].to_numpy()
    period_id = actions["period_id"].to_numpy()
    next_game = np.full(n, -1, dtype=game_id.dtype)
    next_game[:-1] = game_id[1:]
    next_period = np.full(n, -1, dtype=period_id.dtype)
    next_period[:-1] = period_id[1:]
    same_game_period = (next_game == game_id) & (next_period == period_id)

    is_success = same_game_period & (next_type == receival_id)

    xt_delta = np.full(n, np.nan, dtype=np.float64)
    if xt_grid is not None:
        end_x = x + dx
        end_y = y + dy
        # NaN coordinates would crash the .astype(int) zone-binning below.
        # Filter to rows where all four coords are finite. Non-finite rows
        # leave xt_delta at NaN (default initialization).
        coords_finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(end_x) & np.isfinite(end_y)
        eligible = is_distribution & is_success & coords_finite
        if eligible.any():
            zone_x_start = np.clip((x[eligible] / (_PITCH_LENGTH_M / 12.0)).astype(int), 0, 11)
            zone_y_start = np.clip((y[eligible] / (_PITCH_WIDTH_M / 8.0)).astype(int), 0, 7)
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

_ATOMIC_SHOT_TYPE_NAMES: Final[frozenset[str]] = frozenset({"shot", "shot_penalty"})
"""Atomic-SPADL shot types. Note: ``shot_freekick`` is collapsed into
``freekick`` (which also includes pass-class freekicks) in atomic, so the
shot/pass distinction on free kicks is lost. Atomic users opt in to that
lossy collapse — the helper treats ``freekick`` as a non-shot row."""

_ATOMIC_KEEPER_TYPE_NAMES: Final[frozenset[str]] = frozenset(
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

    Atomic-SPADL counterpart to
    :func:`silly_kicks.spadl.utils.add_pre_shot_gk_context`. Same algorithm;
    atomic recognises only ``shot`` and ``shot_penalty`` as shot rows
    (``shot_freekick`` is collapsed into ``freekick`` in atomic, mixing
    pass-class and shot-class freekicks — the helper does not attempt to
    disambiguate).

    For every shot row, looks back up to ``lookback_actions`` rows OR up to
    ``lookback_seconds`` seconds in the same ``(game_id, period_id)`` —
    whichever is smaller — to identify the defending GK and characterise
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
    via the ``silly_kicks.atomic.tracking.features`` canonical compute.
    When ``frames=None`` (default), behavior is bit-identical to silly-kicks 2.8.0.

    Non-shot rows receive default values (False / 0 / NaN).

    .. note::

        This helper is genuinely novel — no published OSS / academic
        equivalent surfaces a goalkeeper's pre-shot activity context as
        explicit per-shot features.

    NaN values in caller-supplied identifier columns (e.g. ``player_id``)
    are treated as "not identifiable" for that row's enrichment lookup;
    when the most-recent defending-keeper-action's ``player_id`` is NaN,
    the shot's GK-context columns receive their per-row default
    (``gk_was_engaged=False``, ``defending_gk_player_id=NaN``). See ADR-003.

    Parameters
    ----------
    actions : pd.DataFrame
        Atomic-SPADL action stream. Required columns: ``game_id``,
        ``period_id``, ``action_id``, ``team_id``, ``player_id``,
        ``type_id``, ``time_seconds``.
    frames : pd.DataFrame | None, default None
        Long-form tracking frames matching ``TRACKING_FRAMES_COLUMNS``.
        When supplied, enables 4 GK-position + 4 linkage-provenance output
        columns. PR-S21 (silly-kicks 2.9.0+).
    lookback_seconds : float, default 10.0
    lookback_actions : int, default 5

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

    - Anzer, G., & Bauer, P. (2021), "A goal scoring probability model for
      shots based on synchronized positional and event data in football
      and futsal." Frontiers in Sports and Active Living, 3, 624475 —
      defending-GK position as xG feature; basis of the 4 GK-position
      columns when ``frames`` is supplied.

    Examples
    --------
    Tag pre-shot defending-GK context on an atomic-SPADL stream
    (events-only, silly-kicks 2.8.0 backward-compat)::

        from silly_kicks.atomic.spadl.utils import add_pre_shot_gk_context
        atomic = add_pre_shot_gk_context(atomic, lookback_seconds=10.0)
        engaged_shots = atomic[atomic["gk_was_engaged"]]

    Events + tracking path (silly-kicks 2.9.0+)::

        from silly_kicks.tracking import sportec
        frames, _ = sportec.convert_to_frames(raw, home_team_id="DFL-CLU-A", home_team_start_left=True)
        atomic = add_pre_shot_gk_context(atomic, frames=frames)
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

    # defending_gk_player_id preserves the input ``player_id`` dtype: numeric -> float64
    # NaN-coded; object/string (Sportec / KLOPPY_SPADL_COLUMNS) -> object/None.
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

    shot_type_ids = {spadlconfig.actiontype_id[name] for name in _ATOMIC_SHOT_TYPE_NAMES}
    keeper_type_ids = {spadlconfig.actiontype_id[name] for name in _ATOMIC_KEEPER_TYPE_NAMES}
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

        relative_indices = np.where(defending_keeper_in_window)[0]
        gk_id_raw = player_id[window_start + relative_indices[-1]]
        if pd.isna(gk_id_raw):
            # Defending keeper action is identified (in window), but its
            # player_id is NaN — caller's data does not provide enough
            # information to identify the defending GK. Leave defaults
            # (gk_was_engaged stays False, defending_gk_player_id stays NaN).
            continue

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

    # PR-S21: when tracking frames supplied, lazy-import + merge GK-position columns
    # from the atomic-namespace canonical compute. Lazy import preserves ADR-005 § 5.
    if frames is not None:
        from silly_kicks.atomic.tracking.features import add_pre_shot_gk_position

        sorted_actions = add_pre_shot_gk_position(sorted_actions, frames)

    return sorted_actions


@nan_safe_enrichment
def add_names(actions: pd.DataFrame) -> pd.DataFrame:
    """Add the type name and bodypart name to an Atomic-SPADL dataframe.

    All columns not in the Atomic-SPADL schema are preserved unchanged.

    NaN values in caller-supplied identifier columns (e.g. ``type_id``)
    are treated as "not identifiable" for that row's enrichment lookup;
    NaN type/bodypart ids produce NaN name outputs via the underlying
    merge. See ADR-003.

    Parameters
    ----------
    actions : pd.DataFrame
        An Atomic-SPADL dataframe.

    Returns
    -------
    pd.DataFrame
        The original dataframe with 'type_name' and 'bodypart_name' appended.

    Examples
    --------
    Append name columns for human-readable diagnostics on atomic-SPADL::

        from silly_kicks.atomic.spadl.utils import add_names

        atomic = add_names(atomic)
        atomic[["type_name", "bodypart_name"]].head()
    """
    return (
        actions.drop(columns=["type_name", "bodypart_name"], errors="ignore")
        .merge(spadlconfig.actiontypes_df(), how="left")  # type: ignore[reportOptionalMemberAccess]
        .merge(spadlconfig.bodyparts_df(), how="left")
        .set_index(actions.index)
    )


def validate_atomic_spadl(
    df: pd.DataFrame,
    schema: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Validate that a DataFrame conforms to the Atomic-SPADL schema.

    Atomic-SPADL counterpart to :func:`silly_kicks.spadl.utils.validate_spadl`.
    Returns the input unchanged for chaining; emits a warning for any column
    whose dtype does not match the declared schema.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    schema : dict[str, str], optional
        Column name to dtype mapping. Defaults to ``ATOMIC_SPADL_COLUMNS``.

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
    Validate an atomic converter's output conforms to the Atomic-SPADL schema::

        from silly_kicks.atomic.spadl.utils import validate_atomic_spadl

        validate_atomic_spadl(atomic)  # raises ValueError on missing columns
    """
    if schema is None:
        schema = ATOMIC_SPADL_COLUMNS
    missing = set(schema.keys()) - set(df.columns)
    if missing:
        raise ValueError(f"Missing Atomic-SPADL columns: {sorted(missing)}")
    for col, expected_dtype in schema.items():
        actual_dtype = str(df[col].dtype)
        if actual_dtype != expected_dtype:
            warnings.warn(
                f"Column '{col}' has dtype '{actual_dtype}', expected '{expected_dtype}'",
                stacklevel=2,
            )
    return df


def play_left_to_right(actions: pd.DataFrame, home_team_id: int) -> pd.DataFrame:
    """Perform all action in the same playing direction.

    This changes the location of each action, such that all actions
    are performed as if the team that executes the action plays from left to
    right.

    Parameters
    ----------
    actions : pd.DataFrame
        The SPADL actins of a game.
    home_team_id : int
        The ID of the home team.

    Returns
    -------
    list(pd.DataFrame)
        All actions performed left to right.

    See Also
    --------
    silly_kicks.atomic.vaep.features.play_left_to_right : For transforming gamestates.

    Examples
    --------
    Mirror atomic actions to a single direction (left-to-right per the home team)::

        from silly_kicks.atomic.spadl.utils import play_left_to_right

        ltr = play_left_to_right(atomic, home_team_id=100)
        # All away-team actions now have flipped (x, y) and (dx, dy).
    """
    ltr_actions = actions.copy()
    away_idx = actions.team_id != home_team_id
    ltr_actions.loc[away_idx, "x"] = spadlconfig.field_length - actions[away_idx]["x"].values  # type: ignore[reportAttributeAccessIssue]
    ltr_actions.loc[away_idx, "y"] = spadlconfig.field_width - actions[away_idx]["y"].values  # type: ignore[reportAttributeAccessIssue]
    ltr_actions.loc[away_idx, "dx"] = -actions[away_idx]["dx"].values  # type: ignore[reportAttributeAccessIssue]
    ltr_actions.loc[away_idx, "dy"] = -actions[away_idx]["dy"].values  # type: ignore[reportAttributeAccessIssue]
    return ltr_actions


def coverage_metrics(
    *,
    actions: pd.DataFrame,
    expected_action_types: set[str] | None = None,
) -> CoverageMetrics:
    """Compute Atomic-SPADL action-type coverage for an atomic action DataFrame.

    Atomic-SPADL counterpart to :func:`silly_kicks.spadl.utils.coverage_metrics`.
    Resolves ``type_id`` to action-type name via the **atomic** vocabulary
    (:func:`silly_kicks.atomic.spadl.config.actiontypes_df` — 33 types) and
    counts each action type present. When ``expected_action_types`` is
    provided, returns any of those types with zero rows under ``missing``.

    The return type is the standard :class:`silly_kicks.spadl.utils.CoverageMetrics`
    TypedDict — single source of truth across standard and atomic.

    Use cases:

    1. Test discipline — assert atomic converter X emits action types Y on a
       fixture (parity with standard's cross-provider coverage gate).
    2. Downstream validation — consumers calling silly-kicks-converted bronze
       atomic data can verify expected coverage before downstream aggregation.

    Parameters
    ----------
    actions : pd.DataFrame
        Atomic-SPADL action stream. Must contain ``type_id``.
    expected_action_types : set[str] or None, default ``None``
        Action type names expected to be present. Returned (sorted) as
        ``missing`` if absent. ``None`` skips the expectation check; an empty
        list is then returned for ``missing``. Atomic vocabulary differs from
        standard on collapsed names — pass ``"corner"`` (not ``"corner_short"``)
        and ``"freekick"`` (not ``"freekick_short"``).

    Returns
    -------
    CoverageMetrics
        ``{"counts": {...}, "missing": [...], "total_actions": N}`` — the
        standard TypedDict from ``silly_kicks.spadl.utils``.

    Raises
    ------
    ValueError
        If the ``type_id`` column is missing.

    Examples
    --------
    Validate an atomic converter's output covers all expected action types::

        from silly_kicks.atomic.spadl import convert_to_atomic, coverage_metrics

        atomic = convert_to_atomic(actions)
        m = coverage_metrics(
            actions=atomic,
            expected_action_types={"pass", "shot", "receival", "interception"},
        )
        assert not m["missing"], f"Missing atomic action types: {m['missing']}"
    """
    if "type_id" not in actions.columns:
        raise ValueError(f"coverage_metrics: actions missing required 'type_id' column. Got: {sorted(actions.columns)}")

    n = len(actions)
    counts: dict[str, int] = {}
    if n > 0:
        # Reverse map: id -> name. Built from atomic actiontypes (single source
        # of truth for atomic). Out-of-vocab ids report as "unknown".
        id_to_name = {i: name for i, name in enumerate(spadlconfig.actiontypes)}
        type_id_arr = actions["type_id"].to_numpy()
        for tid in type_id_arr:
            if pd.isna(tid):
                name = "unknown"
            else:
                name = id_to_name.get(int(tid), "unknown")
            counts[name] = counts.get(name, 0) + 1

    expected = set(expected_action_types) if expected_action_types else set()
    missing = sorted(expected - set(counts.keys())) if expected else []

    return CoverageMetrics(counts=counts, missing=missing, total_actions=n)
