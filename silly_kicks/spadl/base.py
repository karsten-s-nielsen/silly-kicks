"""Utility functions for all event stream to SPADL converters.

A converter should implement 'convert_to_actions' to convert the events to the
SPADL format.

"""

import numpy as np
import pandas as pd  # type: ignore

from silly_kicks.id_compat import ids_differ, ids_equal

from . import config as spadlconfig
from .utils import _SET_PIECE_RESTART_TYPE_NAMES

# Type IDs for pass-class actions where the ball physically travels to a
# different location.  Used by _derive_end_coordinates to overwrite
# placeholder end_x/end_y with the next action's start position.
_DERIVE_END_TYPE_IDS: frozenset[int] = frozenset(
    {
        spadlconfig.actiontype_id["pass"],  # 0
        spadlconfig.actiontype_id["cross"],  # 1
        spadlconfig.actiontype_id["throw_in"],  # 2
        spadlconfig.actiontype_id["freekick_crossed"],  # 3
        spadlconfig.actiontype_id["freekick_short"],  # 4
        spadlconfig.actiontype_id["corner_crossed"],  # 5
        spadlconfig.actiontype_id["corner_short"],  # 6
        spadlconfig.actiontype_id["clearance"],  # 18
        spadlconfig.actiontype_id["goalkick"],  # 22
    }
)


def sort_actions_chronologically(
    frame: pd.DataFrame,
    *,
    by: tuple[str, ...] = ("game_id", "period_id", "time_seconds"),
    tiebreak: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Stable-sort an actions/events frame into chronological order.

    The one shared sort seam every converter uses at the top of its frame,
    before ANY positional (``.shift()``-based) derivation, so that
    ``action_id`` becomes a chronological ``(game_id, period_id, time_seconds)``
    index and every time-neighbour lookup resolves the true neighbour.

    Sorts by ``(*by, *tiebreak)`` with a stable ``kind="mergesort"`` so genuine
    intra-timestamp ties keep their input order. An empty frame is returned
    unchanged; the input is never mutated (``sort_values`` returns a new frame).

    ``by`` defaults to the SPADL-actions key; raw-event callers pass their own
    ordering columns (e.g. ``by=("period_id", "milliseconds")``). A missing or
    mistyped ordering key RAISES ``KeyError`` -- the seam whose whole job is to
    never let ordering rot silently must fail loud, not degrade to a partial
    sort.

    Examples
    --------
    Sort a SPADL actions frame chronologically before any positional derivation::

        ordered = sort_actions_chronologically(actions)

    Raw-event callers pass their own ordering columns (``time_seconds`` is not yet
    built), optionally with an intra-timestamp tiebreak::

        ordered = sort_actions_chronologically(
            events, by=("game_id", "period_id", "minute", "second"), tiebreak=("event_id",)
        )
    """
    if len(frame) == 0:
        return frame
    keys = [*by, *tiebreak]
    missing = [c for c in keys if c not in frame.columns]
    if missing:  # M-D: NEVER silently partial-sort -- the exact bug this seam exists to prevent
        raise KeyError(
            f"sort_actions_chronologically: ordering key(s) {missing} absent. Pass a `by=` matching "
            f"this frame (raw-event frames pass e.g. by=('period_id','milliseconds')); a missing key "
            f"must fail loud, not degrade to a partial sort."
        )
    return frame.sort_values(keys, kind="mergesort").reset_index(drop=True)


def _derive_end_coordinates(actions: pd.DataFrame, *, extra_type_ids: frozenset[int] = frozenset()) -> pd.DataFrame:
    """Derive end_x/end_y from next action's start for pass-class types.

    Only overwrites rows where the source data did not provide a separate
    end coordinate (detected by ``end_x == start_x AND end_y == start_y``,
    or ``end_x`` is NaN).
    Period-safe: uses ``groupby("period_id").shift(-1)`` so the last action
    per period keeps its original end coordinates.

    ``extra_type_ids`` opt-in extends the derive set for ONE caller (PR-S116:
    gradientsports passes ``{dribble}`` because its OTB/BC carries ship
    placeholder ends and it never calls ``_add_dribbles``). The module-level
    ``_DERIVE_END_TYPE_IDS`` is shared by all eight converters and must not
    change: the placeholder guard cannot distinguish statsbomb's ~11% genuine
    stationary carries from placeholders, so a global dribble addition would
    rewrite recorded data.

    Replaces the former ``_fix_clearances`` with a broader type set, a
    source-data guard, and period-boundary safety.
    """
    if len(actions) == 0:
        return actions
    actions = actions.copy()

    is_pass_class = actions["type_id"].isin(_DERIVE_END_TYPE_IDS | extra_type_ids)
    placeholder_end = (actions["end_x"] == actions["start_x"]) & (actions["end_y"] == actions["start_y"])
    missing_end = actions["end_x"].isna()
    needs_derivation = is_pass_class & (placeholder_end | missing_end)

    next_start_x = actions.groupby("period_id")["start_x"].shift(-1)
    next_start_y = actions.groupby("period_id")["start_y"].shift(-1)

    mask = needs_derivation & next_start_x.notna()
    actions.loc[mask, "end_x"] = next_start_x[mask].values
    actions.loc[mask, "end_y"] = next_start_y[mask].values
    return actions


#: The advanced-start line of an open-play cross (spec §6.1 condition 3, ADR-133), in SPADL metres from the
#: passer's own goal line. 52.5 is the provisional, most permissive candidate; the start-line measurement
#: (spec §7.2) sets the final value.
CROSS_START_X: float = 52.5

_DELIVERY_SKIP_TYPE_IDS: frozenset[int] = frozenset(
    {spadlconfig.actiontype_id["non_action"], spadlconfig.actiontype_id["foul"]}
)
_RESTART_TYPE_IDS: frozenset[int] = frozenset(
    spadlconfig.actiontype_id[t] for t in (*_SET_PIECE_RESTART_TYPE_NAMES, "shot_penalty", "shot_freekick")
)
_OPEN_PLAY_PASS_IDS = (spadlconfig.actiontype_id["pass"], spadlconfig.actiontype_id["cross"])
_CORNER_IDS = (spadlconfig.actiontype_id["corner_crossed"], spadlconfig.actiontype_id["corner_short"])
_FREEKICK_IDS = (spadlconfig.actiontype_id["freekick_crossed"], spadlconfig.actiontype_id["freekick_short"])
_BOX_X: float = spadlconfig.field_length - spadlconfig.penalty_area_depth
_BOX_Y_LO: float = spadlconfig.field_width / 2 - spadlconfig.penalty_area_half_width
_BOX_Y_HI: float = spadlconfig.field_width / 2 + spadlconfig.penalty_area_half_width


_DELIVERY_CLASS_IDS = (*_OPEN_PLAY_PASS_IDS, *_CORNER_IDS, *_FREEKICK_IDS)
_CROSSED_IDS = (
    spadlconfig.actiontype_id["cross"],
    spadlconfig.actiontype_id["corner_crossed"],
    spadlconfig.actiontype_id["freekick_crossed"],
)


def _native_cross_flag(type_id: pd.Series) -> "pd.arrays.BooleanArray":
    """The provider's own "crossed" label, read from the converter's native typing (ADR-133 ``cross_native``).

    Call it on the natively typed rows, just before :func:`classify_deliveries` re-types them. True on a native
    ``cross`` / ``corner_crossed`` / ``freekick_crossed``, False on a native ``pass`` / ``corner_short`` /
    ``freekick_short``, ``pd.NA`` on every other row.
    """
    t = np.asarray(type_id)
    delivery = np.isin(t, _DELIVERY_CLASS_IDS)
    crossed = np.isin(t, _CROSSED_IDS)
    return pd.arrays.BooleanArray(crossed, ~delivery)


def _synthesized_rows(actions: pd.DataFrame) -> np.ndarray:
    """Rows a converter synthesized (Metrica/Sportec GK distribution: ``original_event_id`` ends ``_synth_*``).

    They carry no provider label, so their ``cross_native`` is ``pd.NA`` (ADR-133; GS marks its own synthesized
    rows the same way through ``is_synthetic``).
    """
    return actions["original_event_id"].astype(str).str.contains("_synth_", regex=False).to_numpy()


def _delivery_context(actions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per row: (delivered, next_is_restart) -- the event-layer delivery condition (spec §6.1 condition 4).

    ``delivered``: the next on-ball action in the same period starts inside the penalty area (mirrored into the
    passer's frame for an opponent) and is not a restart. ``next_is_restart``: the next on-ball action in the
    same period is a restart (the event data has no delivery end). Shared by :func:`classify_deliveries` and the
    corpus measurement, so both read one implementation.
    """
    n = len(actions)
    type_id = actions["type_id"].to_numpy()
    on_ball = np.flatnonzero(~np.isin(type_id, list(_DELIVERY_SKIP_TYPE_IDS)))
    nxt = np.full(n, -1, dtype=np.int64)
    nxt[on_ball[:-1]] = on_ball[1:]
    has_next = nxt >= 0
    nidx = np.where(has_next, nxt, 0)

    period = actions["period_id"].to_numpy()
    team = actions["team_id"].reset_index(drop=True)
    next_team = team.iloc[nidx].reset_index(drop=True)
    same = ids_equal(team, next_team).to_numpy()
    differ = ids_differ(team, next_team).to_numpy()

    sx = actions["start_x"].to_numpy(dtype=np.float64, copy=True)
    sy = actions["start_y"].to_numpy(dtype=np.float64, copy=True)
    nx = np.where(differ, spadlconfig.field_length - sx[nidx], sx[nidx])
    ny = np.where(differ, spadlconfig.field_width - sy[nidx], sy[nidx])
    in_box = (nx >= _BOX_X) & (ny >= _BOX_Y_LO) & (ny <= _BOX_Y_HI)
    same_period_next = has_next & (period[nidx] == period)
    next_restart = same_period_next & np.isin(type_id[nidx], list(_RESTART_TYPE_IDS))
    delivered = same_period_next & (same | differ) & ~next_restart & in_box
    return delivered, next_restart


def classify_deliveries(actions: pd.DataFrame, *, cross_start_x: float = CROSS_START_X) -> pd.DataFrame:
    """Re-type open-play passes and set-piece deliveries by one provider-neutral geometric rule (ADR-133).

    An open-play pass-class row (``pass`` or ``cross``) is a ``cross`` when it starts wide (outside the
    penalty-area width), at or beyond ``cross_start_x``, and the next on-ball action in the same period starts
    inside the penalty area. A corner or free kick is ``*_crossed`` under the delivery condition alone, and
    ``*_short`` otherwise. Every other row and every other column is unchanged.

    The next on-ball action is the next row that is not ``non_action`` or ``foul``. It is mirrored into the
    passer's frame when its team differs. A restart as the next action (the ball left play or the game
    stopped) never delivers: the event data has no delivery end there. A row whose own or next team is
    missing, or whose coordinates are NaN, never delivers.

    Event-only limits: a blocked cross and a restart-ended delivery are typed ``pass`` (and a restart-ended
    set piece ``*_short``). The tracking layer (``add_cross_deliveries``) decides those from the ball path.

    Parameters
    ----------
    actions : pd.DataFrame
        Canonical SPADL LTR rows (after ``to_spadl_ltr``), in final chronological order, before
        ``_add_dribbles``. Needs ``period_id``, ``team_id``, ``type_id``, ``start_x`` and ``start_y``.
    cross_start_x : float
        The advanced-start line of an open-play cross. Defaults to :data:`CROSS_START_X`.

    Returns
    -------
    pd.DataFrame
        A copy with ``type_id`` re-typed.

    Examples
    --------
    >>> import pandas as pd
    >>> from silly_kicks.spadl import config as spadlconfig
    >>> pid = spadlconfig.actiontype_id
    >>> a = pd.DataFrame({"period_id": [1, 1], "team_id": [1, 1], "type_id": [pid["pass"], pid["shot"]],
    ...                   "start_x": [80.0, 95.0], "start_y": [5.0, 34.0]})
    >>> [spadlconfig.actiontypes[t] for t in classify_deliveries(a)["type_id"]]
    ['cross', 'shot']
    """
    out = actions.copy()
    if len(out) == 0:
        return out
    type_id = out["type_id"].to_numpy()
    sx = out["start_x"].to_numpy(dtype=np.float64, copy=True)
    sy = out["start_y"].to_numpy(dtype=np.float64, copy=True)
    delivered, _next_restart = _delivery_context(out)

    is_open = np.isin(type_id, _OPEN_PLAY_PASS_IDS)
    is_corner = np.isin(type_id, _CORNER_IDS)
    is_freekick = np.isin(type_id, _FREEKICK_IDS)
    wide = (sy < _BOX_Y_LO) | (sy > _BOX_Y_HI)
    is_cross = is_open & wide & (sx >= cross_start_x) & delivered
    at = spadlconfig.actiontype_id
    new_type = np.select(
        [
            is_cross,
            is_open,
            is_corner & delivered,
            is_corner,
            is_freekick & delivered,
            is_freekick,
        ],
        [
            at["cross"],
            at["pass"],
            at["corner_crossed"],
            at["corner_short"],
            at["freekick_crossed"],
            at["freekick_short"],
        ],
        default=type_id,
    )
    out["type_id"] = pd.Series(new_type, index=out.index, dtype=out["type_id"].dtype)
    return out


min_dribble_length: float = 3.0
max_dribble_length: float = 60.0
max_dribble_duration: float = 10.0


def _add_dribbles(actions: pd.DataFrame) -> pd.DataFrame:
    next_actions = actions.shift(-1)

    same_team = actions.team_id == next_actions.team_id
    # not_clearance = actions.type_id != actiontypes.index("clearance")
    not_offensive_foul = same_team & (next_actions.type_id != spadlconfig.actiontype_id["foul"])
    not_headed_shot = (next_actions.type_id != spadlconfig.actiontype_id["shot"]) & (
        next_actions.bodypart_id != spadlconfig.bodypart_id["head"]
    )

    dx = actions.end_x - next_actions.start_x
    dy = actions.end_y - next_actions.start_y
    far_enough = dx**2 + dy**2 >= min_dribble_length**2
    not_too_far = dx**2 + dy**2 <= max_dribble_length**2

    dt = next_actions.time_seconds - actions.time_seconds
    same_phase = dt < max_dribble_duration
    same_period = actions.period_id == next_actions.period_id

    dribble_idx = same_team & far_enough & not_too_far & same_phase & same_period & not_offensive_foul & not_headed_shot

    dribbles = pd.DataFrame()
    prev = actions[dribble_idx]
    nex = next_actions[dribble_idx]
    dribbles["game_id"] = nex.game_id
    dribbles["period_id"] = nex.period_id
    dribbles["action_id"] = prev.action_id + 0.1
    dribbles["time_seconds"] = (prev.time_seconds + nex.time_seconds) / 2
    if "timestamp" in actions.columns:
        dribbles["timestamp"] = nex.timestamp
    dribbles["team_id"] = nex.team_id
    dribbles["player_id"] = nex.player_id
    dribbles["start_x"] = prev.end_x
    dribbles["start_y"] = prev.end_y
    dribbles["end_x"] = nex.start_x
    dribbles["end_y"] = nex.start_y
    dribbles["bodypart_id"] = spadlconfig.bodypart_id["foot"]
    dribbles["type_id"] = spadlconfig.actiontype_id["dribble"]
    dribbles["result_id"] = spadlconfig.result_id["success"]

    actions = pd.concat([actions, dribbles], ignore_index=True, sort=False)
    actions = actions.sort_values(["game_id", "period_id", "action_id"]).reset_index(drop=True)  # type: ignore[reportAssignmentType]
    actions["action_id"] = range(len(actions))
    return actions
