"""DAS input port: pandas frames -> ragged float64 arrays keyed by (game, period, frame) (ADR-107).

The single boundary where silly-kicks frames become the native engine's numeric input. It:

* validates the input contract (spec 6.7) -- the fail-loud conditions that the accessible-space
  adapter used to let slide (duplicate rows, multiple ball rows, possession/carrier varying within a
  frame, non +-1 direction) now RAISE here;
* upcasts ``x``/``y``/``vx``/``vy`` to float64 at the read boundary and centres coordinates to the
  ``[-52.5, 52.5] x [-34, 34]`` frame the engine works in (ADR-106 -- the storage-rounding is the sole
  drift, and the numba kernels stay on float64 signatures);
* identifies the ball by the ``is_ball`` MASK -- ``player_id`` is never written (the ADR-106 sentinel
  removal), so a ``category`` ``player_id`` packs cleanly;
* orders players within a frame by canonical id (ADR-019), matching the reference's pivot column order
  so the engine's sequential reductions are bit-identical to accessible-space;
* resolves each frame's attacking direction from the ``GoalMap`` (ADR-055) or a caller column
  (ADR-051 -- direction is a value, never derived from team identity), and tags each frame with a
  ``Reason`` code so the engine can NaN a frame without simulating it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd

from silly_kicks.id_compat import canonical_id, canonical_id_series
from silly_kicks.tracking._das_taxonomy import DAS_SOURCE_UNSCOREABLE_FRAME, DasUnscoreableError
from silly_kicks.tracking._gk_resolve import GoalMap, resolve_defended_goals
from silly_kicks.tracking._velocity_availability import velocity_unavailable_by_design
from silly_kicks.tracking.schema import SPEED_SOURCE_UNAVAILABLE

_X_OFFSET = 52.5
_Y_OFFSET = 34.0
_FRAME_KEYS = ["game_id", "period_id", "frame_id"]
_REQUIRED = (*_FRAME_KEYS, "player_id", "team_id", "is_ball", "x", "y", "team_in_possession")
_DEFAULT_PLAYER_IN_POSSESSION_COL = "ball_carrier_player_id"


class Reason(IntEnum):
    """Why a frame is (not) scoreable. ``OK`` frames are simulated; the rest degrade to NaN."""

    OK = 0
    NO_POSSESSION = 1
    NO_BALL = 2
    NO_PLAYERS = 3
    BALL_NAN = 4
    POSSESSION_TEAM_ABSENT = 5
    DIRECTION_UNRESOLVED = 6


@dataclass(frozen=True)
class PackedFrames:
    """Ragged, contiguous float64 arrays for a set of frames (engine input)."""

    keys: pd.DataFrame  # one row per frame: game_id, period_id, frame_id (sorted)
    offsets: np.ndarray  # int64 (n_frames + 1,); frame f's players = [offsets[f], offsets[f+1])
    px: np.ndarray  # float64, centred x
    py: np.ndarray  # float64, centred y
    pvx: np.ndarray  # float64
    pvy: np.ndarray  # float64
    p_attacking: np.ndarray  # bool, player's team == frame's team_in_possession
    p_is_passer: np.ndarray  # bool, player id == frame's carrier id
    p_input_pos: np.ndarray  # int64, position of each packed player row in the input frame
    ball_xy: np.ndarray  # float64 (n_frames, 2), centred; NaN when no/NaN ball
    direction: np.ndarray  # float64 (n_frames,), +1 / -1 / NaN
    reason: np.ndarray  # uint8 (n_frames,), Reason codes
    ball_input_pos: np.ndarray  # int64 (n_frames,), -1 when no ball row

    @property
    def n_frames(self) -> int:
        return len(self.keys)


def _require_columns(frames: pd.DataFrame, *, need_gk: bool) -> None:
    # Velocity FIRST (before the other required columns): a frame source that declares
    # speed_source == 'unavailable' structurally cannot carry 'vx'/'vy', so it degrades
    # (DasUnscoreableError) even when a DAS-scoring column such as team_in_possession is ALSO
    # absent -- the honest "this source has no velocity" wins over a generic missing-column
    # error. This preserves the pre-native _validate_das_inputs order that add_das relies on to
    # NaN-degrade velocity-less SB360 freeze-frames (ADR-063) instead of raising. A source that
    # merely FORGOT derive_velocities() (vx/vy absent, no marker) still fails loud.
    if "vx" not in frames.columns or "vy" not in frames.columns:
        if velocity_unavailable_by_design(frames):
            raise DasUnscoreableError(
                f"every frame declares speed_source={SPEED_SOURCE_UNAVAILABLE!r}: this frame source has "
                "no per-player temporal history, so 'vx'/'vy' can never exist and DAS is structurally "
                f"unavailable. DAS degrades to NaN (das_source={DAS_SOURCE_UNSCOREABLE_FRAME!r}).",
                das_source=DAS_SOURCE_UNSCOREABLE_FRAME,
            )
        raise ValueError(
            "DAS requires velocity columns ('vx', 'vy'). Call derive_velocities() or smooth_frames() first."
        )
    missing = [c for c in _REQUIRED if c not in frames.columns]
    if missing:
        raise ValueError(f"DAS requires columns {missing} (missing from frames).")
    if need_gk and "is_goalkeeper" not in frames.columns:
        raise ValueError(
            "DAS direction needs an 'is_goalkeeper' column to build the goal map. Pass goal_map= or "
            "attacking_direction_col= to supply direction another way."
        )


def _canon_list(s: pd.Series) -> list:
    """Canonical ids with ``pd.NA`` mapped to ``None`` (so ``is not None`` / ``==`` are unambiguous)."""
    return [None if pd.isna(v) else v for v in canonical_id_series(s).to_numpy()]


def _nunique_nan_aware(values: np.ndarray) -> int:
    """Distinct count treating every NaN/NA as ONE additional distinct value."""
    ser = pd.Series(values)
    n = int(ser.nunique(dropna=True))
    return n + (1 if ser.isna().any() else 0)


def pack_frames(
    frames: pd.DataFrame,
    *,
    goal_map: GoalMap | None = None,
    attacking_direction_col: str | None = None,
    player_in_possession_col: str | None = _DEFAULT_PLAYER_IN_POSSESSION_COL,
) -> PackedFrames:
    """Validate ``frames`` and pack them into :class:`PackedFrames` (see the module docstring)."""
    if goal_map is not None and attacking_direction_col is not None:
        raise ValueError("pass goal_map= OR attacking_direction_col=, not both.")

    _require_columns(frames, need_gk=goal_map is None and attacking_direction_col is None)

    carrier_col = player_in_possession_col if (player_in_possession_col in frames.columns) else None

    is_ball_all = frames["is_ball"].astype(bool).to_numpy()
    players = frames.loc[~is_ball_all]

    # Fail-loud structural checks (spec 6.7). ---------------------------------------------------------
    # A "player" is (team_id, player_id): some providers number players per-team (e.g. jersey-style
    # ids), so team 1's #20 and team 2's #20 are DISTINCT players, not a duplicate. Keying on team_id
    # too removes that false positive without weakening the D-DUP guard -- a genuine duplicate repeats
    # the SAME player on the SAME team, which this still catches (attacking is keyed on team_id anyway).
    if players.duplicated(subset=[*_FRAME_KEYS, "team_id", "player_id"]).any():
        raise ValueError("DAS frames contain duplicate (game_id, period_id, frame_id, team_id, player_id) rows.")
    ball_per_frame = frames.loc[is_ball_all].groupby(_FRAME_KEYS, observed=True).size()
    if (ball_per_frame > 1).any():
        raise ValueError("DAS frames contain more than one ball row (is_ball) in a frame.")
    for col, label in ((("team_in_possession"), "team_in_possession"),) + (
        ((carrier_col, "carrier column"),) if carrier_col is not None else ()
    ):
        varies = frames.groupby(_FRAME_KEYS, observed=True)[col].apply(lambda s: _nunique_nan_aware(s.to_numpy()) > 1)
        if bool(varies.any()):
            raise ValueError(f"DAS {label} varies within a frame; it must be constant per frame.")

    if not frames["team_in_possession"].notna().any():
        raise DasUnscoreableError(
            "team_in_possession is all-NaN (dead-ball window): DAS is undefined here.",
            das_source="unscoreable_call",
        )

    # Sort: frames by key, players within a frame by canonical id (the reference pivot order). --------
    work = frames.copy()
    work["_pos"] = np.arange(len(work), dtype=np.int64)
    work["_is_ball"] = is_ball_all
    work["_pcanon"] = canonical_id_series(work["player_id"]).astype("object")
    work.loc[work["_is_ball"], "_pcanon"] = ""  # ball sorts first within a frame, then dropped
    work = work.sort_values([*_FRAME_KEYS, "_is_ball", "_pcanon"], kind="stable").reset_index(drop=True)

    w_is_ball = work["_is_ball"].to_numpy()
    pw = work.loc[~w_is_ball]

    # Frame keys (sorted, unique) and per-frame player offsets. --------------------------------------
    keys = pw[_FRAME_KEYS].drop_duplicates().reset_index(drop=True)
    if keys.empty:
        keys = frames[_FRAME_KEYS].drop_duplicates().sort_values(_FRAME_KEYS).reset_index(drop=True)
    frame_index = {tuple(canonical_id(v) for v in row): i for i, row in enumerate(keys.to_numpy())}
    n_frames = len(keys)

    pk = [tuple(canonical_id(v) for v in row) for row in pw[_FRAME_KEYS].to_numpy()]
    pf = np.array([frame_index[k] for k in pk], dtype=np.int64)
    offsets = np.zeros(n_frames + 1, dtype=np.int64)
    np.add.at(offsets, pf + 1, 1)
    np.cumsum(offsets, out=offsets)

    px = np.asarray(pw["x"], dtype=np.float64) - _X_OFFSET
    py = np.asarray(pw["y"], dtype=np.float64) - _Y_OFFSET
    pvx = np.asarray(pw["vx"], dtype=np.float64)
    pvy = np.asarray(pw["vy"], dtype=np.float64)
    p_input_pos = pw["_pos"].to_numpy(dtype=np.int64)

    # Attacking mask (player team == frame possession) and passer mask, dtype-safe (ADR-019). --------
    pw_team = _canon_list(pw["team_id"])
    pw_poss = _canon_list(pw["team_in_possession"])
    p_attacking = np.array(
        [a is not None and b is not None and a == b for a, b in zip(pw_team, pw_poss, strict=True)], dtype=bool
    )
    if carrier_col is not None:
        pw_pid = _canon_list(pw["player_id"])
        pw_carrier = _canon_list(pw[carrier_col])
        p_is_passer = np.array(
            [a is not None and b is not None and a == b for a, b in zip(pw_pid, pw_carrier, strict=True)], dtype=bool
        )
    else:
        p_is_passer = np.zeros(len(pw), dtype=bool)

    # Per-frame ball, direction and reason. ----------------------------------------------------------
    ball = work.loc[w_is_ball]
    ball_xy = np.full((n_frames, 2), np.nan)
    ball_input_pos = np.full(n_frames, -1, dtype=np.int64)
    for pos_x, pos_y, bpos, krow in zip(
        np.asarray(ball["x"], dtype=np.float64),
        np.asarray(ball["y"], dtype=np.float64),
        ball["_pos"].to_numpy(dtype=np.int64),
        ball[_FRAME_KEYS].to_numpy(),
        strict=True,
    ):
        k = tuple(canonical_id(v) for v in krow)
        if k in frame_index:
            fi = frame_index[k]
            ball_xy[fi] = (pos_x - _X_OFFSET, pos_y - _Y_OFFSET)
            ball_input_pos[fi] = bpos

    direction = _resolve_direction(frames, keys, goal_map, attacking_direction_col, frame_index)
    reason = _frame_reasons(keys, offsets, ball_xy, ball_input_pos, direction, pw, frame_index)

    return PackedFrames(
        keys=keys,
        offsets=offsets,
        px=px,
        py=py,
        pvx=pvx,
        pvy=pvy,
        p_attacking=p_attacking,
        p_is_passer=p_is_passer,
        p_input_pos=p_input_pos,
        ball_xy=ball_xy,
        direction=direction,
        reason=reason,
        ball_input_pos=ball_input_pos,
    )


_PAIRED_INVARIANT_COLS = ("game_id", "period_id", "frame_id", "player_id", "team_id", "is_ball", "team_in_possession")


def pack_paired(
    actual: pd.DataFrame,
    counterfactual: pd.DataFrame,
    *,
    goal_map: GoalMap | None = None,
    attacking_direction_col: str | None = None,
    player_in_possession_col: str | None = _DEFAULT_PLAYER_IN_POSSESSION_COL,
) -> tuple[PackedFrames, PackedFrames, np.ndarray]:
    """Pack two legs that differ ONLY in a set of moved rows (SC-1; spec 6.6).

    A row is "moved" iff any of ``x``/``y``/``vx``/``vy`` differs bitwise (NaN-aware) between the legs.
    Every other column (``_PAIRED_INVARIANT_COLS`` + the carrier column) and the WHOLE ball row must
    agree exactly on every row, and both legs must list the same rows in the same order; otherwise
    ``ValueError``. Sharing is therefore only ever applied to bit-identical rows, making the ADR-043
    landmine (a counterfactual leg served factual values) structurally impossible.

    Returns ``(packed_actual, packed_counterfactual, moved)`` where ``moved`` is a bool array aligned
    with the packed player rows (both legs share the same packed row order).
    """
    a = actual.reset_index(drop=True)
    c = counterfactual.reset_index(drop=True)
    if len(a) != len(c):
        raise ValueError(f"paired legs have different row counts ({len(a)} vs {len(c)}).")

    # Validate columns on BOTH legs FIRST -- in particular the velocity-unavailable-by-design degrade
    # (DasUnscoreableError) must fire before the vx/vy ball-row equality loop below, or a velocity-less
    # leg (SB360 freeze-frame) KeyErrors on 'vx' instead of honestly NaN-degrading (ADR-063/107).
    need_gk = goal_map is None and attacking_direction_col is None
    _require_columns(a, need_gk=need_gk)
    _require_columns(c, need_gk=need_gk)

    carrier = player_in_possession_col if (player_in_possession_col in a.columns) else None
    check_cols = [*_PAIRED_INVARIANT_COLS] + ([carrier] if carrier is not None else [])
    for col in check_cols:
        if col not in a.columns or col not in c.columns:
            raise ValueError(f"paired legs must both carry column {col!r}.")
        av = _canon_list(a[col]) if col not in ("x", "y", "vx", "vy") else None
        if col in ("is_ball",):
            if not np.array_equal(a[col].astype(bool).to_numpy(), c[col].astype(bool).to_numpy()):
                raise ValueError("paired legs disagree on is_ball row-for-row.")
            continue
        cv = _canon_list(c[col])
        if av != cv:
            raise ValueError(f"paired legs disagree on {col!r} row-for-row (only kinematics may differ).")

    is_ball = a["is_ball"].astype(bool).to_numpy()
    for col in ("x", "y", "vx", "vy"):
        ba = np.asarray(a.loc[is_ball, col], dtype=np.float64)
        bc = np.asarray(c.loc[is_ball, col], dtype=np.float64)
        if not np.array_equal(ba, bc, equal_nan=True):
            raise ValueError(f"paired legs move the ball ({col}); the ball row must be identical.")

    packed_a = pack_frames(
        a,
        goal_map=goal_map,
        attacking_direction_col=attacking_direction_col,
        player_in_possession_col=player_in_possession_col,
    )
    packed_c = pack_frames(
        c,
        goal_map=goal_map,
        attacking_direction_col=attacking_direction_col,
        player_in_possession_col=player_in_possession_col,
    )
    if not np.array_equal(packed_a.offsets, packed_c.offsets) or not np.array_equal(
        packed_a.p_input_pos, packed_c.p_input_pos
    ):
        raise ValueError("paired legs pack to different structures (row order differs).")

    moved = np.zeros(len(packed_a.px), dtype=bool)
    for aa, cc in (
        (packed_a.px, packed_c.px),
        (packed_a.py, packed_c.py),
        (packed_a.pvx, packed_c.pvx),
        (packed_a.pvy, packed_c.pvy),
    ):
        diff = ~((aa == cc) | (np.isnan(aa) & np.isnan(cc)))
        moved |= diff
    return packed_a, packed_c, moved


def _resolve_direction(frames, keys, goal_map, attacking_direction_col, frame_index) -> np.ndarray:
    n = len(keys)
    direction = np.full(n, np.nan)
    if attacking_direction_col is not None:
        if attacking_direction_col not in frames.columns:
            raise ValueError(f"attacking_direction_col={attacking_direction_col!r} not found in frames.")
        col = frames[[*_FRAME_KEYS, attacking_direction_col]]
        per_frame = col.groupby(_FRAME_KEYS, observed=True)[attacking_direction_col].agg(
            lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan
        )
        for krow, val in per_frame.items():
            k = tuple(canonical_id(v) for v in (krow if isinstance(krow, tuple) else (krow,)))
            if k in frame_index:
                if pd.notna(val) and val not in (1.0, -1.0):
                    raise ValueError(
                        f"attacking_direction_col={attacking_direction_col!r} must be +1/-1 per frame; got {val!r}."
                    )
                direction[frame_index[k]] = float(val) if pd.notna(val) else np.nan
        return direction

    gm = goal_map if goal_map is not None else resolve_defended_goals(frames)
    poss_by_frame = frames.groupby(_FRAME_KEYS, observed=True)["team_in_possession"].agg(
        lambda s: s.dropna().iloc[0] if s.notna().any() else None
    )
    for i, krow in enumerate(keys.to_numpy()):
        g, p, _f = krow
        poss = poss_by_frame.get(tuple(krow))
        if poss is None or pd.isna(poss):
            continue
        attacked = gm.attacked_goal(g, p, poss, allow_guess=True)
        if attacked is None:
            continue
        direction[i] = 1.0 if attacked >= _X_OFFSET else -1.0
    return direction


def _frame_reasons(keys, offsets, ball_xy, ball_input_pos, direction, pw, frame_index) -> np.ndarray:
    n = len(keys)
    reason = np.zeros(n, dtype=np.uint8)
    # possession present per frame? (poss constant per frame already; NaN => NO_POSSESSION)
    poss_present = np.zeros(n, dtype=bool)
    team_present: list[set] = [set() for _ in range(n)]
    poss_id = [None] * n
    if len(pw):
        pw_poss = _canon_list(pw["team_in_possession"])
        pw_team = _canon_list(pw["team_id"])
        pk = [tuple(canonical_id(v) for v in row) for row in pw[["game_id", "period_id", "frame_id"]].to_numpy()]
        for poss, team, k in zip(pw_poss, pw_team, pk, strict=True):
            fi = frame_index[k]
            if poss is not None:
                poss_present[fi] = True
                if poss_id[fi] is None:
                    poss_id[fi] = poss
            if team is not None:
                team_present[fi].add(team)

    n_players = np.diff(offsets)
    for i in range(n):
        if not poss_present[i]:
            reason[i] = Reason.NO_POSSESSION
        elif ball_input_pos[i] < 0:
            reason[i] = Reason.NO_BALL
        elif n_players[i] == 0:
            reason[i] = Reason.NO_PLAYERS
        elif not np.isfinite(ball_xy[i]).all():
            reason[i] = Reason.BALL_NAN
        elif poss_id[i] is not None and poss_id[i] not in team_present[i]:
            reason[i] = Reason.POSSESSION_TEAM_ABSENT
        elif not np.isfinite(direction[i]):
            reason[i] = Reason.DIRECTION_UNRESOLVED
        else:
            reason[i] = Reason.OK
    return reason
