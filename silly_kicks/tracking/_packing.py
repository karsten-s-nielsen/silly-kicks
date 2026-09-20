"""TF-49 packing: Impect-faithful bypass counts over tracking frames.

Packing (Impect; Reinartz & Hegeler ~2015) counts opponents removed from the
defensive phase by a COMPLETED pass/cross/set-piece pass/dribble. Longitudinal
(goal-to-goal) geometry per the published formalization (Goes et al. 2019) --
identical inequality to structural_lbs; the far-touchline caveat is canon.
Practitioner rules (goal-threat last-N, secured reception) from the Modern
Soccer Coach "Packing Data" lesson + Twelve/Soccermatics course.

The ~15-line defender-extraction/mirror block is DELIBERATELY duplicated from
_structural_pass.py (frozen kernel isolation; consolidation trigger = a third
consumer, ADR-039). Cross-checked by the golden identity gate.

See docs/superpowers/specs/2026-07-16-tf49-packing-design.md and NOTICE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.id_compat import ids_match, same_id

from ._defensive_line import select_back_line_players
from ._gk_resolve import GoalEndUnresolvedError, GoalMap

_DEFAULT_ACTION_TYPES: tuple[str, ...] = (
    "pass",
    "cross",
    "throw_in",
    "freekick_crossed",
    "freekick_short",
    "corner_crossed",
    "corner_short",
    "goalkick",
    "dribble",
)
_NAN_METRICS = {
    "packing_made": np.nan,
    "packing_net": np.nan,
    "packing_goal_threat": np.nan,
    "line_x": np.nan,
}


@dataclass(frozen=True)
class PackingParams:
    """Tunable parameters for packing metrics. See the TF-49 spec for semantics.

    Examples
    --------
    >>> p = PackingParams(back_line_n=5)
    >>> p.back_line_n
    5
    """

    action_types: tuple[str, ...] = _DEFAULT_ACTION_TYPES
    include_gk: bool = False
    back_line_n: int = 4
    forward_max_deg: float = 45.0
    back_min_deg: float = 135.0
    side_multiplier: float = 0.5
    back_multiplier: float = -1.0
    secured_window_seconds: float = 3.0
    require_secured: bool = False

    def __post_init__(self) -> None:
        if not (0.0 < self.forward_max_deg < self.back_min_deg < 180.0):
            raise ValueError("require 0 < forward_max_deg < back_min_deg < 180")
        if self.secured_window_seconds <= 0:
            raise ValueError("secured_window_seconds must be > 0")
        if self.back_line_n < 1:
            raise ValueError("back_line_n must be >= 1")
        if self.side_multiplier < 0:
            raise ValueError("side_multiplier must be >= 0")
        if self.back_multiplier > 0:
            raise ValueError("back_multiplier must be <= 0")
        if not self.action_types:
            raise ValueError("action_types must be non-empty")
        unknown = set(self.action_types) - set(spadlconfig.actiontype_id)
        if unknown:
            raise ValueError(f"unknown action_types: {sorted(unknown)!r}")


def _frame_ids(frame: pd.DataFrame) -> tuple:
    """``(game_id, period_id)`` for a single linked frame, for the goal-map lookup.

    The map is keyed per (game, period, team), so a per-frame consumer has to say WHICH
    game and period it is looking at. Taking the first row is safe here because the caller
    contract is ONE linked frame; a multi-frame slice would be a caller error and the
    lookup would silently answer for whichever period sorted first, so this asserts rather
    than trusting it.
    """
    if not len(frame):
        return (None, None)
    gids = frame["game_id"].dropna().unique()
    pids = frame["period_id"].dropna().unique()
    if len(gids) > 1 or len(pids) > 1:
        raise ValueError(
            f"compute_packing_metrics expects ONE frame; got game_ids={list(gids)} "
            f"period_ids={list(pids)}. A goal-map lookup on a mixed slice would answer for "
            f"whichever sorted first."
        )
    return (gids[0] if len(gids) else None, pids[0] if len(pids) else None)


def _direction_multiplier(dx: float, dy: float, params: PackingParams) -> float:
    theta = float(np.degrees(np.arctan2(abs(dy), dx)))
    if theta <= params.forward_max_deg:
        return 1.0
    if theta <= params.back_min_deg:
        return params.side_multiplier
    return params.back_multiplier


def _packing_setup(
    frame: pd.DataFrame,
    *,
    attacking_team_id: int | str,
    goal_map: GoalMap,
    params: PackingParams,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The loop-INVARIANT packing setup: the defender x-array ``dx_`` and back-line x-array ``bx``,
    both mirrored into attack-positive x, computed ONCE per (frame, attacking_team, goal_map).

    Everything here is independent of ``passer_xy``/``receiver_xy`` -- the expensive defender
    extraction, the ``goal_map`` lookups and ``select_back_line_players`` -- so the batch hoists it
    out of the per-receiver loop (loop-invariant computation, optimization-audit finding #1). Returns
    ``(dx_, bx)`` where ``dx_ is None`` means "no eligible defender -> all-NaN metrics" and
    ``bx is None`` means "no back line -> goal_threat NaN" (an EMPTY ``bx`` array is distinct: it
    yields ``goal_threat == 0.0``, matching the scalar). Raises ``GoalEndUnresolvedError`` on an
    unresolvable map -- invariant of the receiver, so it fires once for the whole batch.
    """
    # Defender extraction + away-mirror deliberately duplicated from _structural_pass.py
    # (frozen-kernel isolation; consolidation trigger = 3rd consumer, ADR-039). NOTE the
    # deliberate divergence: packing mirrors X ONLY (all three counts are x-interval
    # tests; y is used solely for the direction angle of the ACTION, which lives in
    # attack-positive action coords already) -- structural mirrors both because SGM/SDI
    # consume 2-D defender positions (review minor 13).
    players = frame[~frame["is_ball"].astype(bool)]
    opp_all = players[~ids_match(players["team_id"], attacking_team_id).to_numpy()]
    opp = opp_all if params.include_gk else opp_all[~opp_all["is_goalkeeper"].astype(bool).to_numpy()]
    dx_ = opp["x"].to_numpy(dtype="float64")
    dx_ = dx_[np.isfinite(dx_)]
    if dx_.size == 0:
        return None, None

    # Direction from the map, never from team IDENTITY (ADR-051 D3). The mirror is needed
    # exactly when the ACTING team attacks x=0, which is what `attacked_goal` answers -- and it
    # is a REAL lookup of the opponent's entry, never `105.0 - get(...)`, which would be wrong
    # on a degenerate map.
    _gid, _pid = _frame_ids(frame)
    _attacked = goal_map.attacked_goal(_gid, _pid, attacking_team_id, allow_guess=True)
    if _attacked is None:
        # Explicit: `== 0.0` alone would fail OPEN, silently choosing 'no mirror'.
        raise GoalEndUnresolvedError(
            f"packing: goal_map does not resolve the goal attacked by {attacking_team_id!r} "
            f"in (game={_gid!r}, period={_pid!r})."
        )
    mirror = _attacked == 0.0
    if mirror:
        dx_ = 105.0 - dx_

    # Goal-threat back line: select-then-mirror. select_back_line_players wants the DEFENDING
    # team's id (its "own goal" is the defending team's) -- resolve it NaN-safely from
    # the frame's non-attacking players (review blocker 3). Caveat: the helper
    # short-circuits len(outfield) < 3 -> returns outfield unselected (sparse frames).
    def_team_vals = opp_all["team_id"].dropna().unique()
    if len(def_team_vals) == 0:
        return dx_, None
    # The DEFENDING team's own end -- `get`, not `attacked_goal`: this selects the players
    # nearest the goal they defend. Distinct from the mirror above, which asks where the
    # ATTACKING team is going; `packing_goal_threat` is the only emitted column that
    # witnesses this site, which is why it is named in the entry's gate_c_must_move.
    _def_end = goal_map.get(_gid, _pid, def_team_vals[0], allow_guess=True)
    if _def_end is None:
        raise GoalEndUnresolvedError(
            f"packing: goal_map does not resolve the end defended by {def_team_vals[0]!r} "
            f"in (game={_gid!r}, period={_pid!r})."
        )
    back = select_back_line_players(frame, def_team_vals[0], _def_end == 0.0, n=params.back_line_n)
    if len(back) == 0:
        return dx_, None
    bx = back["x"].to_numpy(dtype="float64")
    bx = bx[np.isfinite(bx)]
    if mirror:
        bx = 105.0 - bx
    return dx_, bx


def compute_packing_metrics_batch(
    frame: pd.DataFrame,
    *,
    attacking_team_id: int | str,
    goal_map: GoalMap,
    passer_xy: tuple[float, float],
    receivers: npt.ArrayLike,
    params: PackingParams | None = None,
) -> dict[str, np.ndarray]:
    """Packing metrics for MANY receivers from ONE frame + passer (the batched kernel).

    The loop-invariant defender/back-line setup is hoisted via :func:`_packing_setup` and only the
    cheap per-receiver counts vary, so N option targets cost one setup instead of N. Byte-identical
    to N scalar :func:`compute_packing_metrics` calls (gated by ``tests/tracking/test_packing_batch``).

    ``receivers`` is an ``(n, 2)`` array of ``(x, y)`` endpoints. Returns a dict of four length-``n``
    float arrays (``packing_made`` / ``packing_net`` / ``packing_goal_threat`` / ``line_x``). A
    non-finite receiver row is NaN; an empty frame or a non-finite passer makes EVERY row NaN.
    Raises ``GoalEndUnresolvedError`` on an unresolvable ``goal_map`` (receiver-invariant).

    Examples
    --------
    Value several option targets from a single keeper distribution::

        from silly_kicks.tracking import compute_packing_metrics_batch
        out = compute_packing_metrics_batch(
            frame, attacking_team_id=1, goal_map=gm,
            passer_xy=(50.0, 34.0), receivers=np.array([[70.0, 34.0], [60.0, 20.0]]),
        )
        out["packing_made"]  # length-2 array
    """
    if params is None:
        params = PackingParams()
    receivers = np.asarray(receivers, dtype=float).reshape(-1, 2)
    n = len(receivers)

    def _all_nan() -> dict[str, np.ndarray]:
        return {k: np.full(n, np.nan) for k in ("packing_made", "packing_net", "packing_goal_threat", "line_x")}

    if frame is None or len(frame) == 0:
        return _all_nan()
    if not all(np.isfinite(v) for v in passer_xy):
        return _all_nan()
    dx_, bx = _packing_setup(frame, attacking_team_id=attacking_team_id, goal_map=goal_map, params=params)
    if dx_ is None:
        return _all_nan()

    p0, p1 = float(passer_xy[0]), float(passer_xy[1])
    made = np.empty(n)
    net = np.empty(n)
    gt = np.empty(n)
    line_x = np.empty(n)
    for i in range(n):
        rx, ry = receivers[i]
        if not (np.isfinite(rx) and np.isfinite(ry)):
            made[i] = net[i] = gt[i] = line_x[i] = np.nan
            continue
        r0, r1 = float(rx), float(ry)
        sel = (dx_ > p0) & (dx_ <= r0)
        made[i] = float(np.count_nonzero(sel))
        bypassed = dx_[sel]
        line_x[i] = float(bypassed.max()) if bypassed.size else np.nan
        lo, hi = (p0, r0) if p0 <= r0 else (r0, p0)
        interval = float(np.count_nonzero((dx_ > lo) & (dx_ <= hi)))
        net[i] = _direction_multiplier(r0 - p0, r1 - p1, params) * interval
        gt[i] = np.nan if bx is None else float(np.count_nonzero((bx > p0) & (bx <= r0)))
    return {"packing_made": made, "packing_net": net, "packing_goal_threat": gt, "line_x": line_x}


def compute_packing_metrics(
    frame: pd.DataFrame,
    *,
    attacking_team_id: int | str,
    goal_map: GoalMap,
    passer_xy: tuple[float, float],
    receiver_xy: tuple[float, float],
    params: PackingParams | None = None,
) -> dict[str, float]:
    """Per-frame packing metrics for ONE linked frame (pure; schema-agnostic endpoints).

    Returns packing_made / packing_net / packing_goal_threat / line_x. NaN when the
    frame is empty, endpoints are non-finite, or no eligible defender exists. Thin scalar wrapper
    over :func:`compute_packing_metrics_batch` (n=1); the receiver-finiteness guard is retained HERE
    so a non-finite receiver returns NaN WITHOUT triggering the batch's goal_map raise (the scalar
    checked receiver finiteness before the invariant work). Defender extraction + away-mirror
    duplicated from _structural_pass.py by design.

    Examples
    --------
    Compute packing metrics for a single pass on a frame::

        from silly_kicks.tracking import compute_packing_metrics
        m = compute_packing_metrics(
            frame, attacking_team_id=1, goal_map=gm,
            passer_xy=(50.0, 34.0), receiver_xy=(70.0, 34.0),
        )
        m["packing_made"]
    """
    if not all(np.isfinite(v) for v in (*passer_xy, *receiver_xy)):
        return dict(_NAN_METRICS)
    out = compute_packing_metrics_batch(
        frame,
        attacking_team_id=attacking_team_id,
        goal_map=goal_map,
        passer_xy=passer_xy,
        receivers=np.asarray([receiver_xy], dtype=float),
        params=params,
    )
    return {k: float(out[k][0]) for k in ("packing_made", "packing_net", "packing_goal_threat", "line_x")}


_SHOT_TYPES = frozenset(spadlconfig.actiontype_id[n] for n in ("shot", "shot_penalty", "shot_freekick"))
_SKIP_TYPES = frozenset(spadlconfig.actiontype_id[n] for n in ("non_action", "foul"))


def secured_reception(
    actions: pd.DataFrame,
    line_x: pd.Series,
    receiver_pos: pd.Series | None = None,
    *,
    params: PackingParams | None = None,
) -> pd.Series:
    """Nullable-boolean 'ball stays past the line' label per action (TF-49 spec s3).

    retains() skeleton (possession-aware; self-heals possession_id via
    spadl.utils.add_possessions) + the REQUIRED foul-skip on top: heuristic
    possessions emit a boundary AT the foul row (verified 2026-07-16), so fouls
    (and non_action rows) are skipped and the next possession-implying event
    decides. NaN-TEAM rows (GS null-actor, ADR-027) are likewise skipped --
    NA-routed comparisons, never raw !=. The window is anchored at the RECEPTION:
    receiver_pos (from spadl.utils._resolve_next_touch_positions, positional Int64)
    locates the receiving row; the scan starts at the row AFTER it and the window
    is (t_r, t_r + secured_window_seconds]. A reception that is ITSELF a same-team
    shot decides True immediately (the literal pass -> shot -> keeper_save shape:
    the shot is the next touch, and the save's possession boundary must not read
    as a loss); the reception row's start_x is never tested (a receiver collecting
    behind the line is not a bounce-pass). Same-team shot -> True; opponent
    possession boundary -> False; same-team action starting behind line_x within
    the window -> False; empty window -> the first subsequent non-skipped event
    decides the boundary/shot tests ONLY (the line_x test does not extend);
    truncation ((t_last - t_r) < window) with no decisive event -> <NA>.

    receiver_pos=None computes positions internally (public-caller path); add_packing
    passes its precomputed positions (one sort/groupby pass per match, not two).

    Rows with NaN ``line_x`` (nothing bypassed / no geometry) or an unresolved
    receiver -> <NA>. The scan runs in ROBUST CHRONOLOGICAL order --
    ``(time_seconds, action_id)`` with ``action_id`` as the tiebreak, the SAME key
    ``_resolve_next_touch_positions`` resolves anchors in (spec 2026-08-20 §3d,
    ADR-065) -- so a persisted mart with a non-chronological ``action_id`` is
    ordered robustly, not rejected (mart reads bypass the ``_finalize_output``
    guard). NaN ``time_seconds`` sorts last and resolves ``<NA>``. A caller-supplied
    ``possession_id`` with missing values never decides the boundary test (NA-routed,
    the ADR-027 discipline).

    Examples
    --------
    Flag whether each pass reception was secured, per action::

        from silly_kicks.tracking import secured_reception
        secured = secured_reception(actions, line_x)
        secured.value_counts(dropna=False)
    """
    if params is None:
        params = PackingParams()
    from silly_kicks.spadl.utils import _resolve_next_touch_positions, add_possessions

    # POSITIONAL WORLD at entry (round-2 plan-review minor 2 -- the blocker-1 bug class):
    # reset ALL inputs positionally; line_x arrives carrying actions.index, receiver_pos
    # carries RangeIndex positions -- realign both, assert equal lengths.
    a = actions.reset_index(drop=True)
    lx = pd.Series(line_x).reset_index(drop=True)
    if receiver_pos is None:  # public callers (round-2 minor 1); add_packing precomputes
        receiver_pos = _resolve_next_touch_positions(actions)
    rp = pd.Series(receiver_pos).reset_index(drop=True)
    if not (len(a) == len(lx) == len(rp)):
        raise ValueError("actions, line_x and receiver_pos must be equal-length")

    out = pd.Series(pd.NA, index=a.index, dtype="boolean", name="packing_secured")
    if len(a) == 0:
        out.index = actions.index
        return out

    if "possession_id" not in a.columns:
        # add_possessions returns a SORTED copy -- realign its ids to a's positional
        # order via a carried position column (robust to non-canonical input order).
        healed = add_possessions(a.assign(_pos_tf49=np.arange(len(a))))
        poss = healed.sort_values("_pos_tf49")["possession_id"].to_numpy()
    else:
        poss = a["possession_id"].to_numpy()

    team = np.asarray(a["team_id"].values)
    typ = a["type_id"].to_numpy()
    time_s = np.asarray(a["time_seconds"].values, dtype=np.float64)
    start_x = np.asarray(a["start_x"].values, dtype=np.float64)
    lx_arr = lx.to_numpy(dtype="float64")
    rp_arr = rp.to_numpy(dtype="object")
    poss_na = pd.isna(poss)  # NA possession never decides (execution-review D6, ADR-027 class)
    window = params.secured_window_seconds

    labels = np.full(len(a), np.nan, dtype=float)  # 1.0 / 0.0 / NaN tri-state
    group_keys = [k for k in ("game_id", "period_id") if k in a.columns]
    groups = a.groupby(group_keys) if group_keys else [(None, a)]
    for _key, grp in groups:
        # Scan in the ROBUST CHRONOLOGICAL order -- (time_seconds, action_id) with action_id
        # as the tiebreak -- the SAME key _resolve_next_touch_positions resolves the reception
        # anchor in (spec 2026-08-20 §3d, ADR-065). A persisted mart may carry a
        # non-chronological action_id (the _finalize_output guard only removes that for FRESH
        # conversions; mart-reading consumers must re-establish order by time_seconds, NOT
        # action_id alone). Scanning action_id-alone diverged from the anchor helper's order and
        # mislabeled non-chronological marts (then raised). The action_id tiebreak keeps
        # time-tied rows in action_id order (execution-review D4 preserved). Mirrors
        # _resolve_next_touch_positions's stable, no-reset sort so `idx` maps into `a`'s positions;
        # NaN time_seconds sorts last (resolves <NA>, never a violation -- finite-only, matching
        # _assert_chronological_action_id).
        _order_cols = [c for c in ("time_seconds", "action_id") if c in grp.columns]
        sorted_grp = grp.sort_values(_order_cols, kind="stable") if _order_cols else grp
        idx = np.asarray(sorted_grp.index)
        t = time_s[idx]
        t_finite = t[np.isfinite(t)]
        t_last = float(t_finite[-1]) if len(t_finite) else 0.0
        rank = {int(p): i for i, p in enumerate(idx)}
        for li in range(len(idx)):
            gi = idx[li]
            if not np.isfinite(lx_arr[gi]) or pd.isna(rp_arr[gi]) or pd.isna(team[gi]):
                continue
            r_pos = int(rp_arr[gi])
            lr_opt = rank.get(r_pos)
            if lr_opt is None:
                continue  # defensive: reception outside this group (helper never emits this)
            lr = lr_opt
            t_r = time_s[r_pos]
            deadline = t_r + window
            label: float | None = None
            if typ[r_pos] in _SHOT_TYPES and same_id(team[r_pos], team[gi]):
                label = 1.0  # first-time shot IS the reception -> decisive retain
            saw_in_window = False
            if label is None:
                for lj in range(lr + 1, len(idx)):
                    gj = idx[lj]
                    if typ[gj] in _SKIP_TYPES or pd.isna(team[gj]):
                        continue  # fouls / non_action / GS null-actor rows never decide
                    in_window = time_s[gj] <= deadline + 1e-9
                    if not in_window and saw_in_window:
                        label = 1.0  # full window observed, no contrary evidence
                        break
                    if typ[gj] in _SHOT_TYPES and same_id(team[gj], team[gi]):
                        label = 1.0
                        break
                    if not same_id(team[gj], team[gi]) and not poss_na[gj] and not poss_na[gi] and poss[gj] != poss[gi]:
                        label = 0.0  # opponent possession boundary (retains() rule; NA never decides)
                        break
                    if in_window:
                        saw_in_window = True
                        if same_id(team[gj], team[gi]) and start_x[gj] < lx_arr[gi]:
                            label = 0.0  # bounce-pass: back behind the line inside the window
                            break
                    else:
                        break  # first event beyond an EMPTY window, undecisive -> arithmetic
            if label is None:
                # retains() truncation arithmetic, reception-anchored.
                label = 1.0 if (t_last - t_r) >= window - 1e-9 else np.nan
            labels[gi] = label

    resolved = np.isfinite(labels)
    out.loc[resolved] = labels[resolved] == 1.0
    out.index = actions.index  # positional reattach; duplicate-safe
    return out
