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
    frame is empty, endpoints are non-finite, or no eligible defender exists.
    Defender extraction + away-mirror duplicated from _structural_pass.py by design.

    Examples
    --------
    Compute packing metrics for a single pass on a frame::

        from silly_kicks.tracking import compute_packing_metrics
        m = compute_packing_metrics(
            frame, attacking_team_id=1, home_team_id=1,
            passer_xy=(50.0, 34.0), receiver_xy=(70.0, 34.0),
        )
        m["packing_made"]
    """
    if params is None:
        params = PackingParams()

    if frame is None or len(frame) == 0:
        return dict(_NAN_METRICS)
    if not all(np.isfinite(v) for v in (*passer_xy, *receiver_xy)):
        return dict(_NAN_METRICS)

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
    ok = np.isfinite(dx_)
    dx_ = dx_[ok]
    if dx_.size == 0:
        return dict(_NAN_METRICS)

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

    p, r = np.asarray(passer_xy, float), np.asarray(receiver_xy, float)
    made = float(np.count_nonzero((dx_ > p[0]) & (dx_ <= r[0])))
    bypassed = dx_[(dx_ > p[0]) & (dx_ <= r[0])]
    line_x = float(bypassed.max()) if bypassed.size else np.nan

    lo, hi = min(p[0], r[0]), max(p[0], r[0])
    interval = float(np.count_nonzero((dx_ > lo) & (dx_ <= hi)))
    net = _direction_multiplier(r[0] - p[0], r[1] - p[1], params) * interval

    # Goal-threat: select-then-mirror. select_back_line_players wants the DEFENDING
    # team's id (its "own goal" is the defending team's) -- resolve it NaN-safely from
    # the frame's non-attacking players (review blocker 3). Caveat: the helper
    # short-circuits len(outfield) < 3 -> returns outfield unselected (sparse frames).
    def_team_vals = opp_all["team_id"].dropna().unique()
    if len(def_team_vals) == 0:
        gt = np.nan
    else:
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
        back = select_back_line_players(
            frame,
            def_team_vals[0],
            _def_end == 0.0,
            n=params.back_line_n,
        )
        if len(back) == 0:
            gt = np.nan
        else:
            bx = back["x"].to_numpy(dtype="float64")
            bx = bx[np.isfinite(bx)]
            if mirror:
                bx = 105.0 - bx
            gt = float(np.count_nonzero((bx > p[0]) & (bx <= r[0])))

    return {"packing_made": made, "packing_net": net, "packing_goal_threat": gt, "line_x": line_x}


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
    receiver -> <NA>. The scan runs in ``action_id`` (canonical play) order within
    each ``(game_id, period_id)`` group -- the same order the positions helper
    resolves anchors in -- with non-decreasing ``time_seconds`` enforced in that
    order. A caller-supplied ``possession_id`` with missing values never decides
    the boundary test (NA-routed, the ADR-027 discipline).

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
        # Scan in ACTION_ID (canonical play) order -- the SAME order the positions
        # helper resolved the anchor in. Scanning positionally instead flips labels
        # on time-tied rows whose positional order differs (execution-review D4).
        sorted_grp = grp.sort_values("action_id", kind="stable") if "action_id" in grp.columns else grp
        idx = np.asarray(sorted_grp.index)
        t = time_s[idx]
        if len(t) > 1 and not (np.diff(t) >= -1e-9).all():
            raise ValueError("time_seconds must be non-decreasing within each (game_id, period_id) group")
        t_last = t[-1] if len(t) else 0.0
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
