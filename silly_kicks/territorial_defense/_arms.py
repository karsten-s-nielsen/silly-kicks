"""Territorial-defense counterfactual ARMS (TF-54b).

Arm A (action-anchored, identity-exact) scores ``threat_suppressed = compute_threat_pc(cf) -
compute_threat_pc(actual)`` for ONE defensive-action frame, where ``cf`` is the factual frame with
the defender D removed (attacker-value units; **positive = D's presence suppressed the attacking
team's threat**).

**No arm accepts a ``pitch_control_cache``, and that is CORRECTNESS not performance (ADR-043):**
``PitchControlCache`` keys on frame IDENTITY and EXCLUDES player positions, so a shared cache would
serve the counterfactual leg the factual leg's surface and the delta would collapse to exactly 0
with no warning. The arm computes both legs directly (``compute_threat_pc`` takes no cache), and
``_assert_single_frame_legs`` REFUSES a counterfactual that is not the factual frame minus exactly
one row -- so serving the factual frame unchanged RAISES rather than silently returning 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks._frame_index import group_rows
from silly_kicks.id_compat import ids_match

from ._config import _DEFAULT_PARAMS, TerritorialDefenseParams
from ._engine import remove_player_row

_FRAME_KEYS = ("game_id", "period_id", "frame_id")


def _assert_single_frame_legs(actual: pd.DataFrame, cf: pd.DataFrame, *, fn: str) -> None:
    ak = actual[list(_FRAME_KEYS)].drop_duplicates().reset_index(drop=True)
    ck = cf[list(_FRAME_KEYS)].drop_duplicates().reset_index(drop=True)
    if len(ak) != 1 or len(ck) != 1 or not ak.equals(ck):
        raise ValueError(
            f"{fn}: both legs must be a SINGLE, matching {_FRAME_KEYS}; got actual={ak.to_dict('records')}, "
            f"cf={ck.to_dict('records')}."
        )
    if len(cf) != len(actual) - 1:
        raise ValueError(
            f"{fn}: the counterfactual must be the factual frame MINUS exactly one row (the removed "
            f"defender); got cf n={len(cf)} vs factual n={len(actual)}. Serving the factual frame "
            f"unchanged would collapse the delta to 0 (the ADR-043 landmine)."
        )


def arm_a_threat_suppressed(
    actual_frame: pd.DataFrame,
    cf_frame: pd.DataFrame,
    *,
    attacking_team_id,
    xt,
    goal_map,
    params: TerritorialDefenseParams = _DEFAULT_PARAMS,
) -> float:
    """Arm A (identity-exact): ``compute_threat_pc(cf) - compute_threat_pc(actual)`` for one frame.

    ``cf_frame`` is ``actual_frame`` with defender D removed (see
    :func:`silly_kicks.territorial_defense._engine.remove_player_row`). Positive = D suppressed the
    attacking team's threat. No ``pitch_control_cache`` (ADR-043).
    """
    from silly_kicks import tracking
    from silly_kicks.tracking import SpearmanParams

    _assert_single_frame_legs(actual_frame, cf_frame, fn="arm_a_threat_suppressed")
    pcp = SpearmanParams(lambda_gk=params.lambda_gk)
    a = tracking.compute_threat_pc(
        actual_frame,
        attacking_team_id=attacking_team_id,
        xt=xt,
        goal_map=goal_map,
        method=params.pitch_control_method,
        params=pcp,
    )
    c = tracking.compute_threat_pc(
        cf_frame,
        attacking_team_id=attacking_team_id,
        xt=xt,
        goal_map=goal_map,
        method=params.pitch_control_method,
        params=pcp,
    )
    return float(c - a)


def arm_a_threat_suppressed_batch(
    actual_frames: pd.DataFrame,
    cf_frames: pd.DataFrame,
    *,
    attacking_team_id_by_frame,
    xt,
    goal_map,
    params: TerritorialDefenseParams = _DEFAULT_PARAMS,
) -> pd.Series:
    """Loop :func:`arm_a_threat_suppressed` over each ``(game_id, period_id, frame_id)`` group.

    ``attacking_team_id_by_frame`` is a scalar (applied to every frame) OR a dict keyed by the
    frame-key tuple. Returns a Series indexed by the frame keys. No ``pitch_control_cache`` (ADR-043).
    """
    # ADR-019/ADR-068: group both legs via group_rows (canonical keys, O(1) lookup) rather than a raw
    # `dict(tuple(df.groupby(...)))`, whose raw-tuple keys would mis-compare across a dtype skew.
    a = group_rows(actual_frames, _FRAME_KEYS)
    c = group_rows(cf_frames, _FRAME_KEYS)
    a_keys, c_keys = set(a.keys()), set(c.keys())
    if a_keys != c_keys:
        raise ValueError(
            "arm_a_threat_suppressed_batch: actual and counterfactual frames cover different "
            f"{_FRAME_KEYS} groups: {sorted(a_keys ^ c_keys)}."
        )
    per_frame = isinstance(attacking_team_id_by_frame, dict)
    out: dict = {}
    for key in a.keys():
        atk = attacking_team_id_by_frame[key] if per_frame else attacking_team_id_by_frame
        out[key] = arm_a_threat_suppressed(
            a.get(*key), c.get(*key), attacking_team_id=atk, xt=xt, goal_map=goal_map, params=params
        )
    s = pd.Series(out, name="a_threat_suppressed", dtype="float64")
    if len(s):
        s.index = pd.MultiIndex.from_tuples(list(s.index), names=list(_FRAME_KEYS))
    return s


# --- Arm B (hull-based, attribution-APPROXIMATE) -----------------------------------------------


def nearest_defender_pos(frame: pd.DataFrame, target_xy, *, defending_team_id) -> int:
    """Positional index of the defending-team (non-ball) row nearest ``target_xy``; -1 if none."""
    is_ball = frame["is_ball"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    on_def = ids_match(frame["team_id"], defending_team_id).to_numpy(dtype=bool)
    mask = on_def & ~is_ball
    if not mask.any():
        return -1
    xs = frame["x"].to_numpy(dtype=float)
    ys = frame["y"].to_numpy(dtype=float)
    d2 = (xs - float(target_xy[0])) ** 2 + (ys - float(target_xy[1])) ** 2
    d2 = np.where(mask, d2, np.inf)
    return int(np.argmin(d2))


def arm_b_threat_suppressed(
    frame: pd.DataFrame,
    *,
    target_xy,
    defending_team_id,
    attacking_team_id,
    xt,
    goal_map,
    params: TerritorialDefenseParams = _DEFAULT_PARAMS,
) -> tuple[float, int]:
    """Arm B (attribution-approximate): remove the defending-team player NEAREST ``target_xy`` (the
    contesting defender, chosen by POSITION -- ``arm_b_rule="nearest_to_target"``) and difference the
    threat: ``(compute_threat_pc(cf) - compute_threat_pc(actual), removed_pos)``.

    ``target_xy`` is in FRAME coordinates (the caller reflects the opponent pass end into the frame's
    convention). Positive delta = the contesting defender suppressed the attacking team's threat. No
    ``pitch_control_cache`` (ADR-043). ``removed_pos = -1`` (delta NaN) when no defender is present.
    """
    from silly_kicks import tracking
    from silly_kicks.tracking import SpearmanParams

    pos = nearest_defender_pos(frame, target_xy, defending_team_id=defending_team_id)
    if pos < 0:
        return float("nan"), -1
    cf = remove_player_row(frame, player_pos=pos)
    pcp = SpearmanParams(lambda_gk=params.lambda_gk)
    a = tracking.compute_threat_pc(
        frame,
        attacking_team_id=attacking_team_id,
        xt=xt,
        goal_map=goal_map,
        method=params.pitch_control_method,
        params=pcp,
    )
    c = tracking.compute_threat_pc(
        cf,
        attacking_team_id=attacking_team_id,
        xt=xt,
        goal_map=goal_map,
        method=params.pitch_control_method,
        params=pcp,
    )
    return float(c - a), pos


def contesting_defender_is_d(frame: pd.DataFrame, target_xy, *, defending_team_id, d_player_id) -> float:
    """Arm-B ATTRIBUTION check: is the nearest-to-target defending player actually D?

    Returns ``1.0`` (the position-chosen contesting defender IS D), ``0.0`` (a KNOWN defender that is
    NOT D), or ``float("nan")`` when the identity is **un-measurable** -- no defender present, OR the
    nearest defender is ANONYMOUS. On an SB360 freeze-frame only the ACTOR carries a real id (the actor
    bridge stamps the passer, not the defenders), so a non-actor defender's snapshot-numbered id cannot
    be compared to D's real id: NaN, **never a fabricated ``0.0``** (ADR-027). The caller reports the
    complement over the MEASURABLE (non-NaN) subset -- ``b_attribution_slippage = 1 - mean(skipna)``, the
    NOT-D rate (attribution error, lower = tighter) -- so the slippage is honest-NaN on an all-anonymous
    corpus (real SB360) rather than a structural value for every defender. Full-tracking frames (no
    ``is_actor`` column, real ids throughout) are measurable end to end.
    """
    from silly_kicks.id_compat import same_id

    pos = nearest_defender_pos(frame, target_xy, defending_team_id=defending_team_id)
    if pos < 0:
        return float("nan")
    if "is_actor" in frame.columns:
        # SB360 snapshot: only the actor's id is real; a non-actor defender is anonymous -> un-measurable.
        is_actor = frame["is_actor"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        if not bool(is_actor[pos]):
            return float("nan")
    return 1.0 if same_id(frame.iloc[pos]["player_id"], d_player_id) else 0.0
