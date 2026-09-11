"""Territorial-defense counterfactual engine (TF-54b).

Domain selection, the model-free REMOVAL counterfactual frame (Fernandez-Bornn marginal player
value), the SPEC-02 FOV local-completeness gate, the PLAN-01 removal-depletion guard, and the
ADR-042 conservation report. Scoring (the ``compute_threat_pc`` differencing) lives in ``_arms.py``.

All gates are dropped-AND-COUNTED (ADR-042): a frame that cannot be scored is assigned a
``td_source`` reason and counted in the report, never a fabricated 0.
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pandas as pd

from silly_kicks._frame_index import group_rows
from silly_kicks.id_compat import canonical_id, ids_match
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking import GoalMap, region_observed_fraction

from ._config import _DEFAULT_PARAMS, TerritorialDefenseParams
from ._report import TerritorialDefenseReport

_FRAME_KEYS = ("game_id", "period_id", "frame_id")

# Drop-reason tokens Arm A produces (the td_source COLUMN is a subset of TD_SOURCE_VALUES; the two
# Arm-B-only reasons -- missing_frame / non_finite_delta -- live in report.arm_b_drop_reasons, not the
# column). Velocity is never a drop reason: the threat integral is an ADR-063 Tier-1 lift valid at zero
# velocity.
SCORED = "scored"
NO_ACTOR = "no_actor"
NO_DEFENDERS = "no_defenders"
UNRESOLVED_GEOMETRY = "unresolved_geometry"
FOV_CROPPED_LOCAL = "fov_cropped_local"
REMOVAL_UNDERSUPPORTED = "removal_undersupported"


def action_ltr_goal_map(game_id, period_id, *, acting_team_id, opponent_team_id) -> GoalMap:
    """Per-action-LTR goal map for ONE frame (ADR-028): the acting team of the frame's action attacks
    x=105 (defends 0); the opponent defends x=105.

    Correct by the per-action-LTR CONVENTION -- no GK, so it is FOV-proof where a per-frame
    :func:`resolve_defended_goals` would fail on a keeperless freeze-frame, and unambiguous where a
    per-MATCH ``resolve_defended_goals`` is bimodal (a team's keeper sits at x=0 in its own actions and
    x=105 in the opponent's). This resolves a DIFFERENT frame convention from ground truth than
    ``resolve_defended_goals`` (which ESTIMATES the match-oriented defended end from mean GK x); it is
    NOT an ADR-055 fork -- both return a :class:`GoalMap` and route through its real opponent-lookup
    ``attacked_goal``. Keys canonical (ADR-055 rule 2), so a Python ``int``/``str`` lookup resolves.

    Examples
    --------
    >>> gm = action_ltr_goal_map(7, 1, acting_team_id=1, opponent_team_id=2)
    >>> gm.attacked_goal(7, 1, 1, allow_guess=True)   # acting team attacks the opponent's end
    105.0
    >>> gm.attacked_goal(7, 1, 2, allow_guess=True)   # opponent attacks the acting team's end
    0.0
    """
    g, p = canonical_id(game_id), canonical_id(period_id)
    fl = float(spadlconfig.field_length)
    resolved = {
        (g, p, canonical_id(acting_team_id)): 0.0,
        (g, p, canonical_id(opponent_team_id)): fl,
    }
    return GoalMap(MappingProxyType(resolved), MappingProxyType({}), frozenset())


def remove_player_row(frame: pd.DataFrame, *, player_pos: int) -> pd.DataFrame:
    """Return a copy of ``frame`` with the row at positional index ``player_pos`` removed. PURE.

    Examples
    --------
    >>> import pandas as pd
    >>> f = pd.DataFrame({"player_id": [0, 1, 2], "x": [1.0, 2.0, 3.0]})
    >>> remove_player_row(f, player_pos=1)["player_id"].tolist()
    [0, 2]
    """
    keep = [i for i in range(len(frame)) if i != player_pos]
    return frame.iloc[keep].reset_index(drop=True)


def _defender_count(frame: pd.DataFrame, defending_team_id) -> int:
    is_ball = frame["is_ball"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    on_def = ids_match(frame["team_id"], defending_team_id).to_numpy(dtype=bool)
    return int((on_def & ~is_ball).sum())


def removal_leaves_enough_defenders(frame: pd.DataFrame, *, defending_team_id, min_after: int) -> bool:
    """True iff removing ONE defending-team player leaves >= ``min_after`` defending-team rows.

    PLAN-01 guard: a 0-defender counterfactual degrades to attacker-controls-all (the spearman
    surface's ``def_tti.shape[0] > 0`` branch does not raise), an upward-biased outlier -- so a
    frame that cannot leave enough defenders is dropped ``removal_undersupported``, never scored.
    """
    return (_defender_count(frame, defending_team_id) - 1) >= min_after


def _disk_polygon(center_xy: tuple[float, float], radius_m: float, n: int = 24) -> np.ndarray:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([cx + radius_m * np.cos(ang), cy + radius_m * np.sin(ang)])


def local_completeness_ok(polygon, center_xy, *, radius_m: float, min_fraction: float) -> bool:
    """SPEC-02: the disk of radius ``radius_m`` around ``center_xy`` must be >= ``min_fraction``
    observed by the ``visible_area`` polygon. A missing/degenerate polygon or a NaN fraction is
    NOT observed (False) -- removal biases the delta upward on an under-observed re-absorbing
    neighbourhood, so such a frame is dropped ``fov_cropped_local``.
    """
    if polygon is None or len(polygon) < 3:
        return False
    frac = region_observed_fraction(polygon, _disk_polygon(center_xy, radius_m))
    return bool(np.isfinite(frac) and frac >= min_fraction)


def select_arm_a_domain(actions: pd.DataFrame, *, params: TerritorialDefenseParams = _DEFAULT_PARAMS) -> pd.DataFrame:
    """One row per D defensive-intervention action (Arm-A domain).

    Columns: ``game_id, period_id, action_id, frame_id, defender_id, defending_team_id,
    attacking_team_id`` (``frame_id == action_id``; ``attacking_team_id`` = the OTHER of the match's
    two teams, ``pd.NA`` if not exactly two are present).
    """
    mask = actions["type_id"].isin(list(params.defensive_action_type_ids))
    dom = actions.loc[mask, ["game_id", "period_id", "action_id", "team_id", "player_id"]].copy()
    dom = dom.rename(columns={"team_id": "defending_team_id", "player_id": "defender_id"})
    dom["frame_id"] = dom["action_id"]
    # attacking_team_id = the OTHER of the match's two teams, resolved PER GAME. A multi-match batch
    # spans >2 teams, so a global unique() would yield NA for every row and Arm A would silently score
    # nothing (Arm B is already per-game). NA when a game lacks exactly two teams. Keys canonical (ADR-019).
    pairs: dict = {}
    for gid, grp in actions.groupby(actions["game_id"].map(canonical_id), sort=False):
        gt = list(pd.unique(grp["team_id"].dropna()))
        pairs[gid] = {canonical_id(gt[0]): gt[1], canonical_id(gt[1]): gt[0]} if len(gt) == 2 else {}
    dom["attacking_team_id"] = [
        pairs.get(canonical_id(g), {}).get(canonical_id(d), pd.NA)
        for g, d in zip(dom["game_id"], dom["defending_team_id"], strict=False)
    ]
    return dom.reset_index(drop=True)


def _polygons_by_action(visible_area) -> dict:
    if visible_area is None:
        return {}
    return {canonical_id(a): p for a, p in zip(visible_area["action_id"], visible_area["polygon"], strict=False)}


def _classify_one(row, groups, polys, params, goal_map_for) -> tuple[str, int]:
    fr = groups.get(row.game_id, row.period_id, row.frame_id)
    if len(fr) == 0:
        return NO_ACTOR, -1
    if "is_actor" in fr.columns:
        is_actor = fr["is_actor"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    else:
        is_actor = np.zeros(len(fr), dtype=bool)
    if not is_actor.any():
        return NO_ACTOR, -1
    dpos = int(np.flatnonzero(is_actor)[0])
    # Convention-agnostic: the compute-owned factory resolves the goal for THIS frame (per-action-LTR
    # per-frame map, or the shared per-match map). Mirror the arm's consumer -- compute_threat_pc reads
    # attacked_goal(attacking_team) (the END it attacks), NOT .get -- so pre-check the same end.
    gm = goal_map_for(row.game_id, row.period_id, row.defending_team_id, row.attacking_team_id)
    if gm.attacked_goal(row.game_id, row.period_id, row.attacking_team_id, allow_guess=True) is None:
        return UNRESOLVED_GEOMETRY, -1
    if _defender_count(fr, row.defending_team_id) == 0:
        return NO_DEFENDERS, -1
    if not removal_leaves_enough_defenders(
        fr, defending_team_id=row.defending_team_id, min_after=params.min_defenders_after_removal
    ):
        return REMOVAL_UNDERSUPPORTED, -1
    poly = polys.get(canonical_id(row.action_id))
    if poly is not None:
        d_xy = (float(fr.iloc[dpos]["x"]), float(fr.iloc[dpos]["y"]))
        if not local_completeness_ok(
            poly, d_xy, radius_m=params.local_radius_m, min_fraction=params.min_local_observed_fraction
        ):
            return FOV_CROPPED_LOCAL, -1
    return SCORED, dpos


def classify_arm_a_domain(
    domain: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    params: TerritorialDefenseParams = _DEFAULT_PARAMS,
    visible_area=None,
    goal_map_for,
    groups=None,
) -> pd.DataFrame:
    """Assign ``td_source`` per Arm-A candidate via the first-failing cascade (no scoring).

    Cascade order (deterministic): ``no_actor -> unresolved_geometry -> no_defenders ->
    removal_undersupported -> fov_cropped_local -> scored``. Adds ``td_source`` and ``defender_pos``
    (positional index of D in its frame; ``-1`` when not scored) so the arm can score without
    re-finding D. Convention-AGNOSTIC: ``goal_map_for(game_id, period_id, acting_team_id,
    opponent_team_id) -> GoalMap`` is the compute-owned factory (per-frame ``action_ltr_goal_map`` for
    ``per_action_ltr``, or a closure over the per-match ``resolve_defended_goals`` map for ``match_ltr``);
    this function never sees ``frame_convention``. ``groups`` is a precomputed
    ``group_rows(frames, _FRAME_KEYS)`` (the ``links=`` pre-build-and-thread idiom); built locally when
    ``None``.
    """
    if groups is None:
        groups = group_rows(frames, _FRAME_KEYS)
    polys = _polygons_by_action(visible_area)
    sources: list[str] = []
    dposs: list[int] = []
    for row in domain.itertuples():
        src, dpos = _classify_one(row, groups, polys, params, goal_map_for)
        sources.append(src)
        dposs.append(dpos)
    out = domain.copy()
    out["td_source"] = sources
    out["defender_pos"] = dposs
    return out


def build_report(
    td_source, *, params: TerritorialDefenseParams, n_frames_in: int, arm_b: dict | None = None
) -> TerritorialDefenseReport:
    """Assemble the conserving report (ADR-042): Arm A over its domain from ``td_source``, Arm B over
    its ``(defender, in-hull-pass)`` pairs from the optional ``arm_b`` census
    (``{"n_in", "n_scored", "drop_reasons"}``)."""
    s = pd.Series(list(td_source), dtype="object")
    scored = int((s == SCORED).sum())
    drops = {str(k): int(v) for k, v in s[s != SCORED].value_counts().to_dict().items()}
    ab = arm_b or {}
    return TerritorialDefenseReport(
        params=params,
        n_frames_in=int(n_frames_in),
        n_frames_scored=scored,
        drop_reasons=drops,
        arm_b_n_in=int(ab.get("n_in", 0)),
        arm_b_n_scored=int(ab.get("n_scored", 0)),
        arm_b_drop_reasons={str(k): int(v) for k, v in ab.get("drop_reasons", {}).items()},
    )
