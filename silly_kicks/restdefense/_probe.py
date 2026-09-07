"""restdefense Layer-3 instrument-validity probe (TF-60, ADR-089).

Mirrors ``gkdv/_probe.py`` (TF-19 A+2, ADR-082): a dose imposer substitutes only A's rearguard (or
keeper) at an imposed position, reusing the :func:`._counterfactual.build_restdefense_ghost_frames`
domain/provenance so the scored set matches the arm exactly; paired-vector controls displace ONE of
A's OTHER outfielders by the per-frame representative dose vector (a single-player null). The two
pooled-corpus verdict FUNCTIONS are REUSED verbatim from ``silly_kicks.gkdv`` -- restdefense inherits
the TF-19-established physics-arm thresholds WITHOUT importing any private gkdv constant (the §10
import-allowlist rule): ``SATURATING_MULTIPLE`` / ``MIN_DOMAIN_FRAMES`` / ``PHYSICS_ARM_PROBE_RATIO``
are private module constants inside ``gkdv/_probe.py`` and the verdict functions read them internally.

Reported-not-gated (repo convention). Depends on ``silly_kicks.tracking`` / ``silly_kicks.gkdv``
PUBLIC seams + ``silly_kicks.id_compat`` + ``silly_kicks._frame_index`` ONLY (ADR-037).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

# Public verdict FUNCTIONS only (they read the private gkdv thresholds internally); re-exported so the
# driver can import the whole probe surface from one place, exactly as gkdv/_probe defines them.
from silly_kicks.gkdv import layer0_instrument_verdict, layer1_responsiveness_verdict
from silly_kicks.id_compat import align_join_keys, canonical_id
from silly_kicks.tracking import resolve_defended_goals

from ._columns import (
    RD_FRAME_KEYS,
    RD_GK_DETER_SPACE,
    RD_GK_DETER_THREAT,
    RD_OUTFIELD_DETER_SPACE,
    RD_OUTFIELD_DETER_THREAT,
)
from ._config import RestDefenseParams
from ._counterfactual import _goal_lookup, build_restdefense_ghost_frames

__all__ = [
    "EXPECTED_DIRECTION",
    "LADDER_M",
    "REARGUARD_REALISTIC_MIN_DISP_M",
    "SATURATING_X30_GR",
    "expected_direction_for_arm",
    "impose_rearguard_dose",
    "layer0_instrument_verdict",
    "layer1_responsiveness_verdict",
    "paired_vector_controls",
]

_DEFAULT_PARAMS = RestDefenseParams()

#: The discrete Layer-0 doses. `ladder` additionally takes a `displacement` (metres).
Dose = Literal["realistic", "ladder", "saturating_goalline", "saturating_x30"]

#: Imposed ladder displacements (mirrors gkdv LADDER_M; restdefense-local, not imported).
LADDER_M: tuple[float, float, float] = (2.0, 3.0, 4.0)
#: The per-frame |ghost - actual| max floor for the realistic dose (the ghost's ~m-scale MAE).
REARGUARD_REALISTIC_MIN_DISP_M: float = 2.0
#: Goal-relative x of the second saturating position.
SATURATING_X30_GR: float = 30.0

#: Per-arm expected sign (attacker-value units; `negative == deterrent` for ALL four rd arms). Keyed
#: on the ARM OUTPUT column directly (unlike gkdv's two-level column->key->direction bridge, because
#: the rd arm columns ARE the direction keys). An unmapped arm raises rather than silently skipping.
EXPECTED_DIRECTION: dict[str, str] = {
    RD_GK_DETER_THREAT: "negative",
    RD_GK_DETER_SPACE: "negative",
    RD_OUTFIELD_DETER_THREAT: "negative",
    RD_OUTFIELD_DETER_SPACE: "negative",
}


def expected_direction_for_arm(arm_column: str) -> str:
    """The expected sign for a rest-defense arm's OUTPUT column (``"negative"`` == deterrent).

    An arm column absent from :data:`EXPECTED_DIRECTION` raises ``KeyError`` (never a silent skip),
    mirroring gkdv's :func:`expected_direction_for_arm`.

    Examples
    --------
    >>> from silly_kicks.restdefense._probe import expected_direction_for_arm
    >>> expected_direction_for_arm("rd_outfield_deter_threat")
    'negative'
    """
    return EXPECTED_DIRECTION[arm_column]


def _scored_targets(frames: pd.DataFrame, provenance: pd.DataFrame, goal_map) -> pd.DataFrame:
    """Per-scored (frame, team=A, player) dose-target rows sourced from the engine's provenance.

    Carries ``actual_x/actual_y`` + ``ghost_x/ghost_y`` + ``displacement_m`` (from provenance) and
    ``own_goal_x`` (A's defended goal end, from the ``GoalMap``) -- everything the dose geometry +
    :func:`paired_vector_controls` read. The engine has already validated the scored set (finite ghost,
    committed-forward, GoalMap resolvable), so this only annotates it.
    """
    scored = provenance[provenance["drop_reason"].isna()].copy()
    cols = [*RD_FRAME_KEYS, "team_id", "player_id", "actual_x", "actual_y", "ghost_x", "ghost_y", "displacement_m"]
    scored = scored[cols].reset_index(drop=True)
    scored["own_goal_x"] = [
        _goal_lookup(goal_map, g, p, t)
        for g, p, t in zip(scored["game_id"], scored["period_id"], scored["team_id"], strict=True)
    ]
    return scored


def impose_rearguard_dose(
    frames: pd.DataFrame,
    *,
    which: Literal["keeper", "rearguard"],
    home_team_id: int | str,
    dose: Dose,
    displacement: float | None = None,
    model=None,
    carrier: pd.DataFrame | None = None,
    visible_area: pd.DataFrame | None = None,
    params: RestDefenseParams = _DEFAULT_PARAMS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Substitute ONLY A's rearguard (``which="rearguard"``) or keeper (``which="keeper"``) at the
    ``dose`` position. PURE (a new frame). Returns ``(imposed_frames, targets)``.

    ``targets`` carries the per-scored (frame, team=A, player) contract (``RD_FRAME_KEYS``, ``team_id``,
    ``player_id``, ``actual_x/actual_y``, ``own_goal_x``, ``ghost_x/ghost_y``, ``displacement_m``) plus
    ``imp_x/imp_y`` (the applied dose) and the per-frame representative vector ``frame_vec_x/frame_vec_y``
    (the MEAN ``(imp - actual)`` over that frame's targets) that :func:`paired_vector_controls` displaces
    a single control player by.

    The doses generalize the gkdv keeper doses to the (possibly multi-player) rearguard set: ``ladder``
    moves each target ``displacement`` m toward A's own goal; ``saturating_goalline`` puts each on A's
    own goal line (keeping its lateral y, so the set is not laterally collapsed); ``saturating_x30`` puts
    each at goal-relative x=30; ``realistic`` uses the model's own ghost position, keeping only frames
    whose max rearguard displacement clears :data:`REARGUARD_REALISTIC_MIN_DISP_M`.

    Examples
    --------
    Impose a 2 m ladder dose on the in-possession rearguard for a Layer-1 responsiveness probe::

        imposed, targets = impose_rearguard_dose(
            frames, which="rearguard", home_team_id=1, dose="ladder", displacement=2.0, model=model
        )
    """
    _cf, provenance, _report = build_restdefense_ghost_frames(
        frames,
        which=which,
        model=model,
        home_team_id=home_team_id,
        carrier=carrier,
        visible_area=visible_area if which == "rearguard" else None,
        params=params,
    )
    goal_map = resolve_defended_goals(frames)
    targets = _scored_targets(frames, provenance, goal_map)
    empty_cols = [
        *RD_FRAME_KEYS,
        "team_id",
        "player_id",
        "actual_x",
        "actual_y",
        "own_goal_x",
        "ghost_x",
        "ghost_y",
        "displacement_m",
        "imp_x",
        "imp_y",
        "frame_vec_x",
        "frame_vec_y",
    ]
    if not len(targets):
        return frames.copy(), pd.DataFrame(columns=empty_cols)

    own_goal = targets["own_goal_x"].to_numpy(dtype=float)
    actual_x = targets["actual_x"].to_numpy(dtype=float)
    actual_y = targets["actual_y"].to_numpy(dtype=float)

    if dose == "saturating_goalline":
        imp_x = own_goal.copy()
        imp_y = actual_y.copy()
    elif dose == "saturating_x30":
        imp_x = np.where(own_goal == 0.0, SATURATING_X30_GR, own_goal - SATURATING_X30_GR)
        imp_y = actual_y.copy()
    elif dose == "ladder":
        if displacement is None:
            raise ValueError("dose='ladder' requires displacement=")
        sign = np.where(own_goal == 0.0, -1.0, 1.0)  # toward A's own goal
        imp_x = actual_x + sign * float(displacement)
        imp_y = actual_y.copy()
    elif dose == "realistic":
        imp_x = targets["ghost_x"].to_numpy(dtype=float)
        imp_y = targets["ghost_y"].to_numpy(dtype=float)
        # Keep a FRAME iff its rearguard max displacement clears the floor (a coherent whole-set move,
        # never a partial rearguard). Frame keys are canonicalised for the group-max.
        fkey = list(zip(_idk(targets["game_id"]), _idk(targets["period_id"]), _idk(targets["frame_id"]), strict=True))
        disp = targets["displacement_m"].to_numpy(dtype=float)
        frame_max: dict = {}
        for k, d in zip(fkey, disp, strict=True):
            frame_max[k] = max(frame_max.get(k, 0.0), float(d))
        keep = np.array([frame_max[k] >= REARGUARD_REALISTIC_MIN_DISP_M for k in fkey], dtype=bool)
        targets = targets.loc[keep].reset_index(drop=True)
        imp_x, imp_y = imp_x[keep], imp_y[keep]
        actual_x, actual_y = actual_x[keep], actual_y[keep]
    else:
        raise ValueError(f"unknown dose: {dose!r}")

    targets = targets.copy()
    targets["imp_x"] = imp_x
    targets["imp_y"] = imp_y
    targets = _attach_frame_vector(targets)
    imposed = _substitute(frames, targets)
    return imposed, targets


def _attach_frame_vector(targets: pd.DataFrame) -> pd.DataFrame:
    """Add ``frame_vec_x/frame_vec_y`` = per-frame MEAN ``(imp - actual)`` (broadcast to each row).

    For ``which="keeper"`` (one target per frame) this is exactly the keeper's own displacement, so the
    paired control is byte-equivalent to the gkdv single-player control.
    """
    if not len(targets):
        targets["frame_vec_x"] = pd.Series(dtype=float)
        targets["frame_vec_y"] = pd.Series(dtype=float)
        return targets
    dx = targets["imp_x"].to_numpy(dtype=float) - targets["actual_x"].to_numpy(dtype=float)
    dy = targets["imp_y"].to_numpy(dtype=float) - targets["actual_y"].to_numpy(dtype=float)
    tmp = targets[RD_FRAME_KEYS].copy()
    tmp["_dx"] = dx
    tmp["_dy"] = dy
    means = tmp.groupby(RD_FRAME_KEYS, dropna=False, sort=False)[["_dx", "_dy"]].transform("mean")
    out = targets.copy()
    out["frame_vec_x"] = means["_dx"].to_numpy(dtype=float)
    out["frame_vec_y"] = means["_dy"].to_numpy(dtype=float)
    return out


def _substitute(frames: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Move A's target (frame, team, player) rows to ``imp_x/imp_y``. PURE (new frame).

    Only ``x``/``y`` are overwritten (mirroring ``_counterfactual._write_back`` -- the ghost/dose keeps
    the actual velocity so the dose is comparable to the arm's ghost leg)."""
    out = frames.copy()
    if not len(targets):
        return out
    non_ball = ~out["is_ball"].astype(bool)
    side = out.loc[non_ball, [*RD_FRAME_KEYS, "team_id", "player_id"]]
    move = targets[[*RD_FRAME_KEYS, "team_id", "player_id", "imp_x", "imp_y"]]
    left, right = align_join_keys(side, move, [*RD_FRAME_KEYS, "team_id", "player_id"])
    joined = left.merge(right, on=[*RD_FRAME_KEYS, "team_id", "player_id"], how="left")
    joined.index = side.index
    hit = joined["imp_x"].notna().to_numpy() & joined["imp_y"].notna().to_numpy()
    idx = joined.index[hit]
    if len(idx):
        out.loc[idx, "x"] = joined.loc[idx, "imp_x"].to_numpy(dtype=float)
        out.loc[idx, "y"] = joined.loc[idx, "imp_y"].to_numpy(dtype=float)
    return out


def _idk(values) -> list:
    return [canonical_id(v) for v in values]


def paired_vector_controls(
    frames: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    r: int,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Single-player paired-vector controls (parent idiom ``tracking/_model_eval.py``): displace ONE of
    A's OTHER outfielders per control by the per-frame representative dose vector.

    Returns ``{"nearest": frames, "placebo_0": frames, ..., "placebo_{r-1}": frames}`` -- the outfielder
    NEAREST the target centroid (the rearguard/keeper-comparable single-player control) + ``r`` SINGLE
    random outfielders (the placebo band). Each is a NEW frame (``frames`` is never mutated). Moving ONE
    player per control keeps ``nearest`` and each placebo DISTINCT single-player quantities, so the
    Layer-1 ``max(nd_med, placebo_p95)`` is meaningful.

    The pool is A's outfielders that are NOT dose targets (for ``which="rearguard"`` this excludes the
    rearguard set; for ``which="keeper"`` the keeper is a GK, absent from the outfield pool, so the pool
    is all of A's outfielders -- byte-equivalent to the gkdv control). ADR-068: the pool rows are
    grouped ONCE via :func:`silly_kicks._frame_index.group_rows`. ``targets`` must carry ``imp_x``/
    ``imp_y`` / ``actual_x`` / ``actual_y`` / ``frame_vec_x`` / ``frame_vec_y`` / ``team_id`` -- exactly
    what :func:`impose_rearguard_dose` returns.

    Examples
    --------
    Build single-player paired-vector controls for the imposed dose (nearest + r placebos)::

        controls = paired_vector_controls(frames, targets, r=3, rng=np.random.default_rng(0))
        nd_control = controls["nearest"]  # one displaced A-outfielder, comparable to the dose
    """
    from silly_kicks._frame_index import group_rows  # ADR-068 grouping seam (imports only id_compat)

    control_names = ["nearest", *(f"placebo_{k}" for k in range(int(r)))]
    if not len(targets):
        return {name: frames.copy() for name in control_names}

    # Pool = A's outfielders that are NOT dose targets. The target exclusion happens ONCE (an anti-join
    # before grouping), so the per-frame loop is O(1) group lookups -- never a rescan (ADR-073).
    outfield = frames[(~frames["is_ball"].astype(bool)) & (~frames["is_goalkeeper"].astype(bool))]
    tkeys = targets[[*RD_FRAME_KEYS, "team_id", "player_id"]].drop_duplicates().copy()
    tkeys["_is_target"] = True
    left, right = align_join_keys(outfield, tkeys, [*RD_FRAME_KEYS, "team_id", "player_id"])
    marked = left.merge(right, on=[*RD_FRAME_KEYS, "team_id", "player_id"], how="left")
    pool = outfield[marked["_is_target"].isna().to_numpy()]
    groups = group_rows(pool, (*RD_FRAME_KEYS, "team_id"))

    # One control per scored (frame, team=A): the target centroid + the frame's representative vector.
    frame_ref = (
        targets.groupby([*RD_FRAME_KEYS, "team_id"], dropna=False, sort=False)
        .agg(cx=("actual_x", "mean"), cy=("actual_y", "mean"), vx=("frame_vec_x", "first"), vy=("frame_vec_y", "first"))
        .reset_index()
    )

    # Numeric fields as float arrays indexed positionally (itertuples attrs are typed `Scalar` and are
    # NOT float()-able under pyright); the itertuples attrs are used ONLY as canonicalised group keys.
    cxs = frame_ref["cx"].to_numpy(dtype=float)
    cys = frame_ref["cy"].to_numpy(dtype=float)
    vxs = frame_ref["vx"].to_numpy(dtype=float)
    vys = frame_ref["vy"].to_numpy(dtype=float)
    picks: dict[str, list[tuple[object, float, float]]] = {name: [] for name in control_names}
    for i, row in enumerate(frame_ref.itertuples(index=False)):
        sub = groups.get(row.game_id, row.period_id, row.frame_id, row.team_id)
        if sub.empty:
            continue
        cx, cy = float(cxs[i]), float(cys[i])
        dx, dy = float(vxs[i]), float(vys[i])
        dist = np.hypot(sub["x"].to_numpy(dtype=float) - cx, sub["y"].to_numpy(dtype=float) - cy)
        order = np.argsort(dist, kind="stable")  # positional-in-`sub`; nearest to the target centroid first
        pool_labels = sub.index.to_numpy()
        picks["nearest"].append((sub.index[int(order[0])], dx, dy))
        for k in range(int(r)):
            picks[f"placebo_{k}"].append((pool_labels[int(rng.integers(len(pool_labels)))], dx, dy))

    out: dict[str, pd.DataFrame] = {}
    for name in control_names:
        cf = frames.copy()
        recs = picks[name]
        if recs:
            labels = [lbl for lbl, _, _ in recs]
            dxs = np.array([d for _, d, _ in recs], dtype=float)
            dys = np.array([d for _, _, d in recs], dtype=float)
            cf.loc[labels, "x"] = cf.loc[labels, "x"].to_numpy(dtype=float) + dxs
            cf.loc[labels, "y"] = cf.loc[labels, "y"].to_numpy(dtype=float) + dys
        out[name] = cf
    return out
