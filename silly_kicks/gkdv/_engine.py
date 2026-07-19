"""Ghost-substitution counterfactual engine (spec §4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ADR-019 mandates this seam repo-wide for every id comparison. It is a PUBLIC module
# (`silly_kicks.id_compat`, promoted in 4.53.0 from `tracking/_id_compat.py`) precisely
# because it is a repo-wide requirement rather than a tracking internal -- so importing it
# is not an exemption from gkdv's public-seams-only rule, it IS the rule.
#
# `home_team_id` is a caller-supplied scalar of uncontrolled dtype, and a raw `==`/`!=`
# against an id COLUMN is the single most damaging bug shape in this codebase (measured live
# at spadl/utils.py: an object-string `team_id` vs an int scalar makes `!=` True for EVERY
# row). The guard on that boundary is the ENUMERATED registry
# (`tests/invariants/conftest_id_scalar.py`'s `PUBLIC_ID_SCALAR_ENTRIES`, asserted by
# `tests/invariants/test_public_id_scalar_registry.py`), which calls each public entry point
# twice -- matched and mismatched-but-value-equal scalar -- and requires identical output.
# It replaced an AST lint that was incomplete by construction: safe and unsafe uses are the
# IDENTICAL AST, separable only by the scalar's provenance, which no syntactic rule can see.
from silly_kicks.id_compat import align_join_keys, canonical_id, ids_equal, same_id
from silly_kicks.spadl import config as spadlconfig

#: Attacking-third predicate: ball within this distance of the attacked goal (spec §4.1).
_DOMAIN_BALL_TO_GOAL_M = 35.0

_FIELD_LENGTH = spadlconfig.field_length  # 105.0
_GOAL_Y = spadlconfig.field_width / 2.0  # 34.0

#: Frame-key grain. One (game_id, period_id, frame_id) triple is ONE unit of drop accounting.
_FRAME_KEYS = ["game_id", "period_id", "frame_id"]

# Drop reasons (spec §4.1). Frames with a missing/NaN GK block are dropped-and-COUNTED,
# never scored as Delta = 0 -- a zero delta from a missing keeper reads as "no deterrence"
# and biases keeper aggregates toward the null.
_DROP_BALL_MISSING = "ball_row_missing"
_DROP_BALL_NOT_ALIVE = "ball_not_alive"
_DROP_BALL_COORDS = "ball_coordinates_missing"
_DROP_NO_POSSESSION = "no_possession"
_DROP_NO_DEFENDING_TEAM = "no_defending_team"
_DROP_NO_GOAL_MAP = "no_goal_map_entry"
_DROP_GOAL_MAP_DEGENERATE = "goal_map_degenerate"
_DROP_BALL_FAR = "ball_far_from_attacked_goal"
_DROP_NO_DEFENDING_GK = "no_defending_gk"
_DROP_STRIDE = "stride_skipped"
_DROP_NO_GHOST = "no_ghost_served"

#: The probe's ghost-target contract, DECLARED HERE rather than imported from
#: ``tracking/_model_eval.py``: gkdv consumes tracking's PUBLIC seams only, and
#: ``_TARGET_COLUMNS`` is private there. The two are pinned together by a cross-package
#: contract test (``tests/gkdv/test_provenance_to_targets.py``) that fails loudly if either
#: side drifts -- which is what a data contract between packages should look like.
_TARGET_COLUMNS = ("game_id", "period_id", "frame_id", "target_x", "target_y", "ghost_clamped", "ghost_out_of_box")

#: Provenance columns, in emission order. `drop_reason` is NaN on a SCORED row.
_PROVENANCE_COLUMNS = [
    "game_id",
    "period_id",
    "frame_id",
    "gk_team_id",
    "defending_team_id",
    "player_id",
    "is_goalkeeper_source",
    "ghost_gr_x",
    "ghost_gr_y",
    "ghost_x",
    "ghost_y",
    "displacement_m",
    "ghost_clamped",
    "ghost_out_of_box",
    "drop_reason",
]


@dataclass(frozen=True)
class GkdvParams:
    """Registered knobs for the GKDV v1 counterfactual. Frozen; echoed into GkdvReport.

    Examples
    --------
    >>> GkdvParams().possession_stride
    5
    """

    #: Sample every Nth eligible frame per possession (cost control, spec §5).
    possession_stride: int = 5
    #: Ball-to-attacked-goal distance bounding the domain, metres.
    domain_ball_to_goal_m: float = _DOMAIN_BALL_TO_GOAL_M
    #: Pitch-control method for the threat arm. HARD CONSTRAINT, not guidance: lambda_gk
    #: exists ONLY on SpearmanParams, so any other method silently produces a GK-BLIND arm.
    #: Validated fail-loud in __post_init__ -- the field is kept (rather than hard-wired) so
    #: the constraint is self-documenting and a future GK-aware method can join the allowlist.
    pitch_control_method: str = "spearman"
    #: GK control-rate multiplier. FORWARDED by the threat arm into
    #: ``SpearmanParams(lambda_gk=...)`` -- it is the ONLY mechanism by which the arm sees the
    #: keeper at all, which is what ``pitch_control_method``'s guard above is defending. The
    #: default equals ``SpearmanParams.lambda_gk``, so leaving it alone is byte-identical to
    #: passing no params; changing it genuinely changes the arm's keeper gain.
    lambda_gk: float = 3.0
    #: Ghost keeps the factual keeper's velocity (minimal-intervention counterfactual).
    ghost_keeps_actual_velocity: bool = True

    #: Methods that carry a GK term. Only these may score the threat arm.
    _GK_AWARE_METHODS = ("spearman",)

    def __post_init__(self) -> None:
        """Fail at CONSTRUCTION, not at call time, on a GK-blind pitch-control method."""
        if self.pitch_control_method not in self._GK_AWARE_METHODS:
            raise ValueError(
                f"pitch_control_method={self.pitch_control_method!r} is GK-BLIND: lambda_gk "
                f"exists only on SpearmanParams, so a ghost-GK substitution through it loses "
                f"the keeper's control-rate multiplier entirely and the threat arm would "
                f"measure nothing about the keeper. Allowed: {self._GK_AWARE_METHODS}."
            )


@dataclass(frozen=True)
class GkdvReport:
    """Run-level audit. Echoes the params actually used -- registration without
    traceability is not registration.

    Examples
    --------
    >>> GkdvReport(params=GkdvParams(), n_frames_in=0, n_frames_scored=0,
    ...            drop_reasons={}, n_clamped=0, n_out_of_box=0).n_frames_scored
    0
    """

    params: GkdvParams
    n_frames_in: int
    n_frames_scored: int
    drop_reasons: dict
    n_clamped: int
    n_out_of_box: int


#: Module-level default so callers get the frozen params without a call in the signature
#: default (ruff B008 forbids that, and its own guidance is a module-level singleton).
_DEFAULT_PARAMS = GkdvParams()


def _id_key(values) -> pd.Series:
    """Canonical id key as an object Series, mapped through the SCALAR ``canonical_id`` truth.

    PRIMARY ROLE: building **hashable composite keys** for set membership -- the
    ``(game_id, period_id, frame_id, team_id)`` tuples this module zips into sets. There is
    no ``ids_equal`` equivalent for that: element-wise comparison helpers answer "are these
    two columns equal", not "give me one canonical scalar per row that hashes consistently
    across dtypes". Two frames carrying the same id as ``2`` and ``"2"`` must land on the
    same tuple key, and only a per-element canonicalization delivers that.

    It originally ALSO guarded ``_same_team`` against an integral float stored in an object
    column -- the shape ``infer_ball_carrier`` then emitted for ``ball_carrier_team_id``,
    which canonicalized to ``"2.0"`` against the frames' ``"2"`` and made ``ids_equal`` False
    for EVERY row. That defect is now fixed at the source (``infer_ball_carrier`` restores
    its source dtype; ``canonical_id_series`` probes object CONTENT), so ``_same_team`` no
    longer needs the pre-mapping -- but the set-key uses are not redundant and remain.
    """
    return pd.Series(values).map(canonical_id).astype(object)


def _same_team(a, b) -> np.ndarray:
    """Dtype-safe element-wise team identity (ADR-019).

    Calls ``ids_equal`` directly. The former ``_id_key`` pre-mapping here was verified
    redundant rather than assumed: a differential test over every ordered pair of ten id
    representations (int64, Int64, float64, object-of-float, object-of-int, object-of-str,
    string, mixed, NA-bearing, genuine string ids) found ZERO divergence between the bare and
    wrapped forms once ``silly_kicks.id_compat`` was fixed. Tolerance of a caller-supplied boxed-object
    ``carrier=`` is pinned by
    ``test_object_boxed_carrier_team_id_still_resolves_the_defending_keeper``.
    """
    return ids_equal(pd.Series(a), pd.Series(b)).to_numpy()


def _pin_defended_goal(frames: pd.DataFrame) -> dict:
    """The ONE goal-map instance (spec §4.2), reused for the defended-goal flip AND for
    defending-keeper selection. Never re-derived on counterfactual frames -- goal-map drift
    across the two legs would contaminate the delta.

    Re-keyed on CANONICAL ids (ADR-019): the upstream map keys on raw groupby values, so a
    lookup with a differently-boxed-but-value-equal id would miss and present as a plausible
    pile of ``no_goal_map_entry`` drops rather than as the dtype bug it is.
    """
    from silly_kicks.tracking import defended_goal_x

    return {(canonical_id(g), canonical_id(p), canonical_id(t)): v for (g, p, t), v in defended_goal_x(frames).items()}


def _goal_lookup(goal_map: dict, game_id, period_id, team_id) -> float:
    """Goal-map lookup returning NaN (a counted drop) rather than raising on a miss."""
    key = (canonical_id(game_id), canonical_id(period_id), canonical_id(team_id))
    return float(goal_map.get(key, np.nan))


def _apply_domain(
    frames: pd.DataFrame,
    *,
    carrier: pd.DataFrame,
    goal_map: dict,
    params: GkdvParams,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the frame keys into ``(eligible, dropped)`` per the spec §4.1 domain.

    Returns one row per ``(game_id, period_id, frame_id)`` on BOTH sides, so the two
    partition the input exactly -- that conservation is what makes the drop accounting
    trustworthy. ``eligible`` carries the pinned per-frame context (possession team,
    defending team, both goal ends) so nothing downstream re-derives it.
    """
    players = frames[~frames["is_ball"].astype(bool)]
    ball = frames[frames["is_ball"].astype(bool)]

    keys = frames[_FRAME_KEYS].drop_duplicates().sort_values(_FRAME_KEYS).reset_index(drop=True)

    # --- ball facts -------------------------------------------------------------------
    ball_slim = ball.drop_duplicates(subset=_FRAME_KEYS, keep="first")[[*_FRAME_KEYS, "x", "y", "ball_state"]].rename(
        columns={"x": "ball_x", "y": "ball_y"}
    )
    work = keys.merge(ball_slim, on=_FRAME_KEYS, how="left")

    # --- possession (caller-suppliable => a cross-source join; ADR-019) -----------------
    carrier_slim = carrier[[*_FRAME_KEYS, "ball_carrier_team_id"]].drop_duplicates(subset=_FRAME_KEYS, keep="first")
    left, right = align_join_keys(work, carrier_slim, list(_FRAME_KEYS))
    work = left.merge(right, on=_FRAME_KEYS, how="left").rename(columns={"ball_carrier_team_id": "possession_team_id"})

    # --- the defending team = the OTHER team on the pitch this (game, period) ----------
    team_keys = players[["game_id", "period_id", "team_id"]].dropna(subset=["team_id"]).drop_duplicates()
    cand = work[[*_FRAME_KEYS, "possession_team_id"]].merge(team_keys, on=["game_id", "period_id"], how="left")
    # ids_equal is column-vs-column and dtype-safe; a raw != would mark every row a
    # candidate under an Int64-vs-object mismatch and silently pick the WRONG keeper.
    cand = cand[~_same_team(cand["team_id"], cand["possession_team_id"])]
    if len(cand):
        n_cand = cand.groupby(_FRAME_KEYS, dropna=False).size().rename("n_defending")
        defending = (
            cand.drop_duplicates(subset=_FRAME_KEYS, keep="first")
            .set_index(_FRAME_KEYS)["team_id"]
            .rename("defending_team_id")
        )
        work = work.join(n_cand, on=_FRAME_KEYS).join(defending, on=_FRAME_KEYS)
        work["n_defending"] = work["n_defending"].fillna(0).astype(int)
    else:
        work["n_defending"] = 0
        work["defending_team_id"] = pd.NA

    # --- pinned goal ends ---------------------------------------------------------------
    # The ATTACKED goal is the goal the DEFENDING team defends -- NOT `goal_map[possession]`,
    # which is the goal the attackers defend at the OTHER end. Conflating the two puts every
    # in-domain frame ~85 m from its "attacked" goal and silently empties the domain.
    work["defended_goal_x"] = [
        _goal_lookup(goal_map, g, p, t)
        for g, p, t in zip(work["game_id"], work["period_id"], work["defending_team_id"], strict=True)
    ]
    work["possession_defends_x"] = [
        _goal_lookup(goal_map, g, p, t)
        for g, p, t in zip(work["game_id"], work["period_id"], work["possession_team_id"], strict=True)
    ]
    work["attacked_goal_x"] = work["defended_goal_x"]

    # --- defending GK presence (finite coordinates) -------------------------------------
    gk_rows = players[players["is_goalkeeper"].astype(bool)]
    gk_ok = gk_rows[gk_rows["x"].notna() & gk_rows["y"].notna()]
    # Canonical keys on BOTH sides (ADR-019) -- a raw tuple-set membership test is a raw
    # `==` in disguise and would silently report "no defending GK" on a dtype mismatch.
    gk_present = set(
        zip(
            _id_key(gk_ok["game_id"]),
            _id_key(gk_ok["period_id"]),
            _id_key(gk_ok["frame_id"]),
            _id_key(gk_ok["team_id"]),
            strict=True,
        )
    )
    work["has_defending_gk"] = [
        (g, p, f, t) in gk_present
        for g, p, f, t in zip(
            _id_key(work["game_id"]),
            _id_key(work["period_id"]),
            _id_key(work["frame_id"]),
            _id_key(work["defending_team_id"]),
            strict=True,
        )
    ]

    # --- first-failing-reason cascade ---------------------------------------------------
    ball_dist = np.hypot(
        work["ball_x"].to_numpy(dtype=float) - work["attacked_goal_x"].to_numpy(dtype=float),
        work["ball_y"].to_numpy(dtype=float) - _GOAL_Y,
    )
    reason = pd.Series(pd.NA, index=work.index, dtype="object")

    def _mark(mask: np.ndarray, why: str) -> None:
        reason.loc[reason.isna() & pd.Series(mask, index=work.index)] = why

    _mark(work["ball_state"].isna().to_numpy(), _DROP_BALL_MISSING)
    _mark((work["ball_state"] != "alive").to_numpy(), _DROP_BALL_NOT_ALIVE)
    _mark((work["ball_x"].isna() | work["ball_y"].isna()).to_numpy(), _DROP_BALL_COORDS)
    _mark(work["possession_team_id"].isna().to_numpy(), _DROP_NO_POSSESSION)
    _mark((work["n_defending"] != 1).to_numpy(), _DROP_NO_DEFENDING_TEAM)
    _mark(
        (work["defended_goal_x"].isna() | work["possession_defends_x"].isna()).to_numpy(),
        _DROP_NO_GOAL_MAP,
    )
    # Both teams mapped to the SAME end: an unusable goal map for this frame (upstream
    # is_goalkeeper mis-flag). Dropped-and-counted rather than silently scored at one end.
    _mark(
        (work["defended_goal_x"] == work["possession_defends_x"]).to_numpy(dtype=bool),
        _DROP_GOAL_MAP_DEGENERATE,
    )
    _mark(~np.isfinite(ball_dist) | (ball_dist > float(params.domain_ball_to_goal_m)), _DROP_BALL_FAR)
    _mark(~work["has_defending_gk"].to_numpy(dtype=bool), _DROP_NO_DEFENDING_GK)

    # NOTE on `_DROP_NO_GOAL_MAP`: it is a guard, not a reachable branch today. Both team
    # ids fed to `_goal_lookup` are drawn from the same `players` population the goal map is
    # built from, and `defended_goal_x` never returns NaN, so the lookup cannot currently
    # miss. It is retained (cheap) against a future goal-map source, but deliberately WITHOUT
    # an all-miss raise: an untestable raise is not a guard, it is unexercised code.

    # --- stride (cost control, spec §5) -------------------------------------------------
    # Counted as a drop reason, NOT silently discarded, so the report still conserves.
    # v1 does not segment possessions: the stride runs over the ordered eligible frames
    # within each (game_id, period_id). Documented limitation, not a hidden approximation.
    stride = int(params.possession_stride)
    if stride > 1:
        surviving = work.index[reason.isna()]
        rank = work.loc[surviving].groupby(["game_id", "period_id"], dropna=False).cumcount()
        reason.loc[surviving[(rank % stride != 0).to_numpy()]] = _DROP_STRIDE

    work["drop_reason"] = reason
    eligible = work[work["drop_reason"].isna()].drop(columns=["drop_reason"]).reset_index(drop=True)
    dropped = work[work["drop_reason"].notna()][[*_FRAME_KEYS, "defending_team_id", "drop_reason"]]
    return eligible, dropped.reset_index(drop=True)


def _build_provenance(
    frames: pd.DataFrame,
    *,
    served: pd.DataFrame,
    eligible: pd.DataFrame,
    dropped: pd.DataFrame,
    goal_map: dict,
) -> pd.DataFrame:
    """Per-(frame, gk_team) provenance for eligible frames + one row per dropped frame.

    Both teams' keepers are kept (that is what the serving seam returns, and ``_metric``
    consumes the pair); selecting the DEFENDING one is :func:`provenance_to_targets`'s job.
    """
    players = frames[~frames["is_ball"].astype(bool)]
    gk_rows = players[players["is_goalkeeper"].astype(bool)].drop_duplicates(
        subset=[*_FRAME_KEYS, "team_id"], keep="first"
    )
    actual_cols = [*_FRAME_KEYS, "team_id", "x", "y", "player_id"]
    if "is_goalkeeper_source" in gk_rows.columns:
        actual_cols.append("is_goalkeeper_source")
    actual = gk_rows[actual_cols].rename(columns={"team_id": "gk_team_id", "x": "actual_x", "y": "actual_y"})

    ctx = eligible[[*_FRAME_KEYS, "defending_team_id"]]
    left, right = align_join_keys(served, ctx, list(_FRAME_KEYS))
    scored = left.merge(right, on=_FRAME_KEYS, how="inner")
    left, right = align_join_keys(scored, actual, [*_FRAME_KEYS, "gk_team_id"])
    scored = left.merge(right, on=[*_FRAME_KEYS, "gk_team_id"], how="left")

    if len(scored):
        defended = np.array(
            [
                _goal_lookup(goal_map, g, p, t)
                for g, p, t in zip(scored["game_id"], scored["period_id"], scored["gk_team_id"], strict=True)
            ],
            dtype=float,
        )
        gr_x = scored["ghost_gr_x"].to_numpy(dtype=float)
        scored["ghost_x"] = np.where(defended == 0.0, gr_x, _FIELD_LENGTH - gr_x)
        scored["ghost_y"] = scored["ghost_gr_y"].to_numpy(dtype=float)
        scored["displacement_m"] = np.hypot(
            scored["ghost_x"].to_numpy(dtype=float) - scored["actual_x"].to_numpy(dtype=float),
            scored["ghost_y"].to_numpy(dtype=float) - scored["actual_y"].to_numpy(dtype=float),
        )
    else:
        for col in ("ghost_x", "ghost_y", "displacement_m"):
            scored[col] = pd.Series(dtype=float)
    scored["drop_reason"] = pd.Series(pd.NA, index=scored.index, dtype="object")

    # Eligible frames the serving seam returned nothing for are DROPS, not silent absences.
    served_keys = set(
        zip(
            _id_key(scored["game_id"]),
            _id_key(scored["period_id"]),
            _id_key(scored["frame_id"]),
            strict=True,
        )
    )
    unserved_mask = np.array(
        [
            (g, p, f) not in served_keys
            for g, p, f in zip(
                _id_key(eligible["game_id"]),
                _id_key(eligible["period_id"]),
                _id_key(eligible["frame_id"]),
                strict=True,
            )
        ],
        dtype=bool,
    )
    unserved = eligible.loc[unserved_mask, [*_FRAME_KEYS, "defending_team_id"]].copy()
    unserved["drop_reason"] = _DROP_NO_GHOST

    drop_rows = pd.concat([dropped, unserved], ignore_index=True) if len(unserved) else dropped
    out = pd.concat([scored, drop_rows], ignore_index=True)
    for col in _PROVENANCE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    # bool(NaN) is True, so a null flag would silently widen the probe's trusted stratum.
    # Nullable "boolean" keeps DROPPED rows honestly NA while scored rows stay real booleans.
    for col in ("ghost_clamped", "ghost_out_of_box"):
        out[col] = out[col].astype("boolean")
    return out[_PROVENANCE_COLUMNS].sort_values([*_FRAME_KEYS, "gk_team_id"], na_position="last").reset_index(drop=True)


def _write_back(frames: pd.DataFrame, *, provenance: pd.DataFrame, params: GkdvParams) -> pd.DataFrame:
    """Substitute the ghost for the DEFENDING keeper. PURE -- returns a new frame."""
    out = frames.copy()
    scored = provenance[provenance["drop_reason"].isna()]
    defending = scored[_same_team(scored["gk_team_id"], scored["defending_team_id"])]
    if not len(defending):
        return out

    gk_mask = (out["is_goalkeeper"].astype(bool) & ~out["is_ball"].astype(bool)).to_numpy()
    gk_side = out.loc[gk_mask, [*_FRAME_KEYS, "team_id"]].rename(columns={"team_id": "gk_team_id"})
    ghost_side = defending[[*_FRAME_KEYS, "gk_team_id", "ghost_x", "ghost_y"]]
    # ADR-019: the join keys are canonicalized on BOTH sides before the merge. A raw merge
    # here would either raise on a numeric-vs-object key or silently match nothing, which
    # would return the FACTUAL frames labelled as counterfactual -- a silent Delta == 0.
    left, right = align_join_keys(gk_side, ghost_side, [*_FRAME_KEYS, "gk_team_id"])
    joined = left.merge(right, on=[*_FRAME_KEYS, "gk_team_id"], how="left")
    joined.index = gk_side.index

    hit_mask = joined["ghost_x"].notna().to_numpy() & joined["ghost_y"].notna().to_numpy()
    hits = joined.index[hit_mask]
    if len(hits):
        out.loc[hits, "x"] = joined.loc[hits, "ghost_x"].to_numpy(dtype=float)
        out.loc[hits, "y"] = joined.loc[hits, "ghost_y"].to_numpy(dtype=float)
        if not params.ghost_keeps_actual_velocity:
            # Registered sensitivity variant (spec §4.5): a teleported-but-still-moving
            # ghost projects the ACTUAL keeper's momentum from the ghost position.
            for col in ("vx", "vy", "speed"):
                if col in out.columns:
                    out.loc[hits, col] = 0.0
    return out


def build_ghost_frames(
    frames: pd.DataFrame,
    *,
    model=None,
    home_team_id: int | str,
    carrier: pd.DataFrame | None = None,
    params: GkdvParams = _DEFAULT_PARAMS,
) -> tuple[pd.DataFrame, pd.DataFrame, GkdvReport]:
    """Build the ghost-keeper counterfactual frames plus per-frame provenance.

    PURE: never mutates ``frames``.

    Domain (spec §4.1): alive ball, in-possession team attacking, ball within
    ``params.domain_ball_to_goal_m`` of the attacked goal, defending-GK row present.
    Frames with a missing/NaN GK block are **dropped-and-counted, never scored as
    Delta = 0** -- a zero delta from a missing keeper reads as "no deterrence" and biases
    keeper aggregates toward the null.

    ``counterfactual_frames`` is the FULL input with the defending keeper substituted, so
    consumers MUST restrict to the scored frames (``provenance["drop_reason"].isna()``, or
    :func:`provenance_to_targets`) rather than differencing every frame -- a dropped frame
    is byte-identical across the two legs and would contribute exactly the Delta = 0 this
    domain exists to exclude.

    The write-back rule above is prose; the gate that enforces it is
    ``tests/gkdv/test_engine.py::test_writeback_moves_the_DEFENDING_keeper_and_ONLY_that_keeper``.

    ``provenance`` keys per ``(game_id, period_id, frame_id, gk_team_id)`` and carries BOTH
    teams' served keepers on a scored frame, plus exactly ONE row per dropped frame.
    ``report`` counts FRAMES (not provenance rows), so
    ``n_frames_scored + sum(drop_reasons.values()) == n_frames_in`` holds exactly.

    Parameters
    ----------
    carrier : pd.DataFrame, optional
        Precomputed :func:`~silly_kicks.tracking.infer_ball_carrier` output. Pinned ONCE
        and shared by the domain filter and the serving seam (spec §4.2). When omitted it
        is inferred here using the model's recorded ``carrier_params`` when ``model`` is a
        ``GhostGkModel`` instance; pass ``model`` as an instance (or supply ``carrier``) if
        those params matter, since a variant NAME cannot be resolved without reaching into
        tracking privates.

    Returns
    -------
    (counterfactual_frames, provenance, report)

    Examples
    --------
    The three returns are used together: the counterfactual frames feed the arms, the
    provenance says WHICH frames may be differenced, and the report is the audit trail::

        from silly_kicks.gkdv import build_ghost_frames

        ghost_frames, provenance, report = build_ghost_frames(frames, home_team_id=1)

        scored = provenance.loc[provenance["drop_reason"].isna()]
        report.n_frames_scored           # frames that passed the §4.1 domain
        report.drop_reasons              # {reason: count} for everything excluded

    ``ghost_frames`` is the FULL input with the defending keeper moved, NOT just the
    scored subset -- so differencing it against ``frames`` wholesale silently averages in
    a ``0.0`` for every out-of-domain frame. Restrict to ``scored`` (or use
    :func:`provenance_to_targets`) first.

    The frame counts reconcile exactly, which is the cheapest way to confirm nothing was
    quietly discarded::

        assert report.n_frames_scored + sum(report.drop_reasons.values()) == report.n_frames_in

    Pass ``model`` as a ``GhostGkModel`` INSTANCE (or supply ``carrier`` yourself) when the
    model's recorded ``carrier_params`` matter: a variant NAME cannot be resolved to those
    params without reaching into tracking privates, so the defaults would be used instead.
    """
    from silly_kicks.tracking import infer_ball_carrier, serve_ghost_gk_positions

    src = frames  # never mutated

    if carrier is None:
        carrier_params = getattr(model, "carrier_params", None)
        carrier = infer_ball_carrier(src, **carrier_params) if carrier_params else infer_ball_carrier(src)

    goal_map = _pin_defended_goal(src)  # ONE instance, spec §4.2
    eligible, dropped = _apply_domain(src, carrier=carrier, goal_map=goal_map, params=params)

    served = serve_ghost_gk_positions(
        src,
        model=model,
        home_team_id=home_team_id,
        carrier=carrier[[*_FRAME_KEYS, "ball_carrier_team_id"]],
        link_frame_ids=set(eligible["frame_id"].unique()),
    )
    prov = _build_provenance(src, served=served, eligible=eligible, dropped=dropped, goal_map=goal_map)
    cf = _write_back(src, provenance=prov, params=params)

    scored_rows = prov[prov["drop_reason"].isna()]
    drop_reasons = {str(k): int(v) for k, v in prov["drop_reason"].dropna().value_counts().to_dict().items()}
    report = GkdvReport(
        params=params,
        n_frames_in=len(src[_FRAME_KEYS].drop_duplicates()),
        n_frames_scored=len(scored_rows[_FRAME_KEYS].drop_duplicates()),
        drop_reasons=drop_reasons,
        n_clamped=int(scored_rows["ghost_clamped"].fillna(False).astype(bool).sum()),
        n_out_of_box=int(scored_rows["ghost_out_of_box"].fillna(False).astype(bool).sum()),
    )
    if len(scored_rows) and not np.isfinite(scored_rows[["ghost_x", "ghost_y"]].to_numpy(dtype=float)).all():
        raise ValueError(
            "build_ghost_frames produced a non-finite ghost coordinate on a SCORED frame. "
            "Pitch control silently DROPS NaN-coordinate rows (_spearman.py dropna), so a "
            "NaN ghost would make the keeper vanish rather than error."
        )
    return cf, prov, report


def _select_defending_keeper(
    scored: pd.DataFrame,
    *,
    frames: pd.DataFrame,
    home_team_id: int | str,
) -> pd.DataFrame:
    """Keep the DEFENDING team's keeper per frame, using the §4.2 pinned goal map.

    ``home_team_id`` is cross-checked against the goal map: it is the same caller-supplied
    scalar :func:`build_ghost_frames` threads into the ghost feature extractor, so a
    mismatch (wrong match, or an id-dtype mismatch) means those features were built against
    a bogus home team. Failing here is the cheapest place to learn that.
    """
    goal_map = _pin_defended_goal(frames)
    mapped_teams = [key[2] for key in goal_map]
    if not any(same_id(t, home_team_id) for t in mapped_teams):
        raise ValueError(
            f"provenance_to_targets: home_team_id={home_team_id!r} matches no team in these "
            f"frames (teams present: {sorted({str(t) for t in mapped_teams})}). Either the "
            "frames are from a different match, or the id dtypes disagree (ADR-019)."
        )
    if "defending_team_id" not in scored.columns:
        raise ValueError(
            "provenance_to_targets: provenance is missing 'defending_team_id'. The defending "
            "keeper is PINNED by the engine (spec §4.2) and must not be re-derived here."
        )
    return scored[_same_team(scored["gk_team_id"], scored["defending_team_id"])]


def provenance_to_targets(
    provenance: pd.DataFrame,
    *,
    frames: pd.DataFrame,
    home_team_id: int | str,
) -> pd.DataFrame:
    """Project the engine's provenance frame onto the probe's ``_TARGET_COLUMNS`` contract.

    The provenance frame and the targets frame are DIFFERENT views and this adapter is the
    only supported bridge (spec §4.6):

    * renames ``ghost_x``/``ghost_y`` -> ``target_x``/``target_y``;
    * DROPS dropped frames (the probe requires finite coordinates on every row);
    * selects the **defending-team** keeper via the pinned goal map, so the result carries
      exactly one row per ``(game_id, period_id, frame_id)`` -- the serving seam writes
      BOTH teams' keepers and a naive pass-through would either trip the probe's uniqueness
      check or silently select the wrong keeper;
    * guarantees both flags are non-null (``bool(NaN)`` is ``True``, which would silently
      shrink the probe's trusted stratum).

    Examples
    --------
    The supported bridge from the engine to the GK-substitution probe::

        from silly_kicks.gkdv import build_ghost_frames, provenance_to_targets

        _, provenance, _ = build_ghost_frames(frames, home_team_id=1)
        targets = provenance_to_targets(provenance, frames=frames, home_team_id=1)

        # exactly one row per scored frame, with finite coordinates on every row:
        assert not targets.duplicated(["game_id", "period_id", "frame_id"]).any()
        assert targets[["target_x", "target_y"]].notna().all().all()

    Passing ``provenance`` to the probe directly does NOT work and does not fail cleanly.
    The serving seam writes BOTH teams' keepers per scored frame, so a naive pass-through
    either trips the probe's uniqueness check or -- worse -- selects the ATTACKING
    keeper, silently measuring the wrong player. This adapter applies the pinned goal map
    to keep the defending one, drops the dropped frames, and forces both flags non-null
    (``bool(NaN)`` is ``True``, which would quietly inflate the probe's trusted stratum).
    """
    scored = provenance[provenance["drop_reason"].isna()].copy()
    defending = _select_defending_keeper(scored, frames=frames, home_team_id=home_team_id)
    out = defending.rename(columns={"ghost_x": "target_x", "ghost_y": "target_y"})
    out["ghost_clamped"] = out["ghost_clamped"].fillna(False).astype(bool)
    out["ghost_out_of_box"] = out["ghost_out_of_box"].fillna(False).astype(bool)
    out = out[list(_TARGET_COLUMNS)]
    if out.duplicated(subset=list(_FRAME_KEYS)).any():
        raise ValueError(
            "provenance_to_targets produced >1 row per (game_id, period_id, frame_id) -- "
            "the defending-keeper selection failed. Do NOT pass both teams' keepers."
        )
    return out.reset_index(drop=True)
