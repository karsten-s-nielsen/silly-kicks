"""TF-19 physics-arm instrument-validity (Layer 0) + responsiveness (Layer 1) probes.

Reported-not-gated (spec ``docs/superpowers/specs/2026-08-28-tf19-a2-...``). Physics arms only
(``delta_das`` / ``delta_threat_suppression``). Depends on the shipped gkdv engine + arms; obeys
ADR-037 (gkdv reaches ``tracking`` only through ``_das_port``; the engine's own public
``resolve_defended_goals`` path is reused via ``_engine`` helpers, not a new tracking import here).

The dose imposer substitutes ONLY the defending keeper at an imposed position, reusing the engine's
domain/provenance so the scored set + defending keeper match ``build_ghost_frames`` exactly. It does
NOT route through ``provenance_to_targets``: that adapter's 7-col ``_TARGET_COLUMNS`` renames
``ghost_x/y`` -> ``target_x/y`` and drops ``actual_*``/``defended_goal_x`` (plan P1).

Layer-0 discrimination proof (why the void ``and`` is load-bearing, not vacuous). The void condition
is ``not passes_multiple and not passes_placebo``; both-pass (valid) and both-fail (void) alone
CANNOT separate that ``and`` from an ``or`` -- in each the two legs agree, so ``and`` and ``or``
return the same verdict. The ``either-leg`` case (saturating clears the placebo band but not the 5x
realistic multiple -> valid) is the one that discriminates: flipping the void ``and`` to ``or`` (i.e.
requiring BOTH legs for validity) leaves the both-pass/both-fail tests unchanged but flips the
``either-leg`` test to ``instrument_void``. It is committed as
``tests/gkdv/test_probe_layer0.py::test_either_leg_suffices_placebo_only_is_valid`` rather than left
as a manual mutation, so the load-bearing boolean is guarded, not merely asserted in prose.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from silly_kicks.id_compat import align_join_keys
from silly_kicks.spadl import config as _spadlconfig

from ._engine import GkdvParams, _goal_lookup, _pin_defended_goal, _same_team, build_ghost_frames

_DEFAULT_PARAMS = GkdvParams()
_FIELD_LENGTH = _spadlconfig.field_length  # 105.0
_GOAL_Y = _spadlconfig.field_width / 2.0  # 34.0
_FRAME_KEYS = ["game_id", "period_id", "frame_id"]

#: The discrete Layer-0 doses. `ladder` additionally takes a `displacement` (metres).
Dose = Literal["realistic", "ladder", "saturating_goalline", "saturating_x30"]

LADDER_M: tuple[float, float, float] = (2.0, 3.0, 4.0)  # imposed displacements (mirrors XS_PROBE_DOSE_LADDER)
REALISTIC_MIN_DISP_M: float = 2.0  # the |ghost - actual| floor (parent spec: the ghost's ~1.1 m MAE)
SATURATING_X30_GR: float = 30.0  # goal-relative x of the second saturating position


def _build_dose_targets(frames: pd.DataFrame, provenance: pd.DataFrame) -> pd.DataFrame:
    """Per-scored-DEFENDING-frame table for the dose imposer + paired-vector controls.

    Sourced correctly (plan P1): ``ghost_x/ghost_y/displacement_m/defending_team_id`` from the
    ``provenance`` frame (``_PROVENANCE_COLUMNS``), ``actual_x/actual_y`` from ``frames``' defending-GK
    rows, ``defended_goal_x`` (0.0/105.0) from the engine's pinned goal map. Carries every column the
    imposer AND :func:`paired_vector_controls` read.

    Examples
    --------
    Built internally by :func:`impose_defending_keeper_dose`; see its Examples.
    """
    scored = provenance[provenance["drop_reason"].isna()]
    # defending keeper rows only (gk_team_id == defending_team_id), ADR-019-safe via the engine helper.
    defending = scored[_same_team(scored["gk_team_id"], scored["defending_team_id"])]
    defending = defending[[*_FRAME_KEYS, "defending_team_id", "ghost_x", "ghost_y", "displacement_m"]]

    players = frames[~frames["is_ball"].astype(bool)]
    # Deduplicate keeper rows to ONE per (frame, team) -- MIRRORS the engine's provenance builder
    # (`_engine._build_provenance` `gk_rows.drop_duplicates(subset=[*_FRAME_KEYS, "team_id"], keep="first")`),
    # and it is load-bearing: a substitution window puts BOTH keepers on the pitch for a defending team
    # in the same frame (real GS matches carry 22/23/24-player frames), so without the dedup this LEFT
    # merge fans out and `_substitute_defending_keeper`'s `joined.index = gk_side.index` raises a length
    # mismatch. `keep="first"` matches the engine so the imposed keeper is the SAME row the engine ghosted.
    gk = players[players["is_goalkeeper"].astype(bool)].drop_duplicates(subset=[*_FRAME_KEYS, "team_id"], keep="first")[
        [*_FRAME_KEYS, "team_id", "x", "y"]
    ]
    gk = gk.rename(columns={"team_id": "defending_team_id", "x": "actual_x", "y": "actual_y"})
    left, right = align_join_keys(defending, gk, [*_FRAME_KEYS, "defending_team_id"])
    targets = left.merge(right, on=[*_FRAME_KEYS, "defending_team_id"], how="left")

    goal_map = _pin_defended_goal(frames)
    targets["defended_goal_x"] = [
        _goal_lookup(goal_map, g, p, tm)
        for g, p, tm in zip(targets["game_id"], targets["period_id"], targets["defending_team_id"], strict=True)
    ]
    return targets.reset_index(drop=True)


def _substitute_defending_keeper(
    frames: pd.DataFrame,
    targets: pd.DataFrame,
    imp_x: np.ndarray,
    imp_y: np.ndarray,
    *,
    params: GkdvParams,
) -> pd.DataFrame:
    """Mirror of ``_engine._write_back`` but for an IMPOSED (dose) position. PURE (new frame)."""
    out = frames.copy()
    move = targets[[*_FRAME_KEYS, "defending_team_id"]].copy()
    move["imp_x"] = imp_x
    move["imp_y"] = imp_y
    gk_mask = (out["is_goalkeeper"].astype(bool) & ~out["is_ball"].astype(bool)).to_numpy()
    gk_side = out.loc[gk_mask, [*_FRAME_KEYS, "team_id"]].rename(columns={"team_id": "defending_team_id"})
    left, right = align_join_keys(gk_side, move, [*_FRAME_KEYS, "defending_team_id"])
    joined = left.merge(right, on=[*_FRAME_KEYS, "defending_team_id"], how="left")
    joined.index = gk_side.index
    hit = joined["imp_x"].notna().to_numpy() & joined["imp_y"].notna().to_numpy()
    idx = joined.index[hit]
    if len(idx):
        out.loc[idx, "x"] = joined.loc[idx, "imp_x"].to_numpy(dtype=float)
        out.loc[idx, "y"] = joined.loc[idx, "imp_y"].to_numpy(dtype=float)
        if not params.ghost_keeps_actual_velocity:
            for col in ("vx", "vy", "speed"):
                if col in out.columns:
                    out.loc[idx, col] = 0.0
    return out


def impose_defending_keeper_dose(
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    dose: Dose,
    displacement: float | None = None,
    model=None,
    params: GkdvParams = _DEFAULT_PARAMS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Substitute ONLY the defending keeper at the ``dose`` position. PURE (new frame).

    Returns ``(imposed_frames, targets)``. ``imposed_frames`` is a NEW frame with only the defending
    keeper moved to the dose position; ``targets`` carries the per-scored-frame contract
    (``_FRAME_KEYS``, ``defending_team_id``, ``actual_x/actual_y``, ``defended_goal_x``,
    ``ghost_x/ghost_y``, ``displacement_m``) with ``imp_x/imp_y`` (the applied dose) attached.

    ``model`` is threaded to :func:`build_ghost_frames` (which produces the domain + provenance the
    imposer reuses); the ``realistic`` dose additionally uses the model's own ghost position filtered
    to ``|displacement| >= REALISTIC_MIN_DISP_M``. Saturating/ladder ignore the model's position.

    Examples
    --------
    Impose the saturating goal-line dose on the defending keeper::

        imposed, targets = impose_defending_keeper_dose(
            frames, home_team_id=1, dose="saturating_goalline", model=ghost_model
        )
    """
    _, provenance, _ = build_ghost_frames(frames, model=model, home_team_id=home_team_id, params=params)
    targets = _build_dose_targets(frames, provenance)
    if not len(targets):
        return frames.copy(), targets

    defended = targets["defended_goal_x"].to_numpy(dtype=float)
    actual_x = targets["actual_x"].to_numpy(dtype=float)
    actual_y = targets["actual_y"].to_numpy(dtype=float)

    if dose == "saturating_goalline":
        imp_x = defended
        imp_y = np.full(len(targets), _GOAL_Y)
    elif dose == "saturating_x30":
        imp_x = np.where(defended == 0.0, SATURATING_X30_GR, _FIELD_LENGTH - SATURATING_X30_GR)
        imp_y = actual_y
    elif dose == "ladder":
        if displacement is None:
            msg = "dose='ladder' requires displacement="
            raise ValueError(msg)
        sign = np.where(defended == 0.0, -1.0, 1.0)  # toward the defended goal
        imp_x = actual_x + sign * float(displacement)
        imp_y = actual_y
    elif dose == "realistic":
        imp_x = targets["ghost_x"].to_numpy(dtype=float)
        imp_y = targets["ghost_y"].to_numpy(dtype=float)
        keep = targets["displacement_m"].to_numpy(dtype=float) >= REALISTIC_MIN_DISP_M
        targets = targets.loc[keep].reset_index(drop=True)
        imp_x, imp_y = imp_x[keep], imp_y[keep]
    else:
        msg = f"unknown dose: {dose!r}"
        raise ValueError(msg)

    imposed = _substitute_defending_keeper(frames, targets, imp_x, imp_y, params=params)
    targets = targets.copy()
    targets["imp_x"] = imp_x
    targets["imp_y"] = imp_y
    return imposed, targets


# --- Task 2: Layer-0 instrument validity -----------------------------------------------------------

#: A live instrument moves the saturating dose >= this multiple of the realistic dose.
SATURATING_MULTIPLE: float = 5.0
#: Pooled-domain floor below which an arm is `arm_unscoreable` (not "broken"). Registered here; its
#: corpus-measured basis is recorded in the driver artifact (never silently thinned into a verdict).
MIN_DOMAIN_FRAMES: int = 200
LAYER0_VERDICTS: tuple[str, str, str] = ("instrument_valid", "instrument_void", "arm_unscoreable")


def layer0_instrument_verdict(*, realistic_abs, saturating_abs, placebo_p95, n_domain) -> str:
    """Layer-0 instrument-validity verdict over ALREADY-POOLED corpus statistics (never per shard).

    ``arm_unscoreable`` fires FIRST and is a first-class verdict DISTINCT from ``instrument_void``:
    a domain below :data:`MIN_DOMAIN_FRAMES` (insufficient support) or any all-NaN/empty leg (the
    provider cannot score this arm, e.g. velocity-less ΔDAS) must never read as "the instrument is
    broken". Otherwise the parent void condition applies: void iff the saturating median clears
    NEITHER ``SATURATING_MULTIPLE x`` the realistic median NOR the placebo 95th percentile; valid if
    either leg clears (see the module docstring's discrimination note). The multiple leg additionally
    requires ``real_med > 0`` -- a zero realistic baseline (plausible on the zero-dominated Delta-DAS
    arm) would make ``sat_med >= 5 * 0`` trivially true and vacuously validate a dead instrument, so
    when there is no realistic baseline the placebo leg is the sole backstop.

    Parameters
    ----------
    realistic_abs, saturating_abs
        Pooled per-frame ``|arm|`` values under the realistic and saturating doses.
    placebo_p95
        Pooled placebo-band 95th percentile (the paired-vector control magnitude).
    n_domain
        Pooled scored-frame count for this arm.

    Examples
    --------
    >>> import numpy as np
    >>> layer0_instrument_verdict(
    ...     realistic_abs=np.full(300, 0.05), saturating_abs=np.full(300, 0.5),
    ...     placebo_p95=0.02, n_domain=300,
    ... )
    'instrument_valid'
    >>> layer0_instrument_verdict(
    ...     realistic_abs=np.full(300, 0.05), saturating_abs=np.full(300, 0.04),
    ...     placebo_p95=0.10, n_domain=300,
    ... )
    'instrument_void'
    >>> layer0_instrument_verdict(
    ...     realistic_abs=np.full(3, 0.05), saturating_abs=np.full(3, 0.04),
    ...     placebo_p95=0.10, n_domain=3,
    ... )
    'arm_unscoreable'
    """
    real = np.asarray(realistic_abs, dtype=float)
    sat = np.asarray(saturating_abs, dtype=float)
    # (a) insufficient support; (b) provider can't score this arm -> arm_unscoreable, NOT void.
    if n_domain < MIN_DOMAIN_FRAMES:
        return "arm_unscoreable"
    real_med = float(np.nanmedian(real)) if real.size and bool(np.isfinite(real).any()) else float("nan")
    sat_med = float(np.nanmedian(sat)) if sat.size and bool(np.isfinite(sat).any()) else float("nan")
    if not (np.isfinite(real_med) and np.isfinite(sat_med) and np.isfinite(float(placebo_p95))):
        return "arm_unscoreable"
    # The `>= 5x real_med` test is VACUOUS when real_med == 0: `sat_med >= 0` is trivially true, so a
    # DEAD instrument would validate. Delta-DAS is zero-dominated, so real_med == 0 is plausible on the
    # very arm this pass scores. Require real_med > 0 for the multiple leg; when there is no realistic
    # baseline, validity rests on the placebo leg (the real backstop) -- never on `5 * 0`.
    passes_multiple = real_med > 0.0 and sat_med >= SATURATING_MULTIPLE * real_med
    passes_placebo = sat_med > float(placebo_p95)
    # Parent VOID condition: void iff NOT multiple AND NOT placebo; valid if either leg clears.
    return "instrument_void" if (not passes_multiple and not passes_placebo) else "instrument_valid"


# --- Task 3: Layer-1 responsiveness + paired-vector controls ---------------------------------------

#: Layer-1 responsiveness ratio. A NEW registration -- NOT `TF19_PROBE_RATIO` (the xCross threshold)
#: nor `XS_PROBE_RATIO` (both model-specific). The parent Layer-1 idiom fixes the FORM, not the value.
PHYSICS_ARM_PROBE_RATIO: float = 2.0
#: Paired-vector control count: the nearest defending-team outfielder + R random ones.
R: int = 3
#: Pinned regimes so `_measure_match` + the controls are unambiguous: Regime O = the observed ghost
#: (the shipped metric); Regime I = the imposed, discriminating 2 m ladder dose.
REGIME_O_DOSE: str = "realistic"
REGIME_I_DOSE: str = "ladder"
REGIME_I_LADDER_M: float = 2.0
LAYER1_VERDICTS: tuple[str, str, str] = ("responsive", "not_responsive", "arm_unscoreable")


def layer1_responsiveness_verdict(*, gk_med, nd_med, placebo_p95, n_domain) -> str:
    """Layer-1 responsiveness verdict over ALREADY-POOLED corpus statistics (never per shard).

    The shipped idiom ``gk_med >= PHYSICS_ARM_PROBE_RATIO * max(nd_med, placebo_p95)`` with the same
    ``arm_unscoreable`` short-circuits as :func:`layer0_instrument_verdict` (thin domain OR any NaN
    scalar). Layer 1 is comparable-not-decisive, so there is deliberately NO absolute floor.

    Parameters
    ----------
    gk_med
        Pooled median ``|arm|`` for the imposed-keeper (Regime I) leg.
    nd_med
        Pooled median ``|arm|`` for the nearest-defender paired-vector control leg.
    placebo_p95
        Pooled placebo-band 95th percentile.
    n_domain
        Pooled scored-frame count for this arm.

    Examples
    --------
    >>> layer1_responsiveness_verdict(gk_med=0.30, nd_med=0.10, placebo_p95=0.12, n_domain=300)
    'responsive'
    >>> layer1_responsiveness_verdict(gk_med=0.20, nd_med=0.15, placebo_p95=0.12, n_domain=300)
    'not_responsive'
    """
    if n_domain < MIN_DOMAIN_FRAMES:
        return "arm_unscoreable"
    if not (np.isfinite(float(gk_med)) and np.isfinite(float(nd_med)) and np.isfinite(float(placebo_p95))):
        return "arm_unscoreable"
    thresh = PHYSICS_ARM_PROBE_RATIO * max(float(nd_med), float(placebo_p95))
    return "responsive" if float(gk_med) >= thresh else "not_responsive"


def paired_vector_controls(
    frames: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    r: int,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Single-player paired-vector controls (parent idiom ``tracking/_model_eval.py``): displace ONE
    defending-team outfielder per control by the per-frame ``(imp - actual)`` vector the keeper moved.

    Returns ``{"nearest": frames, "placebo_0": frames, ..., "placebo_{r-1}": frames}`` -- the NEAREST
    defender (``nd``, the keeper-comparable single-player control) and ``r`` SINGLE random outfielders
    (the placebo band). Each is a NEW frame (``frames`` is never mutated). Moving ONE player per control
    is what keeps ``nd`` and each placebo replicate DISTINCT single-player quantities, so the Layer-1
    verdict's ``max(nd_med, placebo_p95)`` is meaningful; a combined multi-player control (nearest + r at
    once) would collapse them onto ONE array and make the ``max`` decorative -- and compare a 1-player
    keeper move against an r+1-player control.

    The placebo pool is the DEFENDING team's outfielders (the keeper's teammates -- a keeper-positioning
    control relocates a generic DEFENDER, never an attacker), drawn one-per-replicate with the nearest in
    the pool (a random single-player null, per the parent idiom). ADR-068: the outfield rows are grouped
    ONCE via :func:`silly_kicks._frame_index.group_rows`. ``targets`` must carry ``imp_x``/``imp_y`` +
    ``actual_x``/``actual_y`` + ``defending_team_id`` -- exactly what
    :func:`impose_defending_keeper_dose` returns.

    Examples
    --------
    Displace the nearest defending-team outfielder + r single-player placebos by the keeper's vector::

        controls = paired_vector_controls(frames, targets, r=3, rng=np.random.default_rng(0))
        nd_control = controls["nearest"]  # only the nearest defender moved
    """
    from silly_kicks._frame_index import group_rows  # ADR-068 grouping seam (imports only id_compat)

    outfield_mask = (~frames["is_ball"].astype(bool) & ~frames["is_goalkeeper"].astype(bool)).to_numpy()
    outfield = frames.loc[outfield_mask]
    groups = group_rows(outfield, (*_FRAME_KEYS, "team_id"))  # key -> DataFrame (empty on miss)
    # numeric targets as float arrays (itertuples attrs are typed `Scalar` -> not float()-able);
    # the itertuples attrs are used ONLY as canonicalised group keys (group_rows handles Scalar).
    actual_x = targets["actual_x"].to_numpy(dtype=float)
    actual_y = targets["actual_y"].to_numpy(dtype=float)
    imp_x = targets["imp_x"].to_numpy(dtype=float)
    imp_y = targets["imp_y"].to_numpy(dtype=float)

    control_names = ["nearest", *(f"placebo_{k}" for k in range(int(r)))]
    # picks[name] accumulates (label, dx, dy). Distinct target frames touch DISJOINT player rows, so
    # each control's label list is duplicate-free and a vectorized per-control loc-assign is exact.
    picks: dict[str, list[tuple[object, float, float]]] = {name: [] for name in control_names}
    for i, tgt in enumerate(targets.itertuples(index=False)):
        sub = groups.get(tgt.game_id, tgt.period_id, tgt.frame_id, tgt.defending_team_id)
        if sub.empty:
            continue
        ax, ay = float(actual_x[i]), float(actual_y[i])
        dx, dy = float(imp_x[i]) - ax, float(imp_y[i]) - ay
        dist = np.hypot(sub["x"].to_numpy(dtype=float) - ax, sub["y"].to_numpy(dtype=float) - ay)
        order = np.argsort(dist, kind="stable")  # positional-in-`sub`; nearest defender first
        pool = sub.index.to_numpy()  # original `frames` labels of this frame's defending outfielders
        picks["nearest"].append((sub.index[int(order[0])], dx, dy))
        for k in range(int(r)):
            # one random defending outfielder per replicate (independent uniform draw; the pool includes
            # the nearest -- a random single-player null, matching the parent idiom's placebo pool).
            picks[f"placebo_{k}"].append((pool[int(rng.integers(len(pool)))], dx, dy))

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
