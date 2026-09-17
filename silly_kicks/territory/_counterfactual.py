"""counterfactual_rows -- the joint ``q * c * xT`` prevented-valuation (TF-54b, spec §5.1/5.2).

The reserved ``method="counterfactual"`` valuation of a defender's territory: a GSAA-analog
``(P_complete - outcome) * xT(target)`` summed over the opponent passes aimed into the defender's
trimmed hull. For a **completed** pass the target is observed (its end); for a **failed** pass -- whose
SPADL ``end`` is the death/recovery location, not the intended target -- the target is modeled over the
death-direction cone's transition distribution ``q`` (renormalized over the cone-and-hull zones), each
hypothesized target ``z`` weighted by a fitted pass-completion ``c(origin, z)`` and the injected xT
value ``xT(z)``.

Frame reconciliation (ADR-028): the hull is in the DEFENDER's action-LTR frame, while the opponent
passes AND the xT grid / transition matrix are in the OPPONENT's frame -- a 180 degree POINT REFLECTION
apart, ``(x, y) -> (fl - x, fw - y)``. The cone, the target distribution ``q``, the completion ``c`` and
``xT(z)`` are all computed in the OPPONENT frame (origin, death, zone centres all opponent-frame -- the
threat the opponent creates at ``z`` in their own attacking frame). ONLY hull MEMBERSHIP reflects the
zone centres (and a completed pass end) into the DEFENDER frame before the point-in-hull test.

Value lookups go through the public ``xthreat`` seam (``destination_profiles`` / ``values_at_points``),
never the raw flat-indexed, y-inverted ``.transition_matrix`` / ``.xT`` (ADR-041). The completion model
is duck-typed and injected (ADR-022) -- no runtime import. A failed pass whose cone-and-hull zones carry
transition support below ``min_transition_support`` (which includes the cone-misses-the-hull case, i.e.
zero selected zones) is an unresolvable target: dropped and COUNTED (never a fabricated 0, ADR-042).

PURE -- never mutates ``defs_grouped`` / ``passes_by_game`` / the caller's ``actions``.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from silly_kicks.xthreat import destination_profiles

from ._columns import (
    TR_DEF_ACTIONS_IN_HULL,
    TR_EXPECTED_THREAT_FACED,
    TR_HULL_AREA,
    TR_HULL_CENTROID_X,
    TR_HULL_CENTROID_Y,
    TR_HULL_SOURCE,
    TR_MEAN_COMPLETION_FACED,
    TR_PASSES_AIMED_INTO_HULL,
    TR_PASSES_INTO_HULL,
    TR_TARGET_SOURCE,
    TR_XT_CONCEDED,
    TR_XT_CONCEDED_FWD,
    TR_XT_CONCEDED_RATE,
    TR_XT_NET,
    TR_XT_PREVENTED,
    TR_XT_PREVENTED_ABOVE_EXPECTATION,
    TR_XT_PREVENTED_FWD,
    TR_XT_PREVENTED_RATE,
)

if TYPE_CHECKING:  # duck-typed at runtime (ADR-022) -- no runtime edge on either injected port
    from collections.abc import Collection, Iterable

    from silly_kicks.expected_passing import PassCompletionModel
    from silly_kicks.xthreat import ExpectedThreat

    from ._config import CounterfactualParams
    from ._hull import Hull

    #: One pre-built defender group: raw player id, raw game id (NA under a window), canonical team id
    #: (a non-NA canonical STRING -- the defender is filtered to a real team upstream), the (possibly
    #: ``None``) trimmed hull, the (N, 2) own-half defensive-action coords, and the canonical game ids
    #: the defender appears in (for opponent-pass lookup). The hull is built by the caller (which owns
    #: ``TerritoryParams.trim_fraction``); this function values the passes.
    DefenderGroup = tuple[object, object, str, "Hull | None", np.ndarray, list]


def _within_cone(dvec: tuple[float, float], zc: np.ndarray, origin: tuple[float, float], cone_deg: float) -> np.ndarray:
    """Boolean mask of zone centres ``zc`` within ``cone_deg`` of the death direction ``dvec`` (OPPONENT
    frame): angle between ``(death - origin)`` and ``(zone_centre - origin)`` <= ``cone_deg``. A
    zero-length death vector (death == origin) selects nothing; a zone centre coincident with the origin
    is excluded (its direction is undefined)."""
    dnorm = float(np.hypot(dvec[0], dvec[1]))
    if dnorm == 0.0:
        return np.zeros(len(zc), dtype=bool)
    zvec = zc - np.asarray(origin, dtype=float)
    znorm = np.hypot(zvec[:, 0], zvec[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (zvec @ np.asarray(dvec, dtype=float)) / (znorm * dnorm)
    cos = np.clip(cos, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    return (ang <= cone_deg) & (znorm > 0)


def counterfactual_rows(
    defs_grouped: Iterable[DefenderGroup],
    passes_by_game: dict[object, pd.DataFrame],
    *,
    xt: ExpectedThreat,
    completion_model: PassCompletionModel,
    params: CounterfactualParams,
    fl: float,
    fw: float,
    window: Collection | None,
) -> tuple[list[dict], dict]:
    """Value each defender's territory under ``method="counterfactual"``. Returns ``(rows, census)``.

    Parameters
    ----------
    defs_grouped : iterable of ``DefenderGroup``
        Pre-built per-defender groups ``(player_raw, game_raw, team_canon, hull, def_xy, games)`` -- the
        hull is built by the caller (which owns the hull params). A ``None`` hull emits a degenerate row.
    passes_by_game : dict
        Canonical-game-id -> the opponent-and-teammate passes for that game, carrying the v1-prepared
        ``start_x``/``start_y``/``end_x``/``end_y``/``result_id`` plus the derived ``_xt_end``
        (``values_at_points`` at the observed end), ``_completed``, ``_forward``, ``_g``, ``_t`` columns.
    xt : ExpectedThreat
        The injected fitted xT model (duck-typed; only consumed via the public ``xthreat`` seam).
    completion_model : PassCompletionModel
        The injected fitted pass-completion model (duck-typed): ``predict_completion(ox, oy, tx, ty)``.
    params : CounterfactualParams
        ``direction_cone_degrees`` + ``min_transition_support``.
    fl, fw : float
        Field length / width (the ADR-028 reflection constants).
    window : Collection or None
        When set, ``game_raw`` is ``pd.NA`` and each group pools its opponent passes across ``games``.

    Returns
    -------
    (rows, census)
        ``rows`` is a list of per-defender output dicts (a resolved row carries the full counterfactual
        schema; a degenerate hull carries only ``territory_hull_source="degenerate"``). ``census`` carries
        ``n_scored`` / ``n_degenerate_hull`` / ``n_passes_into_hull`` / ``n_target_modeled`` /
        ``n_target_unresolved`` for the ``TerritoryReport``.
    """
    cone_deg = float(params.direction_cone_degrees)
    min_support = float(params.min_transition_support)

    rows: list[dict] = []
    n_scored = n_degenerate = n_into_total = 0
    n_target_modeled = n_target_unresolved = 0

    for player_raw, game_raw, team_canon, hull, def_xy, games in defs_grouped:
        base = {"game_id": game_raw, "player_id": player_raw}
        if hull is None:
            n_degenerate += 1
            rows.append({**base, TR_HULL_SOURCE: "degenerate"})  # NaN metrics via the caller's reindex
            continue
        n_scored += 1

        # Opponent passes across this defender's games (a REAL team that is not the defender's; the
        # notna() excludes a NaN-team pass, which .ne would otherwise keep -- mirrors the v1 rule).
        subs = [passes_by_game[g] for g in games if g in passes_by_game]
        opp = pd.concat(subs) if subs else None
        if opp is not None:
            opp = opp[opp["_t"].notna() & opp["_t"].ne(team_canon)]

        conceded = prevented = conceded_fwd = prevented_fwd = 0.0
        expected_faced = completion_sum = 0.0
        n_into = n_aimed = 0
        has_observed = has_modeled = has_unresolved = False

        if opp is not None and len(opp):
            completed = opp["_completed"].to_numpy(dtype=bool)
            forward = opp["_forward"].to_numpy(dtype=bool)
            end_x = opp["end_x"].to_numpy(dtype=float)
            end_y = opp["end_y"].to_numpy(dtype=float)
            start_x = opp["start_x"].to_numpy(dtype=float)
            start_y = opp["start_y"].to_numpy(dtype=float)
            xt_end = opp["_xt_end"].to_numpy(dtype=float)

            # v1 descriptive passes_into_hull: any pass (completed OR failed) whose observed end reflects
            # into the hull with a finite xT (method-invariant, spec §5.3).
            refl_end = np.column_stack([fl - end_x, fw - end_y])
            end_in_hull = hull.contains(refl_end)
            n_into = int((end_in_hull & np.isfinite(xt_end)).sum())

            # --- completed passes: valued at the observed end (no integration) --------------------------
            comp = completed & end_in_hull & np.isfinite(xt_end)
            if comp.any():
                c_comp = np.asarray(
                    completion_model.predict_completion(start_x[comp], start_y[comp], end_x[comp], end_y[comp]),
                    dtype=float,
                )
                v_comp = xt_end[comp]
                conceded = float(v_comp.sum())
                expected_faced += float((c_comp * v_comp).sum())
                completion_sum += float(c_comp.sum())
                n_aimed += int(comp.sum())
                conceded_fwd = float(v_comp[forward[comp]].sum())
                has_observed = True

            # --- failed passes: valued over the death-direction cone's target distribution q ------------
            failed = ~completed
            if failed.any():
                fox, foy = start_x[failed], start_y[failed]
                fdx, fdy = end_x[failed], end_y[failed]
                f_forward = forward[failed]
                prof = destination_profiles(xt, fox, foy)
                centres = prof.zone_centres  # (n_zones, 2) opponent-frame physical
                values = prof.zone_values  # (n_zones,) xT at each zone centre (opponent frame)
                probs = prof.probabilities  # (n_failed, n_zones) raw transition rows
                # Hull membership of the zone centres is origin-independent -> compute once.
                refl_centres = np.column_stack([fl - centres[:, 0], fw - centres[:, 1]])
                in_hull = hull.contains(refl_centres)

                for i in range(int(failed.sum())):
                    in_cone = _within_cone((fdx[i] - fox[i], fdy[i] - foy[i]), centres, (fox[i], foy[i]), cone_deg)
                    sel = in_cone & in_hull
                    support = float(probs[i][sel].sum())
                    if support < min_support:  # includes zero selected zones (cone misses the hull)
                        n_target_unresolved += 1
                        has_unresolved = True
                        continue
                    q = probs[i][sel] / support
                    c_z = np.asarray(
                        completion_model.predict_completion(fox[i], foy[i], centres[sel, 0], centres[sel, 1]),
                        dtype=float,
                    )
                    contrib = float((q * c_z * values[sel]).sum())
                    prevented += contrib
                    expected_faced += contrib
                    completion_sum += float((q * c_z).sum())  # q-weighted per-pass completion
                    n_aimed += 1
                    n_target_modeled += 1
                    has_modeled = True
                    if f_forward[i]:
                        prevented_fwd += contrib

        n_into_total += n_into
        # Worst-case per-defender provenance summary (per-pass provenance lives in the census counts): a
        # single unresolved target surfaces, else modeled, else observed; NaN if no aimed passes at all.
        target_source = (
            "unresolved" if has_unresolved else "modeled" if has_modeled else "observed" if has_observed else None
        )

        rows.append(
            {
                **base,
                TR_XT_CONCEDED: conceded,
                TR_XT_PREVENTED: prevented,
                TR_XT_NET: conceded - prevented,
                TR_XT_CONCEDED_FWD: conceded_fwd,
                TR_XT_PREVENTED_FWD: prevented_fwd,
                TR_PASSES_INTO_HULL: n_into,
                # Rates use the counterfactual scoring denominator passes_aimed_into_hull (spec §5.3).
                TR_XT_CONCEDED_RATE: (conceded / n_aimed) if n_aimed else np.nan,
                TR_XT_PREVENTED_RATE: (prevented / n_aimed) if n_aimed else np.nan,
                TR_HULL_AREA: hull.area,
                TR_HULL_CENTROID_X: hull.centroid[0],
                TR_HULL_CENTROID_Y: hull.centroid[1],
                TR_DEF_ACTIONS_IN_HULL: int(hull.contains(def_xy).sum()),
                TR_HULL_SOURCE: "resolved",
                TR_EXPECTED_THREAT_FACED: expected_faced,
                TR_XT_PREVENTED_ABOVE_EXPECTATION: expected_faced - conceded,
                TR_PASSES_AIMED_INTO_HULL: n_aimed,
                TR_MEAN_COMPLETION_FACED: (completion_sum / n_aimed) if n_aimed else np.nan,
                TR_TARGET_SOURCE: target_source,
            }
        )

    census = {
        "n_scored": n_scored,
        "n_degenerate_hull": n_degenerate,
        "n_passes_into_hull": n_into_total,
        "n_target_modeled": n_target_modeled,
        "n_target_unresolved": n_target_unresolved,
    }
    return rows, census
