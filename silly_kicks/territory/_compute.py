"""compute_territorial_dominance -- TF-54 "Van Dijk" territorial dominance (spec §5).

Event-only. For each defender, build the trimmed convex hull of their OWN-HALF defensive-action
locations, then value the OPPONENT passes whose destination lands inside that hull: threat CONCEDED
(completed) vs PREVENTED (failed), by an INJECTED fitted ``ExpectedThreat`` (silly-kicks ships no xT).

Frame reconciliation (ADR-028): the hull is in the DEFENDER's action-LTR frame while the opponent's
passes are in the opponent's frame -- a 180 degree point reflection apart. Hull MEMBERSHIP reflects the
opponent pass end into the defender frame ``(105 - end_x, 68 - end_y)``; the xT VALUE is taken on the
pass end in the opponent's OWN frame (``values_at_points``). A failed pass's SPADL ``end`` is the
death/recovery location, not the intended target (``_derive_end_coordinates``), so the default
``completed_failed`` method measures "threat that reached / died in the territory", not "threat that
would have been created" -- the ``counterfactual`` method (TF-54b, ``_counterfactual.py``) closes that
gap by modeling the failed pass's intended target over its death-direction cone (spec §5.1/5.2). Output
ids are RAW; keepers are grouped on the CANONICAL id (ADR-019). PURE -- never mutates ``actions``.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from silly_kicks.id_compat import canonical_id_series
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.xthreat import require_fitted_xt, values_at_points

from ._columns import (
    TERRITORY_METHODS,
    TR_DEF_ACTIONS_IN_HULL,
    TR_HULL_AREA,
    TR_HULL_CENTROID_X,
    TR_HULL_CENTROID_Y,
    TR_HULL_SOURCE,
    TR_PASSES_INTO_HULL,
    TR_XT_CONCEDED,
    TR_XT_CONCEDED_FWD,
    TR_XT_CONCEDED_RATE,
    TR_XT_NET,
    TR_XT_PREVENTED,
    TR_XT_PREVENTED_FWD,
    TR_XT_PREVENTED_RATE,
    columns_for_method,
)
from ._config import CounterfactualParams, TerritoryParams
from ._counterfactual import counterfactual_rows
from ._hull import build_trimmed_hull
from ._report import TerritoryReport

if TYPE_CHECKING:  # duck-typed at runtime (ADR-022) -- no runtime xthreat / completion-model edge
    from collections.abc import Collection

    from silly_kicks.expected_passing import PassCompletionModel
    from silly_kicks.xthreat import ExpectedThreat

    from ._counterfactual import DefenderGroup

_DEFAULT = TerritoryParams()
_PASS_TYPE_ID = spadlconfig.actiontype_id["pass"]  # SPADL uses type_id; type_name is add_names-only
_SUCCESS = spadlconfig.result_id["success"]
_PASS_COLS = ["game_id", "team_id", "start_x", "start_y", "end_x", "end_y", "result_id"]


def compute_territorial_dominance(
    actions: pd.DataFrame,
    *,
    xt: ExpectedThreat,
    method: str = "completed_failed",
    window: Collection | None = None,
    params: TerritoryParams | CounterfactualParams = _DEFAULT,
    completion_model: PassCompletionModel | None = None,
    cf_params: CounterfactualParams | None = None,
) -> tuple[pd.DataFrame, TerritoryReport]:
    """Territorial dominance per defender. Returns ``(samples, TerritoryReport)``.

    ``window=None`` -> one row per ``(game_id, player_id)``; a ``window`` (a collection of ``game_id``s)
    -> one row per ``player_id`` pooled over those games (the hull re-derived over the pooled actions).

    ``method="completed_failed"`` (default) values opponent passes at their observed end.
    ``method="counterfactual"`` (TF-54b) requires an injected ``completion_model`` and models each
    failed pass's intended target over its death-direction cone (spec §5.1/5.2); its
    ``CounterfactualParams`` may be passed either as ``cf_params=`` or directly as ``params=`` (in which
    case the hull uses the default ``TerritoryParams``).

    Examples
    --------
    Value opponent passes into each defender's trimmed defensive hull with an injected fitted xT
    (``tests/territory/test_compute.py`` has the worked numbers)::

        from silly_kicks.xthreat import ExpectedThreat
        from silly_kicks.territory import compute_territorial_dominance

        xt = ExpectedThreat().fit(corpus_actions)
        samples, report = compute_territorial_dominance(actions, xt=xt)
    """
    require_fitted_xt(xt, caller="compute_territorial_dominance")
    if method not in TERRITORY_METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(TERRITORY_METHODS)}")

    # Resolve the hull/defensive params (always a TerritoryParams) and, for the counterfactual method,
    # its CounterfactualParams. Passing a CounterfactualParams directly as `params` is a convenience that
    # keeps the hull on the default TerritoryParams (the method has no hull knobs of its own).
    if isinstance(params, CounterfactualParams):
        territory_params: TerritoryParams = _DEFAULT
        cf: CounterfactualParams = params
    else:
        territory_params = params
        cf = cf_params if cf_params is not None else CounterfactualParams.default()

    if method == "counterfactual" and completion_model is None:
        raise ValueError(
            "method='counterfactual' requires a fitted completion_model "
            "(silly-kicks ships no pass-completion model; inject one -- spec §5b)."
        )

    fl = float(spadlconfig.field_length)
    fw = float(spadlconfig.field_width)

    a = actions
    if window is not None:
        wanted = set(canonical_id_series(pd.Series(list(window), dtype="object")))
        a = a[canonical_id_series(a["game_id"]).isin(wanted)]

    # --- defensive actions that seed each hull (own-half, real player) ---
    # SPADL carries type_id (type_name is add_names-only); map the params' type NAMES to ids.
    def_type_ids = frozenset(spadlconfig.actiontype_id[n] for n in territory_params.defensive_action_types)
    is_def = (
        a["type_id"].isin(def_type_ids)
        & a["player_id"].notna()
        & a["team_id"].notna()  # a defender must have a real team (so the opponent is well-defined)
        & (a["start_x"] < territory_params.own_half_max_x)
    )
    defs = a.loc[is_def, ["game_id", "player_id", "team_id", "start_x", "start_y"]].copy()

    # --- opponent passes valued at their end (opponent frame) ---
    passes = a.loc[a["type_id"] == _PASS_TYPE_ID, _PASS_COLS].copy()
    passes["_xt_end"] = values_at_points(xt, passes["end_x"].to_numpy(), passes["end_y"].to_numpy())
    passes["_completed"] = (passes["result_id"] == _SUCCESS).to_numpy()
    passes["_forward"] = (passes["end_x"] - passes["start_x"]).to_numpy() > territory_params.forward_threshold_m
    passes["_g"] = canonical_id_series(passes["game_id"])
    passes["_t"] = canonical_id_series(passes["team_id"])
    passes_by_game: dict[object, pd.DataFrame] = {g: sub for g, sub in passes.groupby("_g", sort=False)}

    out_columns = columns_for_method(method)
    empty = pd.DataFrame({c: pd.Series(dtype=t) for c, t in out_columns.items()})
    if defs.empty:
        report = TerritoryReport(territory_params, 0, 0, 0, 0, len(passes), 0)
        return empty, report

    defs["_g"] = canonical_id_series(defs["game_id"])
    defs["_p"] = canonical_id_series(defs["player_id"])
    defs["_t"] = canonical_id_series(defs["team_id"])
    group_cols = ["_p"] if window is not None else ["_g", "_p"]

    if method == "counterfactual":
        return _counterfactual_dispatch(
            defs,
            passes,
            passes_by_game,
            group_cols=group_cols,
            window=window,
            xt=xt,
            completion_model=completion_model,  # type: ignore[arg-type]  (guarded non-None above)
            territory_params=territory_params,
            cf=cf,
            out_columns=out_columns,
            fl=fl,
            fw=fw,
        )

    rows: list[dict] = []
    n_scored = n_degenerate = n_into_total = 0
    for _key, drows in defs.groupby(group_cols, sort=False):
        player_raw = drows["player_id"].iloc[0]
        game_raw = pd.NA if window is not None else drows["game_id"].iloc[0]
        team_canon = drows["_t"].iloc[0]
        def_xy = drows[["start_x", "start_y"]].to_numpy(dtype=float)

        hull = build_trimmed_hull(def_xy, trim_fraction=territory_params.trim_fraction)
        base = {"game_id": game_raw, "player_id": player_raw}
        if hull is None:
            n_degenerate += 1
            rows.append({**base, TR_HULL_SOURCE: "degenerate"})  # NaN metrics via reindex below
            continue
        n_scored += 1

        games = list(dict.fromkeys(drows["_g"].tolist()))
        subs = [passes_by_game[g] for g in games if g in passes_by_game]
        opp = pd.concat(subs) if subs else passes.iloc[:0]
        # Opponent passes only: a REAL team (notna) that is not the defender's. Canonical strings -> a
        # plain .ne is safe (ADR-019); the notna() excludes a NaN-team pass (which .ne would keep as
        # True), so an unattributable pass is never counted as an opponent's.
        opp = opp[opp["_t"].notna() & opp["_t"].ne(team_canon)]

        conceded = prevented = conceded_fwd = prevented_fwd = 0.0
        n_into = 0
        if len(opp):
            refl = np.column_stack([fl - opp["end_x"].to_numpy(float), fw - opp["end_y"].to_numpy(float)])
            in_hull = hull.contains(refl)
            xt_end = opp["_xt_end"].to_numpy(float)
            completed = opp["_completed"].to_numpy(bool)
            forward = opp["_forward"].to_numpy(bool)
            valid = in_hull & np.isfinite(xt_end)  # only valued passes count toward the metric
            n_into = int(valid.sum())
            conceded = float(xt_end[valid & completed].sum())
            prevented = float(xt_end[valid & ~completed].sum())
            conceded_fwd = float(xt_end[valid & completed & forward].sum())
            prevented_fwd = float(xt_end[valid & ~completed & forward].sum())
        n_into_total += n_into

        rows.append(
            {
                **base,
                TR_XT_CONCEDED: conceded,
                TR_XT_PREVENTED: prevented,
                TR_XT_NET: conceded - prevented,
                TR_XT_CONCEDED_FWD: conceded_fwd,
                TR_XT_PREVENTED_FWD: prevented_fwd,
                TR_PASSES_INTO_HULL: n_into,
                TR_XT_CONCEDED_RATE: (conceded / n_into) if n_into else np.nan,
                TR_XT_PREVENTED_RATE: (prevented / n_into) if n_into else np.nan,
                TR_HULL_AREA: hull.area,
                TR_HULL_CENTROID_X: hull.centroid[0],
                TR_HULL_CENTROID_Y: hull.centroid[1],
                TR_DEF_ACTIONS_IN_HULL: int(hull.contains(def_xy).sum()),
                TR_HULL_SOURCE: "resolved",
            }
        )

    out = pd.DataFrame(rows).reindex(columns=list(out_columns)).astype(out_columns)
    report = TerritoryReport(
        territory_params,
        n_players_in=n_scored + n_degenerate,
        n_scored=n_scored,
        n_degenerate_hull=n_degenerate,
        n_no_actions=0,
        n_passes_considered=len(passes),
        n_passes_into_hull=n_into_total,
    )
    return out, report


def _counterfactual_dispatch(
    defs: pd.DataFrame,
    passes: pd.DataFrame,
    passes_by_game: dict[object, pd.DataFrame],
    *,
    group_cols: list[str],
    window: Collection | None,
    xt: ExpectedThreat,
    completion_model: PassCompletionModel,
    territory_params: TerritoryParams,
    cf: CounterfactualParams,
    out_columns: dict[str, str],
    fl: float,
    fw: float,
) -> tuple[pd.DataFrame, TerritoryReport]:
    """Build the pre-hull'd defender groups (the hull uses ``territory_params.trim_fraction``) and hand
    them to ``counterfactual_rows`` for the ``q * c * xT`` valuation, then assemble the frame + report."""
    groups: list[DefenderGroup] = []
    for _key, drows in defs.groupby(group_cols, sort=False):
        player_raw = drows["player_id"].iloc[0]
        game_raw = pd.NA if window is not None else drows["game_id"].iloc[0]
        team_canon = drows["_t"].iloc[0]
        def_xy = drows[["start_x", "start_y"]].to_numpy(dtype=float)
        hull = build_trimmed_hull(def_xy, trim_fraction=territory_params.trim_fraction)
        games = list(dict.fromkeys(drows["_g"].tolist()))
        groups.append((player_raw, game_raw, team_canon, hull, def_xy, games))

    rows, census = counterfactual_rows(
        groups,
        passes_by_game,
        xt=xt,
        completion_model=completion_model,
        params=cf,
        fl=fl,
        fw=fw,
        window=window,
    )

    out = pd.DataFrame(rows).reindex(columns=list(out_columns)).astype(out_columns)
    report = TerritoryReport(
        territory_params,
        n_players_in=census["n_scored"] + census["n_degenerate_hull"],
        n_scored=census["n_scored"],
        n_degenerate_hull=census["n_degenerate_hull"],
        n_no_actions=0,
        n_passes_considered=len(passes),
        n_passes_into_hull=census["n_passes_into_hull"],
        n_target_modeled=census["n_target_modeled"],
        n_target_unresolved=census["n_target_unresolved"],
    )
    return out, report
