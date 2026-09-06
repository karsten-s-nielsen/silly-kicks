"""Territorial-dominance metric output columns (TF-54). Single source for the schema every gate iterates.

Grain: one output row per (game_id, player_id) -- the defender whose own-half defensive-action centroid
defines the hull. Columns carry a ``territory_`` prefix (collision-safe in the global glossary).
"""

from __future__ import annotations

#: Grain keys.
TERRITORY_KEYS = ["game_id", "player_id"]

#: The `method=` valuation family (spec §5.3). ``completed_failed`` is the default -- opponent passes
#: valued at their observed end. ``counterfactual`` (TF-54b, ADR-089) is implemented: a
#: completion-weighted expected-minus-realized valuation over a modeled failed-pass target
#: distribution (requires an injected ``PassCompletionModel``).
TERRITORY_METHODS = frozenset({"completed_failed", "counterfactual"})

# Metric column names (spec §5.6). xT / area / coords / rates -> float64; counts -> Int64.
TR_XT_CONCEDED = "territory_xt_conceded"
TR_XT_PREVENTED = "territory_xt_prevented"
TR_XT_NET = "territory_xt_net"  # == conceded - prevented
TR_XT_CONCEDED_FWD = "territory_xt_conceded_forward"
TR_XT_PREVENTED_FWD = "territory_xt_prevented_forward"
TR_PASSES_INTO_HULL = "territory_passes_into_hull"
TR_XT_CONCEDED_RATE = "territory_xt_conceded_rate"  # conceded / passes_into_hull (NaN on zero volume)
TR_XT_PREVENTED_RATE = "territory_xt_prevented_rate"
TR_HULL_AREA = "territory_hull_area_m2"
TR_HULL_CENTROID_X = "territory_hull_centroid_x"
TR_HULL_CENTROID_Y = "territory_hull_centroid_y"
TR_DEF_ACTIONS_IN_HULL = "territory_defensive_actions_in_hull"

#: Provenance over {resolved, degenerate, no_actions} (the das_source idiom). Never a fabricated 0/NaN
#: masquerading as data -- a degenerate/absent hull drops the row (counted in TerritoryReport).
TR_HULL_SOURCE = "territory_hull_source"

#: The derived metric columns (documented in feature_glossary; keys + provenance are structural).
TERRITORY_METRIC_COLUMNS = [
    TR_XT_CONCEDED,
    TR_XT_PREVENTED,
    TR_XT_NET,
    TR_XT_CONCEDED_FWD,
    TR_XT_PREVENTED_FWD,
    TR_PASSES_INTO_HULL,
    TR_XT_CONCEDED_RATE,
    TR_XT_PREVENTED_RATE,
    TR_HULL_AREA,
    TR_HULL_CENTROID_X,
    TR_HULL_CENTROID_Y,
    TR_DEF_ACTIONS_IN_HULL,
]

#: Full output column order + dtype (keys/provenance object; counts nullable Int64; xT/area/coords/rates
#: float64).
TERRITORY_COLUMNS: dict[str, str] = {
    "game_id": "object",
    "player_id": "object",
    TR_XT_CONCEDED: "float64",
    TR_XT_PREVENTED: "float64",
    TR_XT_NET: "float64",
    TR_XT_CONCEDED_FWD: "float64",
    TR_XT_PREVENTED_FWD: "float64",
    TR_PASSES_INTO_HULL: "Int64",
    TR_XT_CONCEDED_RATE: "float64",
    TR_XT_PREVENTED_RATE: "float64",
    TR_HULL_AREA: "float64",
    TR_HULL_CENTROID_X: "float64",
    TR_HULL_CENTROID_Y: "float64",
    TR_DEF_ACTIONS_IN_HULL: "Int64",
    TR_HULL_SOURCE: "object",
}

#: Closed provenance vocabulary for TR_HULL_SOURCE.
TERRITORY_HULL_SOURCE_VALUES = frozenset({"resolved", "degenerate", "no_actions"})

# --- counterfactual-only columns (SPEC-04 -- method="counterfactual", spec sec 5.3) --------------------
#: sum P_complete * xT(target) over aimed-in passes (failed + completed).
TR_EXPECTED_THREAT_FACED = "territory_expected_threat_faced"
#: GSAA-style headline: sum (P_complete - outcome) * xT(target) over aimed-in passes.
TR_XT_PREVENTED_ABOVE_EXPECTATION = "territory_xt_prevented_above_expectation"
#: The counterfactual scoring denominator -- passes whose death-direction cone intersects the hull.
TR_PASSES_AIMED_INTO_HULL = "territory_passes_aimed_into_hull"
#: Mean completion probability c over aimed-in passes (interpretability companion).
TR_MEAN_COMPLETION_FACED = "territory_mean_completion_faced"
#: Per-pass target-resolution provenance (never a fabricated target -- ADR-042 drop-and-count idiom).
TR_TARGET_SOURCE = "territory_target_source"

#: Closed provenance vocabulary for TR_TARGET_SOURCE. "observed" = a completed pass's real end;
#: "modeled" = a failed pass whose target distribution q cleared min_transition_support; "unresolved" =
#: a failed pass whose cone-restricted transition mass was insufficient (dropped-and-counted, never a
#: fabricated 0 -- ADR-042).
TERRITORY_TARGET_SOURCE_VALUES = frozenset({"observed", "modeled", "unresolved"})

#: The 5 columns that exist ONLY under method="counterfactual" (spec sec 5.3 table). 4 are real
#: METRIC columns, documented in silly_kicks.feature_glossary (emitting_module
#: "silly_kicks.territory._counterfactual"). TR_TARGET_SOURCE is PROVENANCE and is deliberately
#: EXCLUDED from the glossary -- mirroring TR_HULL_SOURCE above, which is likewise never
#: glossaried (structural, not a metric; the ADR-042 drop-and-count idiom). Neither provenance
#: column appears in TERRITORY_METRIC_COLUMNS / the run-and-diff glossary-coverage harness.
_COUNTERFACTUAL_ONLY_COLUMNS: dict[str, str] = {
    TR_EXPECTED_THREAT_FACED: "float64",
    TR_XT_PREVENTED_ABOVE_EXPECTATION: "float64",
    TR_PASSES_AIMED_INTO_HULL: "Int64",
    TR_MEAN_COMPLETION_FACED: "float64",
    TR_TARGET_SOURCE: "object",
}


def columns_for_method(method: str) -> dict[str, str]:
    """Column-name -> dtype schema for one ``method`` in the ``TERRITORY_METHODS`` family (SPEC-04).

    ``"completed_failed"`` returns exactly the v1 schema (``TERRITORY_COLUMNS`` verbatim, so its
    shape/values are untouched by this resolver's existence); ``"counterfactual"`` returns that same
    v1 schema plus the 5 counterfactual-only columns (spec sec 5.3). An unrecognized method raises
    ``ValueError`` rather than silently returning an incomplete schema.

    Examples
    --------
    >>> from silly_kicks.territory._columns import columns_for_method, TERRITORY_COLUMNS
    >>> columns_for_method("completed_failed") == dict(TERRITORY_COLUMNS)
    True
    >>> "territory_expected_threat_faced" in columns_for_method("counterfactual")
    True
    """
    if method == "completed_failed":
        return dict(TERRITORY_COLUMNS)
    if method == "counterfactual":
        return {**TERRITORY_COLUMNS, **_COUNTERFACTUAL_ONLY_COLUMNS}
    raise ValueError(f"unknown method {method!r}; expected one of {sorted(TERRITORY_METHODS)}")
