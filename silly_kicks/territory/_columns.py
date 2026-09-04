"""Territorial-dominance metric output columns (TF-54). Single source for the schema every gate iterates.

Grain: one output row per (game_id, player_id) -- the defender whose own-half defensive-action centroid
defines the hull. Columns carry a ``territory_`` prefix (collision-safe in the global glossary).
"""

from __future__ import annotations

#: Grain keys.
TERRITORY_KEYS = ["game_id", "player_id"]

#: The `method=` valuation family (spec §5.3). ``completed_failed`` is the default + only implemented
#: leg; ``counterfactual`` is a reserved typed door (raises NotImplementedError until its own
#: construct-validated follow-on).
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
