"""xT framework — pluggable transition family + held-out NLL evaluator.

See NOTICE for full bibliographic citations.
"""

from silly_kicks.xthreat._counterfactual_seam import DestinationProfile, destination_profiles
from silly_kicks.xthreat._eval import (
    compute_holdout_nll,
    compute_holdout_nll_per_group,
    holdout_split,
)
from silly_kicks.xthreat._model import ExpectedThreat
from silly_kicks.xthreat._params import (
    GridSpec,
    KdeKernel,
    KDEParams,
    Method,
    SinghParams,
    XtParams,
    validate_params_for_method,
)
from silly_kicks.xthreat._physical import physical_grid, require_fitted_xt, values_at_points
from silly_kicks.xthreat._transitions import (
    kde_smoothed_transition_matrix,
    silverman_2d,
    singh_transition_matrix,
)
from silly_kicks.xthreat._value_iteration import value_iteration

__all__ = [
    "DestinationProfile",
    "ExpectedThreat",
    "GridSpec",
    "KDEParams",
    "KdeKernel",
    "Method",
    "SinghParams",
    "XtParams",
    "compute_holdout_nll",
    "compute_holdout_nll_per_group",
    "destination_profiles",
    "holdout_split",
    "kde_smoothed_transition_matrix",
    "physical_grid",
    "require_fitted_xt",
    "silverman_2d",
    "singh_transition_matrix",
    "validate_params_for_method",
    "value_iteration",
    "values_at_points",
]
