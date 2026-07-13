"""Public causal-validation toolkit (ADR-015, promoted by TF-19/ADR-037 as the second
consumer). Pure numpy/sklearn propensity matching + a parameterized opportunity-row
builder. The xCross harness configuration is the default-constants path; the TF-19
shot arm is expressible purely as builder arguments (tested)."""

from silly_kicks.causal.matching import (
    GK_ABLATION_MIN_SHIFT,
    PLACEBO_BAND_PERCENTILE,
    CausalEstimate,
    abadie_imbens_se,
    estimate_atnt,
    estimate_att,
    fit_propensity,
    placebo_shift,
    propensity_match,
    smd_balance,
)
from silly_kicks.causal.opportunities import (
    SHOT_ARM_CONFOUNDERS,
    OpportunityConfig,
    build_opportunities,
    shot_arm_config,
    xcross_config,
)

__all__ = [
    "GK_ABLATION_MIN_SHIFT",
    "PLACEBO_BAND_PERCENTILE",
    "SHOT_ARM_CONFOUNDERS",
    "CausalEstimate",
    "OpportunityConfig",
    "abadie_imbens_se",
    "build_opportunities",
    "estimate_atnt",
    "estimate_att",
    "fit_propensity",
    "placebo_shift",
    "propensity_match",
    "shot_arm_config",
    "smd_balance",
    "xcross_config",
]
