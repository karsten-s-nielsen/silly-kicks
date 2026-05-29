"""TF-25 provider-specific-defaults gate + k3 sensitivity (spec §5 diagnostics).

The principled TF-25 trigger: a provider gets its own default ONLY if the gap between the
global-optimum Brier and that provider's own best-k3 Brier exceeds that provider's CV standard
error (computed against the scheme it actually uses — GroupKFold-5 for GS/SkillCorner, LOMO for
IDSSE). A nan SE (single fold) can never justify a provider-specific default.

Examples
--------
>>> from silly_kicks.calibration._diagnostics import tf25_gate_fires
>>> tf25_gate_fires(global_brier=0.06, provider_best_brier=0.05, provider_se=0.005)
True
"""

from __future__ import annotations

import math


def tf25_gate_fires(*, global_brier: float, provider_best_brier: float, provider_se: float) -> bool:
    """True if the (global - provider-best) Brier gap exceeds the provider's CV SE.

    Examples
    --------
    >>> from silly_kicks.calibration._diagnostics import tf25_gate_fires
    >>> tf25_gate_fires(global_brier=0.052, provider_best_brier=0.050, provider_se=0.005)
    False
    """
    if provider_se is None or math.isnan(provider_se):
        return False
    return (global_brier - provider_best_brier) > provider_se
