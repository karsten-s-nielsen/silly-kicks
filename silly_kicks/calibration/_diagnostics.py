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


def exceeds_noise_floor(gain: float, se: float) -> bool:
    """True iff ``se`` is finite and ``gain`` strictly exceeds it.

    A ``None``/``nan``/``inf`` SE (single fold, or an undefined spread) never clears the floor: a gain
    "beats the noise" only when the noise is a finite, measured quantity.

    Examples
    --------
    >>> from silly_kicks.calibration._diagnostics import exceeds_noise_floor
    >>> exceeds_noise_floor(0.06, 0.05)  # a gain clears a finite floor
    True
    >>> exceeds_noise_floor(0.05, 0.05)  # strict: an exact tie does not clear
    False
    >>> exceeds_noise_floor(0.06, float("nan"))  # a non-finite floor never clears
    False
    """
    if se is None or not math.isfinite(se):
        return False
    return gain > se


def tf25_gate_fires(*, global_brier: float, provider_best_brier: float, provider_se: float) -> bool:
    """True if the (global - provider-best) Brier gap exceeds the provider's CV SE.

    Examples
    --------
    >>> from silly_kicks.calibration._diagnostics import tf25_gate_fires
    >>> tf25_gate_fires(global_brier=0.052, provider_best_brier=0.050, provider_se=0.005)
    False
    """
    return exceeds_noise_floor(global_brier - provider_best_brier, provider_se)
