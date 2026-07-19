"""Calibration gates (spec §5).

H1 degenerate-feature gate: if a tuned feature's variance collapses below 10% of its default-param
variance, the trial is steered away with a finite penalty Brier. The penalty MAGNITUDE is anchored
to the default-param held-out Brier (computed once in prepare()), NOT to a running "worst-observed"
value — keeping the objective STATELESS, resume-stable, and path-comparable (so
assert_cache_equivalence can check it). See spec §5 / R1.

Signal-sanity gate: a provider contributing ~0 matched events is excluded with a loud warning at
LOAD time (data-determined, fixed for the whole study) — never silently averaged in.

Examples
--------
>>> from silly_kicks.calibration._gates import h1_penalty_fires
>>> # if h1_penalty_fires(X_trial, default_variances): return penalty
"""

from __future__ import annotations

import warnings

import pandas as pd

from silly_kicks.calibration._features import _TRIAL_DEPENDENT_COLS

VARIANCE_GATE_RATIO = 0.1  # H1: < 10% of default-param variance => degenerate
PENALTY_K = 5.0  # penalty = K * default_param_brier (R1: stateless, ~5x any real trial)


def default_feature_variances(default_x: pd.DataFrame) -> dict[str, float]:
    """Variance of each trial-dependent feature at the DEFAULT params (computed once).

    Only the ``_TRIAL_DEPENDENT_COLS`` are measured: they are the columns a trial can
    actually move, so they are the only ones whose collapse says anything about the trial.
    A trial-INVARIANT column is skipped even when present, and so is a trial-dependent
    column the caller's frame does not carry -- the returned dict is the H1 anchor, and an
    entry that could never change would only add noise to it.

    Examples
    --------
    >>> import pandas as pd
    >>> from silly_kicks.calibration._gates import default_feature_variances
    >>> default_x = pd.DataFrame(
    ...     {
    ...         "pressure_on_actor__link_zones": [0.0, 1.0, 2.0],  # trial-dependent
    ...         "start_x": [10.0, 20.0, 30.0],  # invariant -- not an anchor
    ...     }
    ... )
    >>> default_feature_variances(default_x)
    {'pressure_on_actor__link_zones': 1.0}
    """
    return {c: float(default_x[c].var()) for c in _TRIAL_DEPENDENT_COLS if c in default_x.columns}


def h1_penalty_fires(trial_x: pd.DataFrame, default_variances: dict[str, float]) -> bool:
    """True if any tuned feature's variance dropped below 10% of its default variance.

    A degenerate feature is one the trial has flattened into a near-constant, which scores
    deceptively well while measuring nothing. Firing means "return the penalty Brier", never
    "raise": the trial is steered away, and the study continues.

    Two anchors deliberately CANNOT fire, so the gate never punishes a trial for something
    it did not do: a default variance of 0 (the feature was already constant -- there is no
    collapse to detect and no ratio to take), and a column absent from ``trial_x``.

    Examples
    --------
    >>> import warnings
    >>> import pandas as pd
    >>> from silly_kicks.calibration._gates import h1_penalty_fires
    >>> trial_x = pd.DataFrame({"pressure_on_actor__link_zones": [1.0, 1.0, 1.0001]})
    >>> with warnings.catch_warnings():  # firing always warns
    ...     warnings.simplefilter("ignore")
    ...     h1_penalty_fires(trial_x, {"pressure_on_actor__link_zones": 1.0})
    True
    >>> h1_penalty_fires(trial_x, {"pressure_on_actor__link_zones": 0.0})
    False
    >>> h1_penalty_fires(pd.DataFrame(), {"pressure_on_actor__link_zones": 1.0})
    False
    """
    for col, default_var in default_variances.items():
        if col not in trial_x.columns or default_var <= 0:
            continue
        current_var = float(trial_x[col].var())
        if current_var / default_var < VARIANCE_GATE_RATIO:
            warnings.warn(
                f"H1 gate: {col} variance {current_var:.6g} < 10% of default "
                f"{default_var:.6g} — returning penalty Brier",
                UserWarning,
                stacklevel=2,
            )
            return True
    return False


def signal_sanity(per_provider_value: dict[str, float], *, min_value: float = 0.01) -> tuple[list[str], list[str]]:
    """Split providers into (kept, excluded); a ~0-signal provider is excluded loudly.

    Examples
    --------
    >>> from silly_kicks.calibration._gates import signal_sanity
    >>> import warnings
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     signal_sanity({"a": 0.8, "b": 0.0}, min_value=0.5)
    (['a'], ['b'])
    """
    kept, excluded = [], []
    for provider, value in per_provider_value.items():
        if value is None or value < min_value:
            excluded.append(provider)
            warnings.warn(
                f"Signal-sanity gate: provider {provider!r} contributes ~0 signal "
                f"({value}) — excluded from the equal-weight mean, not silently averaged in",
                UserWarning,
                stacklevel=2,
            )
        else:
            kept.append(provider)
    return kept, excluded
