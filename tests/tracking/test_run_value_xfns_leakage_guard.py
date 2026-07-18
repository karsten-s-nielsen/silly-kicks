"""TF-35 leakage guard: ``off_ball_run_value_xfns`` is opt-in and must stay out of
every default/union xfn list (ADR-042, inheriting the ADR-039 F4 decision).

The TF-35 domain gates on the action's OWN ``result_id`` (only completed passes and
crosses are valued), so as an a0-slot feature it is exactly the result-leakage class
``HybridVAEP`` exists to strip. Same enforcement shape as
``test_packing_xfns_leakage_guard.py``: auto-discover the default surfaces so a future
default list is covered without editing this file, plus a name anchor so a rename fails
loudly here instead of silently neutering the substring check.
"""

import importlib

import numpy as np
import pandas as pd

from silly_kicks.xthreat import ExpectedThreat

_MODULES = (
    "silly_kicks.tracking.features",
    "silly_kicks.atomic.tracking.features",
    "silly_kicks.vaep",
    "silly_kicks.vaep.base",
    "silly_kicks.atomic.vaep",
    "silly_kicks.atomic.vaep.base",
)

#: ``off_ball_run_value_xfns`` names its transformer ``off_ball_run_values``.
_FORBIDDEN_NAME = "off_ball_run_values"


def _fitted_xt() -> ExpectedThreat:
    m = ExpectedThreat()
    m.xT[:] = np.linspace(0.01, 0.3, m.xT.shape[1])[np.newaxis, :]
    return m


def _default_lists():
    found = {}
    for modname in _MODULES:
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        for attr in dir(mod):
            if "default_xfns" in attr or attr.startswith("xfns_default") or attr.startswith("hybrid_xfns_default"):
                obj = getattr(mod, attr)
                if isinstance(obj, list):
                    found[f"{modname}.{attr}"] = obj
    return found


def test_default_lists_discovered():
    """Floor sanity: the guard below is not vacuously green."""
    lists = _default_lists()
    assert any("tracking.features.tracking_default_xfns" in k for k in lists)
    assert any("vaep.base.xfns_default" in k for k in lists)
    assert len(lists) >= 10


def test_run_value_transformer_name():
    """Anchor: the substring guard only protects while the transformer keeps this name."""
    import silly_kicks.atomic.tracking.features as atf
    import silly_kicks.tracking.features as tf

    xt = _fitted_xt()
    assert tf.off_ball_run_value_xfns(xt, home_team_id=1)[0].__name__ == _FORBIDDEN_NAME
    assert atf.off_ball_run_value_xfns(xt, home_team_id=1)[0].__name__ == _FORBIDDEN_NAME


def test_no_run_value_xfns_in_any_default_list():
    for name, lst in _default_lists().items():
        for fn in lst:
            fn_name = getattr(fn, "__name__", str(fn))
            assert _FORBIDDEN_NAME not in fn_name, (
                f"{name} contains a TF-35 run-value xfn ({fn_name}) -- a0 result-leakage "
                f"(ADR-042). off_ball_run_value_xfns is opt-in and MUST NOT enter a "
                f"default/union xfn list feeding HybridVAEP."
            )


def test_factory_rejects_an_unfitted_xt_at_build_time():
    """Fail at list-construction, not deep inside a VAEP fit."""
    import pytest
    from sklearn.exceptions import NotFittedError

    import silly_kicks.tracking.features as tf

    with pytest.raises(NotFittedError):
        tf.off_ball_run_value_xfns(ExpectedThreat(), home_team_id=1)


def test_xfn_emits_the_four_numeric_columns_times_three_slots():
    """Coverage denominator excluded: 4 numeric columns, not 5."""
    import silly_kicks.tracking.features as tf

    xfn = tf.off_ball_run_value_xfns(_fitted_xt(), home_team_id=1)[0]
    states = [pd.DataFrame({"action_id": [1, 2]}, index=[0, 1]) for _ in range(3)]
    out = xfn(states, None)
    assert len(out.columns) == 12
    assert "n_valued_disruptive_runs_a0" not in out.columns, (
        "n_valued_disruptive_runs is a coverage denominator and must not be a VAEP feature"
    )
    for slot in range(3):
        for col in ("run_value_target", "n_disruptive_runs", "run_value_disruptive_sum", "run_value_enabled_pass"):
            assert f"{col}_a{slot}" in out.columns
