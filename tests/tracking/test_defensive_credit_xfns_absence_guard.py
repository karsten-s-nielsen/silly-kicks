"""ADR-039/042: defensive credit gates on the action's own result + downstream shot outcome
(F4 result-leakage), so it ships NO xfns factory and MUST NOT appear in any default xfn list."""

import importlib

_MODULES = (
    "silly_kicks.tracking.features",
    "silly_kicks.atomic.tracking.features",
    "silly_kicks.vaep",
    "silly_kicks.vaep.base",
    "silly_kicks.atomic.vaep",
    "silly_kicks.atomic.vaep.base",
)
_FORBIDDEN = ("defensive_credit", "bravery")


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
    """Floor sanity -- the discovery finds SOME default lists (guard isn't vacuous)."""
    assert _default_lists(), "no default xfn lists discovered -- the absence guard would be vacuous"


def test_no_defensive_credit_transformer_in_any_default_list():
    for name, lst in _default_lists().items():
        for fn in lst:
            fn_name = getattr(fn, "__name__", str(fn))
            for forbidden in _FORBIDDEN:
                assert forbidden not in fn_name, (
                    f"{name} contains a TF-51 transformer ({fn_name}) -- defensive credit gates on "
                    f"result + downstream shot outcome (F4 leakage, ADR-039/042); it ships no xfns "
                    f"and MUST NOT enter a default/union xfn list feeding HybridVAEP."
                )


def test_no_defensive_credit_xfns_factory_exists():
    """TF-51 v1 ships NO xfns factory (spec section 4.1). This pins that decision (delete if v2 adds one)."""
    import silly_kicks.tracking as T

    assert not hasattr(T, "defensive_credit_xfns")
    assert not hasattr(T, "bravery_xfns")
    # TF-51 v2 Item 5: the pressure-commitment cue ships aggregator-only (spec section 6, T4).
    assert not hasattr(T, "press_commitment_xfns")
