"""TF-49 leakage guard: every packing column gates on the action's OWN ``result_id``,
so as an a0-slot feature ``packing_xfns`` is the HybridVAEP result-leakage class
(ADR-039 F4, owner-decided). Unlike shot_goalmouth (irreducible post-contact leakage ->
no factory at all), the packing factory legitimately EXISTS -- it is a useful opt-in
feature for the a1/a2 (past-action) slots -- but its produced transformer must NEVER
enter a default/union xfn list, or the a0 completion-leak silently re-enters HybridVAEP.

This is the executable enforcement the F4 decision assumed: it mirrors the
shot_goalmouth no-xfns guard's auto-discovery (so a future default list is covered
without editing this file) and the xt_xfns opt-in guard's "not in any default list"
contract -- the two established precedents for leakage-class and opt-in factories.
"""

import importlib

_MODULES = (
    "silly_kicks.tracking.features",
    "silly_kicks.atomic.tracking.features",
    "silly_kicks.vaep",
    "silly_kicks.vaep.base",
    "silly_kicks.atomic.vaep",
    "silly_kicks.atomic.vaep.base",
)

# The transformer produced by ``packing_xfns`` sets ``__name__ == "packing"`` (std +
# atomic). The substring guard keys on that; test_packing_transformer_name pins it so
# a rename can't silently make the guard vacuous (discriminating-power discipline).
_FORBIDDEN_NAME = "packing"


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
    """Floor sanity: the known default surfaces are present (guard not vacuously green)."""
    lists = _default_lists()
    assert any("tracking.features.tracking_default_xfns" in k for k in lists)
    assert any("vaep.base.xfns_default" in k for k in lists)
    assert len(lists) >= 10


def test_packing_transformer_name():
    """Discriminating-power anchor: the substring guard below only protects while the
    produced transformer is actually named ``packing``. Pin it on both surfaces so a
    rename fails HERE (loudly) rather than silently neutering the leakage guard."""
    import silly_kicks.atomic.tracking.features as atf
    import silly_kicks.tracking.features as tf

    assert tf.packing_xfns(home_team_id=1)[0].__name__ == _FORBIDDEN_NAME
    assert atf.packing_xfns(home_team_id=1)[0].__name__ == _FORBIDDEN_NAME


def test_no_packing_xfns_in_any_default_list():
    """packing_xfns is opt-in (ADR-039 F4): its a0 slot gates on the action's own
    result_id, so it must never enter a default/union xfn list feeding HybridVAEP."""
    for name, lst in _default_lists().items():
        for fn in lst:
            fn_name = getattr(fn, "__name__", str(fn))
            assert _FORBIDDEN_NAME not in fn_name, (
                f"{name} contains a packing xfn ({fn_name}) -- a0 result-leakage (ADR-039 F4). "
                f"packing_xfns is opt-in and MUST NOT enter a default/union xfn list; use the "
                f"recorded result-free-a0 fork if a default consumer is genuinely needed."
            )
