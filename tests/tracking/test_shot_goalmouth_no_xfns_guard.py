"""TF-48 leakage guard: post-contact outcome descriptors must NEVER enter VAEP
default xfn lists (HybridVAEP leakage class -- ADR-030, owner-decided 2026-06-10).

Auto-discovers every default xfn list (``*default_xfns*`` / ``xfns_default*``)
across the tracking, atomic-tracking, and VAEP surfaces, so a future default
list is covered without editing this guard.
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

_FORBIDDEN_SUBSTRINGS = ("shot_goalmouth", "shot_crossing", "shot_on_target", "shot_time_to_goal")


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
    lists = _default_lists()
    # floor sanity: the known surfaces must be present (guard is not vacuously green)
    assert any("tracking.features.tracking_default_xfns" in k for k in lists)
    assert any("vaep.base.xfns_default" in k for k in lists)
    assert len(lists) >= 10


def test_no_shot_goalmouth_in_any_default_xfn_list():
    for name, lst in _default_lists().items():
        for fn in lst:
            fn_name = getattr(fn, "__name__", str(fn))
            for bad in _FORBIDDEN_SUBSTRINGS:
                assert bad not in fn_name, (
                    f"{name} contains a TF-48 function ({fn_name}) -- post-shot outcome leakage (ADR-030)"
                )


def test_no_xfns_factory_exists():
    import silly_kicks.atomic.tracking.features as atf
    import silly_kicks.tracking.features as tf

    for mod in (tf, atf):
        assert not hasattr(mod, "shot_goalmouth_xfns"), (
            "TF-48 must not ship a VAEP xfns factory (ADR-030 leakage decision)"
        )
