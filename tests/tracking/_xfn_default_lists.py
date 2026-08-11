"""The default xfn lists every leakage guard sweeps, discovered once.

Three guards carried three BYTE-IDENTICAL copies of this discovery rule, each pinned only by
`assert len(lists) >= 10` against a real population of 19. A floor cannot detect an omission -- and
an omission here means a NEW default list that no leakage guard sweeps, i.e. a leaky factory could
be opted into it with nothing looking. CLAUDE.md calls that a HybridVAEP-class correctness break.

Decision: Cycle B.
"""

from __future__ import annotations

import importlib

MODULES = (
    "silly_kicks.tracking.features",
    "silly_kicks.atomic.tracking.features",
    "silly_kicks.vaep",
    "silly_kicks.vaep.base",
    "silly_kicks.atomic.vaep",
    "silly_kicks.atomic.vaep.base",
)


def default_lists() -> dict[str, list]:
    """Every default xfn LIST reachable from `MODULES`, keyed `<module>.<attr>`."""
    found: dict[str, list] = {}
    for modname in MODULES:
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        for attr in dir(mod):
            if "default_xfns" in attr or attr.startswith(("xfns_default", "hybrid_xfns_default")):
                obj = getattr(mod, attr)
                if isinstance(obj, list):
                    found[f"{modname}.{attr}"] = obj
    return found


#: Asserted EXACTLY, both ways. A new default list must be registered here or CI fails; a removed
#: one cannot linger. This is the anti-rot property `>= 10` never had.
#:
#: `tracking.features.das_xfns` is deliberately absent and its absence is CORRECT: it is a factory
#: surface, not a default list, and TF-28 keeps it out of every default list by design.
SWEPT: frozenset[str] = frozenset(
    {
        "silly_kicks.atomic.tracking.features.atomic_actor_pre_window_default_xfns",
        "silly_kicks.atomic.tracking.features.atomic_pitch_control_default_xfns",
        "silly_kicks.atomic.tracking.features.atomic_pre_shot_gk_angle_default_xfns",
        "silly_kicks.atomic.tracking.features.atomic_pre_shot_gk_default_xfns",
        "silly_kicks.atomic.tracking.features.atomic_pre_shot_gk_full_default_xfns",
        "silly_kicks.atomic.tracking.features.atomic_pressure_default_xfns",
        "silly_kicks.atomic.tracking.features.atomic_tracking_default_xfns",
        "silly_kicks.atomic.vaep.base.xfns_default",
        "silly_kicks.tracking.features.actor_pre_window_default_xfns",
        "silly_kicks.tracking.features.pitch_control_default_xfns",
        "silly_kicks.tracking.features.pre_shot_gk_angle_default_xfns",
        "silly_kicks.tracking.features.pre_shot_gk_default_xfns",
        "silly_kicks.tracking.features.pre_shot_gk_full_default_xfns",
        "silly_kicks.tracking.features.pressure_default_xfns",
        "silly_kicks.tracking.features.tracking_default_xfns",
        "silly_kicks.vaep.base.xfns_default",
        "silly_kicks.vaep.base.xfns_default_no_goalscore",
        "silly_kicks.vaep.hybrid_xfns_default_no_goalscore",
        "silly_kicks.vaep.xfns_default_no_goalscore",
    }
)
