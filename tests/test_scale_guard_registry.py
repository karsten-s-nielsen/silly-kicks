"""Meta-assertions for the scale-guard registry (ADR-073, spec 4.3): the forcing function.

(1) every group_rows caller is registered (a new caller with no guard fails CI);
(2) every registry entry resolves to a real test (self-burning-down);
(3) degenerate-by-design entries carry a discriminating companion.

Non-degeneracy for ordinary entries is enforced by the harness `work_floor` at TEST time (a vacuous
guard's own test fails), so no meta-test re-runs measures.
"""

import importlib

from tests._scale_guarded import DEGENERATE_OK, SCALE_GUARD_MODULE, SCALE_GUARDED, group_rows_callers


def _has_test(testname: str) -> bool:
    mod = importlib.import_module(SCALE_GUARD_MODULE)
    return hasattr(mod, testname)


def test_registry_is_superset_of_group_rows_callers():
    missing = group_rows_callers() - set(SCALE_GUARDED)
    assert not missing, f"group_rows callers with no scale guard: {sorted(missing)}"


def test_registry_entries_resolve_to_collected_tests():
    for qual, testname in SCALE_GUARDED.items():
        assert _has_test(testname), f"stale registry entry: {qual} -> {SCALE_GUARD_MODULE}::{testname}"


def test_degenerate_entries_carry_a_discriminating_companion():
    for qual, companion in DEGENERATE_OK.items():
        assert qual in SCALE_GUARDED, f"{qual} in DEGENERATE_OK but not registered"
        assert _has_test(companion), f"{qual} is degenerate_ok but companion {companion} is missing"
