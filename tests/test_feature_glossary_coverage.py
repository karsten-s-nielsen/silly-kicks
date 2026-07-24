"""Coverage gate: every emitted derived column has a FEATURE_GLOSSARY entry (glossary Task 11).

Completeness is inspection-driven: ``emitted_columns()`` (tests/invariants/glossary_emitted_columns.py)
is the base-normalised union of every default-config producer's output, so a NEW emitted column fails
``test_no_undocumented_columns`` until authored. The ``__all__``-less discovery fallback that keeps this
robust is proven live by ``tests/invariants/test_glossary_discovery.py::test_all_less_module_is_discovered``.

This module calls the run-and-diff harness, whose tracking leg sweeps add_obso/add_pausa/
add_space_creation on default (no injected xt) config, so it opts out of the SyntheticEPVWarning/
IgnoredSurfaceInputsWarning error-filter at module level (ADR-041), like the other auto-enumerating gates.
"""

import pytest

from silly_kicks.feature_glossary import FEATURE_GLOSSARY, emitting_module_is_importable
from tests.invariants.glossary_emitted_columns import emitted_columns

pytestmark = [
    pytest.mark.filterwarnings("ignore::silly_kicks.tracking.SyntheticEPVWarning"),
    pytest.mark.filterwarnings("ignore::silly_kicks.tracking.IgnoredSurfaceInputsWarning"),
]

_NON_CONFORMING_PRODUCERS = {  # recorded exception: vaep fs.* aren't add_*/*_xfns (enumerated by list-invocation)
    "silly_kicks.vaep.base.xfns_default",
    "silly_kicks.vaep.hybrid.hybrid_xfns_default",
}
# Metrics genuinely computed inline in features.py (NO separate _compute module) -- add CONSCIOUSLY, never lazily.
_FEATURES_HOMED_ALLOWLIST: set[str] = set()


def test_no_undocumented_columns():
    missing = emitted_columns() - set(FEATURE_GLOSSARY)
    assert not missing, f"emitted columns with no glossary entry: {sorted(missing)}"


def test_no_stale_entries():
    stale = set(FEATURE_GLOSSARY) - emitted_columns()
    assert not stale, f"glossary entries for non-emitted columns: {sorted(stale)}"


def test_emitting_module_importable_and_not_lazily_features():
    bad_import = [fc.name for fc in FEATURE_GLOSSARY.values() if not emitting_module_is_importable(fc.emitting_module)]
    assert not bad_import, f"non-importable emitting_module: {bad_import}"
    # Enforce the home-module convention (spec section 1.1): don't lazily point the catalogue at the
    # features.py monolith (importable but zero provenance). Genuinely-features-homed metrics go in the allowlist.
    lazy = [
        fc.name
        for fc in FEATURE_GLOSSARY.values()
        if fc.emitting_module.endswith(".features") and fc.name not in _FEATURES_HOMED_ALLOWLIST
    ]
    assert not lazy, (
        "emitting_module points at the features.py monolith (no provenance). Use the metric's home/compute "
        f"module (_packing/_obso/...), or add to _FEATURES_HOMED_ALLOWLIST if it has none: {lazy}"
    )
    # NOTE (honest): beyond importable + non-features, emitting_module is DOCUMENTATION, not gate-verified --
    # it can still name the WRONG home module and pass. The monolithic features.py (every producer's
    # __module__ == ...features) makes run-and-diff attribution impossible (spec section 1.1 / rev-2 review).


def test_name_shape_completeness_is_a_documented_limitation():
    # HONEST LIMITATION (not anti-rot): discovery finds producers only by the add_*/*_xfns NAME SHAPE
    # (tests/invariants/glossary_discovery.py). A public function emitting derived columns but named
    # otherwise AND not in the exception set is invisible to the gate -- detecting it needs running every
    # public function on a fixture (out of scope). This PINS the known exception set so KNOWN
    # non-conforming producers stay tracked; a NEW one is an accepted blind spot, documented here rather
    # than dressed up as a guard that catches nothing.
    assert _NON_CONFORMING_PRODUCERS
    for q in _NON_CONFORMING_PRODUCERS:
        assert q.count(".") >= 2, q  # real dotted qualnames, not typos
