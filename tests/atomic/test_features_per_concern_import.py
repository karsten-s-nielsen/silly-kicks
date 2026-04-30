"""Locks the A9-partial-closure pattern: silly_kicks.atomic.vaep.features
imports per-concern from vaep.features submodules, NOT from the package root.

If a future PR consolidates atomic's imports back to the package root (a
monolith-coupling regression), this test fails fast.
"""

from __future__ import annotations

import inspect

import silly_kicks.atomic.vaep.features as atomic_features


def test_atomic_imports_per_concern_not_from_monolith() -> None:
    """Atomic VAEP features imports per-concern submodules, not the package root."""
    source = inspect.getsource(atomic_features)
    forbidden_pattern = "from silly_kicks.vaep.features import"
    assert forbidden_pattern not in source, (
        f"silly_kicks.atomic.vaep.features should import from per-concern "
        f"submodules (e.g. 'from silly_kicks.vaep.features.core import ...'), "
        f"not from the package root '{forbidden_pattern}'."
    )

    # Specifically verify the 4 expected per-concern imports are present.
    expected_modules = (
        "silly_kicks.vaep.features.core",
        "silly_kicks.vaep.features.bodypart",
        "silly_kicks.vaep.features.context",
        "silly_kicks.vaep.features.temporal",
    )
    for mod_path in expected_modules:
        expected_line = f"from {mod_path} import"
        assert expected_line in source, (
            f"silly_kicks.atomic.vaep.features should import from {mod_path} "
            f"(expected line containing '{expected_line}')"
        )
