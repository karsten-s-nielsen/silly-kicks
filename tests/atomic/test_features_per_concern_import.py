"""Atomic-coupling lock: silly_kicks.atomic.vaep.features imports framework
from the dedicated cross-package module (silly_kicks.vaep.feature_framework)
and per-concern feature reuse from the appropriate submodules.

Codifies the A9-closure shape (silly-kicks 2.4.0). If a future PR consolidates
atomic's imports back to the package root or reaches into vaep.features.core
for framework primitives, this test fails fast.
"""

from __future__ import annotations

import inspect

import silly_kicks.atomic.vaep.features as atomic_features


def test_atomic_imports_framework_and_per_concern() -> None:
    """Atomic VAEP features imports framework from feature_framework + per-concern features."""
    source = inspect.getsource(atomic_features)

    # Forbid: package-root re-export reach (would mask which submodules atomic
    # actually depends on, eroding the per-concern boundary).
    assert "from silly_kicks.vaep.features import" not in source, (
        "atomic.vaep.features must import per-concern, not from package root."
    )

    # Forbid: reaching into vaep.features.core for framework primitives
    # (post-2.4.0 those live in silly_kicks.vaep.feature_framework).
    assert "from silly_kicks.vaep.features.core import" not in source, (
        "atomic.vaep.features must import framework primitives from "
        "silly_kicks.vaep.feature_framework, not from vaep.features.core."
    )

    # Forbid: the now-deleted `_actiontype` symbol.
    assert "_actiontype" not in source, (
        "atomic.vaep.features must use actiontype_categorical (public) — "
        "the private _actiontype was promoted in silly-kicks 2.4.0."
    )

    # Require: framework module import + 3 per-concern submodule imports.
    expected_lines = (
        "from silly_kicks.vaep.feature_framework import",
        "from silly_kicks.vaep.features.bodypart import",
        "from silly_kicks.vaep.features.context import",
        "from silly_kicks.vaep.features.temporal import",
    )
    for line in expected_lines:
        assert line in source, f"atomic.vaep.features missing required import line: {line!r}"
