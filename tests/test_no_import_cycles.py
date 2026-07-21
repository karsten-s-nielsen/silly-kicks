"""Every public subpackage must import STANDALONE, in a fresh interpreter (ADR-041).

`silly_kicks` has a genuine import cycle waiting to be closed:

    xthreat/_grid -> spadl.config -> spadl/__init__ -> tracking.direction
                  -> tracking/__init__ -> tracking.feature_framework
                  -> vaep.feature_framework -> vaep/__init__ -> vaep.features
                  -> ... -> back into xthreat

so ANY module-level ``from silly_kicks.xthreat import ...`` inside ``tracking`` or ``vaep``
closes the loop and breaks ``import silly_kicks.xthreat``. The sanctioned workaround is a
function-local import (the ``tracking/_xt_gk.py`` idiom).

This bit twice while wiring ADR-041 -- once in ``vaep/features/expected_threat.py`` and
again in ``tracking/_player_influence.py`` -- and BOTH times the ordinary test suite stayed
green, because pytest had already imported the packages in a benign order by the time the
affected test ran. Only a FRESH interpreter importing the subpackage FIRST exposes it,
which is what this gate does.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

_PACKAGES = (
    "silly_kicks.reflection",
    "silly_kicks.spadl",
    "silly_kicks.tracking",
    "silly_kicks.vaep",
    "silly_kicks.xthreat",
    "silly_kicks.xtgk",
    "silly_kicks.atomic.spadl",
    "silly_kicks.atomic.tracking",
    "silly_kicks.atomic.vaep",
)


@pytest.mark.parametrize("package", _PACKAGES)
def test_package_imports_standalone(package: str) -> None:
    """A fresh interpreter importing ONLY this package must succeed.

    Deliberately a subprocess: importing in-process would reuse ``sys.modules`` from
    whatever the test session already loaded, which is exactly the masking that let two
    real cycles through.
    """
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", f"import {package}"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, (
        f"`import {package}` failed in a fresh interpreter -- likely a module-level import "
        f"that closes the xthreat/spadl/tracking/vaep cycle; use a function-local import "
        f"instead.\n{proc.stderr.strip()[-1500:]}"
    )


def test_meta_gate_covers_the_public_subpackages() -> None:
    """Non-vacuity: the list must not silently shrink to nothing meaningful."""
    assert len(_PACKAGES) >= 8
    assert "silly_kicks.xthreat" in _PACKAGES  # the package both real cycles broke
