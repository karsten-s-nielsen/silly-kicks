"""A migrated corpus-seam module must import QUALIFIED (``from scripts._xxx``), not bare (``from _xxx``).

Bare module-level imports resolve ONLY when ``tests/scripts/conftest.py`` has put ``scripts/`` on
``sys.path`` -- a scope confined to tests/scripts/. A consumer outside that scope (the calibrate_*
drivers' tests in tests/calibration/, an owner running a driver, any future importer) dies at import
with ``ModuleNotFoundError: No module named '_loader_pining'``. This masked itself once because a
combined ``pytest tests/scripts/ tests/calibration/`` run collects tests/scripts/conftest.py during
the collection phase, leaking scripts/ onto sys.path for the whole session.

This guard imports each seam module in a FRESH subprocess whose only extra sys.path entry is the repo
root -- the exact context that broke -- so a bare seam import regresses loudly regardless of pytest
collection order. It is deliberately robust to that leak (a plain in-process ``import`` here would
pass under the leak and defeat the purpose).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

#: The corpus-load-seam modules imported as ``scripts.<name>`` from outside tests/scripts/.
_SEAM_MODULES = (
    "scripts._item_outcome",
    "scripts._driver",
    "scripts._loader_pining",
    "scripts._sb_open_data",
    "scripts._events_admission",
    "scripts._xt_corpus",
    "scripts.build_skillcorner_s1_event_validity",
)


def test_seam_modules_import_with_only_the_repo_root_on_syspath():
    code = "\n".join(f"import {m}" for m in _SEAM_MODULES)
    proc = subprocess.run(  # noqa: S603 -- sys.executable + a fixed literal, no untrusted input
        [sys.executable, "-c", code],
        cwd=str(_ROOT),  # sys.path gets the repo root (scripts is a package); scripts/ is NOT added
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        "a corpus-seam module has a BARE module-level import (`from _xxx import ...`) that resolves "
        "only under tests/scripts/conftest.py's scripts/-on-sys.path scope. Qualify it to "
        f"`from scripts._xxx import ...`.\nstderr:\n{proc.stderr}"
    )
