"""Put ``scripts/`` on sys.path for this directory.

pyproject sets ``pythonpath = [".", "tests"]``; the script modules (``_corpus``, ``_paired``,
``_cache``, ``_loader_pining``) are not importable without this. Scoped to ``tests/scripts/`` so
the global config is untouched.
"""

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
