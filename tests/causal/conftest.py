"""Put ``scripts/`` on sys.path for the causal e2e tests.

``tests/causal/test_causal_e2e.py`` and ``test_owner_run_refusals.py`` load driver scripts
in-process via importlib; those drivers use bare sibling imports (``from _input_contract import
...``, ``from _provenance import ...``) that resolve only with ``scripts/`` on ``sys.path`` (the
drivers are designed to run as ``python scripts/foo.py``, where ``sys.path[0]`` is ``scripts/``).

Previously this worked ONLY as a collection-order side effect of ``tests/scripts/conftest.py``
during a full-suite run, so the causal e2e tests failed when ``tests/causal/`` was collected in
isolation. Adding the path here makes this directory self-sufficient. Scoped to ``tests/causal/``
so the global config is untouched (mirrors ``tests/scripts/conftest.py``).
"""

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
