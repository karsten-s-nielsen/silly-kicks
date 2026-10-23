"""Owner-gated e2e: the regen reproduces the committed TF-46 fixtures byte-for-byte (Task 2.1 Step 3).

``@pytest.mark.e2e`` + env-gated, so it is skipped in the default ``-m "not e2e"`` CI. It proves the
committed slim fixtures are reproducible from pining (the ``_generate_synthetic_match`` reproduction
principle, token-gated because the source is the pining network). Run on the owner/DGX with
``SILLY_KICKS_RUN_PINING_E2E=1``.
"""

import os
import tempfile
from pathlib import Path

import pytest

_DATASETS = Path(__file__).parent

pytestmark = pytest.mark.e2e


def _manifest(text: str) -> dict[str, str]:
    return {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in text.splitlines() if line.strip()}


@pytest.mark.skipif(
    not os.environ.get("SILLY_KICKS_RUN_PINING_E2E"),
    reason="pining network e2e; set SILLY_KICKS_RUN_PINING_E2E=1 (owner/DGX) to run",
)
def test_regen_reproduces_committed_fixtures():
    import subprocess
    import sys

    repo_root = _DATASETS.parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(  # noqa: S603  # fixed command (this interpreter + a repo script), no untrusted input
            [sys.executable, "scripts/regen_tf46_fixtures.py", "--out-dir", tmp],
            check=True,
            cwd=repo_root,
        )
        regenerated = _manifest((Path(tmp) / "SHA256SUMS").read_text(encoding="utf-8"))
        committed = _manifest((_DATASETS / "tf46_fixture_sha256sums.txt").read_text(encoding="utf-8"))
        assert regenerated == committed
