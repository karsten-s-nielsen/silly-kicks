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
    from scripts import regen_tf46_fixtures as R
    from scripts._loader_pining import _base_url, _resolve_token

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        token, base = _resolve_token(None), _base_url()
        files = R._regen_skillcorner(out, token, base) + R._regen_idsse(out, token, base)
        R._write_sha256sums(out, files)
        regenerated = _manifest((out / "SHA256SUMS").read_text(encoding="utf-8"))
        committed = _manifest((_DATASETS / "tf46_fixture_sha256sums.txt").read_text(encoding="utf-8"))
        assert regenerated == committed
