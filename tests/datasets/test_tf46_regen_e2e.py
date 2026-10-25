"""Owner-gated e2e: the regen reproduces the committed TF-46 fixtures (Task 2.1 Step 3).

``@pytest.mark.e2e`` + env-gated, so it is skipped in the default ``-m "not e2e"`` CI. It proves the
committed slim fixtures are reproducible from pining (the ``_generate_synthetic_match`` reproduction
principle, token-gated because the source is the pining network). Run on the owner/DGX with
``SILLY_KICKS_RUN_PINING_E2E=1``.

CSV / JSON fixtures must reproduce byte-for-byte. A parquet file embeds its writer version (``created_by``,
e.g. "parquet-cpp-arrow version 24.0.0"), so its bytes change with the pyarrow version even when its data does
not (measured 2026-10-25: the DGX pyarrow 26.0.0 rewrote the 8 committed parquet fixtures with equal content). A
parquet fixture must therefore reproduce its CONTENT exactly -- the arrow schema (fields and types, writer
metadata ignored) and every value -- not its bytes (owner ruling 2026-10-25, option A).
"""

import os
import tempfile
from pathlib import Path

import pytest

_DATASETS = Path(__file__).parent

pytestmark = pytest.mark.e2e


def _manifest(text: str) -> dict[str, str]:
    return {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in text.splitlines() if line.strip()}


def _parquet_content_mismatch(regenerated: Path, committed: Path) -> str | None:
    """``None`` when the two parquet files hold the same schema and values; else the reason."""
    import pandas as pd
    import pyarrow.parquet as pq

    if not pq.read_schema(regenerated).equals(pq.read_schema(committed), check_metadata=False):
        return "arrow schema differs"
    try:
        pd.testing.assert_frame_equal(pd.read_parquet(regenerated), pd.read_parquet(committed), check_exact=True)
    except AssertionError as err:
        return f"values differ: {str(err)[:200]}"
    return None


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
        assert sorted(regenerated) == sorted(committed)  # the same fixture set
        mismatched = {}
        for rel, digest in sorted(committed.items()):
            if regenerated[rel] == digest:
                continue  # byte-identical (every CSV / JSON must be)
            if not rel.endswith(".parquet"):
                mismatched[rel] = "bytes differ"
                continue
            reason = _parquet_content_mismatch(Path(tmp) / rel, _DATASETS / rel)
            if reason is not None:
                mismatched[rel] = reason
        assert not mismatched, mismatched
