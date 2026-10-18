import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["scripts.train_xshot_occurrence", "scripts.train_xcross_attempt"])
def test_data_dir_without_arms_fails_loud(module, tmp_path):
    # I4 (ADR-123 Amendment B): the extraction path REQUIRES --arms; it must fail loud, never fall back to
    # an implicit all-private (which silently shipped the wrong candidate). The check precedes the
    # clean-tree provenance gate, so this exits on --arms regardless of the working-tree state.
    (tmp_path / "g1").mkdir()
    r = subprocess.run(  # noqa: S603 -- fixed test argv (sys.executable + a trainer module), never user input
        [sys.executable, "-m", module, "--data-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "--arms is required" in (r.stdout + r.stderr)
