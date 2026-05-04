"""pyright clean gate per spec section 8.10."""

from __future__ import annotations

import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("pyright") is None, reason="pyright not installed")
def test_pyright_clean_tracking_namespace() -> None:
    pyright = shutil.which("pyright")
    assert pyright is not None  # pragma: no cover -- guarded by skipif above
    result = subprocess.run(  # noqa: S603  -- args are static literals; binary resolved via shutil.which
        [pyright, "silly_kicks/tracking/", "silly_kicks/atomic/tracking/"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "0 errors" in result.stdout, f"pyright failed:\n{result.stdout}\n{result.stderr}"
