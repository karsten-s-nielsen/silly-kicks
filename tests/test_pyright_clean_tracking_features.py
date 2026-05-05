"""pyright clean gate per spec section 8.10."""

from __future__ import annotations

import shutil
import subprocess

import pytest


def _get_pyright_cmd() -> list[str] | None:
    """Return pyright command as list, preferring uv run, falling back to direct."""
    # Try uv run pyright first (works in uv-managed projects)
    if shutil.which("uv") is not None:
        result = subprocess.run(
            ["uv", "run", "pyright", "--version"],  # noqa: S607 -- static literal
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return ["uv", "run", "pyright"]
    # Fall back to direct pyright on PATH
    if shutil.which("pyright") is not None:
        return ["pyright"]
    return None


_PYRIGHT_CMD = _get_pyright_cmd()


@pytest.mark.skipif(_PYRIGHT_CMD is None, reason="pyright not available (neither uv run nor direct)")
def test_pyright_clean_tracking_namespace() -> None:
    assert _PYRIGHT_CMD is not None  # guarded by skipif
    result = subprocess.run(  # noqa: S603 -- cmd built from shutil.which + static paths
        [*_PYRIGHT_CMD, "silly_kicks/tracking/", "silly_kicks/atomic/tracking/"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "0 errors" in result.stdout, f"pyright failed:\n{result.stdout}\n{result.stderr}"
