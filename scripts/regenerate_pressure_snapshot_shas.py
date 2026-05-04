"""Regenerate expected SHA-256 hashes for tests/tracking/test_pressure_snapshot.py.

Float-precision drift (e.g., numpy / pandas minor-version bumps, kernel-arithmetic
changes) requires re-pinning the snapshot SHAs. This script computes the live
hashes from the deterministic synthetic fixture used by the test, edits the
``EXPECTED_SHAS`` dict in the test file in-place, and prints a summary diff so
the operator can audit the change before committing.

Usage::

    uv run python scripts/regenerate_pressure_snapshot_shas.py

Mirrors the ``regenerate_*.py`` pattern from PR-S24
(per ``feedback_codegen_for_data_to_code_integrity``).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Lazy-import pandas via the test fixture builder + production kernel.
# The fixture lives in the test module so script + test share a single
# source of truth for inputs.
from silly_kicks.tracking.features import pressure_on_actor  # noqa: E402
from tests.tracking.test_pressure_snapshot import (  # noqa: E402
    _build_fixture,
    _hash_series,
)

SNAPSHOT_TEST = REPO_ROOT / "tests" / "tracking" / "test_pressure_snapshot.py"
METHODS = ("andrienko_oval", "link_zones", "bekkers_pi")


def _compute_live_shas() -> dict[str, str]:
    actions, frames = _build_fixture()
    out: dict[str, str] = {}
    for method in METHODS:
        result = pressure_on_actor(actions, frames, method=method)
        out[method] = _hash_series(result)
    return out


def _read_pinned_shas(text: str) -> dict[str, str]:
    pinned: dict[str, str] = {}
    for method in METHODS:
        match = re.search(rf'"{method}"\s*:\s*"([^"]+)"', text)
        if match is not None:
            pinned[method] = match.group(1)
    return pinned


def _patch_file(text: str, live: dict[str, str]) -> str:
    new_text = text
    for method, sha in live.items():
        pattern = rf'("{method}"\s*:\s*)"[^"]*"'
        new_text, n = re.subn(pattern, rf'\1"{sha}"', new_text)
        if n != 1:
            raise RuntimeError(f"Failed to patch '{method}' SHA in {SNAPSHOT_TEST}; matches={n}")
    return new_text


def main() -> int:
    text = SNAPSHOT_TEST.read_text(encoding="utf-8")
    pinned = _read_pinned_shas(text)
    live = _compute_live_shas()

    if pinned == live:
        print("All snapshot SHAs already match live computation. No changes.")
        for method in METHODS:
            print(f"  {method}: {live[method]}")
        return 0

    new_text = _patch_file(text, live)
    SNAPSHOT_TEST.write_text(new_text, encoding="utf-8")

    print("Snapshot SHAs regenerated:")
    for method in METHODS:
        old = pinned.get(method, "<unset>")
        new = live[method]
        marker = "->" if old != new else "  "
        print(f"  {method}: {old} {marker} {new}")
    print(f"\nWrote new SHAs into {SNAPSHOT_TEST.relative_to(REPO_ROOT)}")
    print("Verify the change is intentional, then commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
