"""Capture the tracking-batch value snapshots (owner ruling 2026-11-23; see README.md in this folder).

Writes ``golden_<case>.parquet`` for every case in ``_builders.CASES``. Re-runnable and deterministic. Run it only
when a value change is approved, and say so in the CHANGELOG entry that ships the new goldens.

Run: python tests/regressions/tracking_batch/capture_goldens.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Standalone-script path setup (pytest sets these via pythonpath=[".","tests"]).
_TESTS = Path(__file__).resolve().parents[2]
for p in (str(_TESTS.parent), str(_TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from regressions.tracking_batch._builders import CASES  # noqa: E402

OUT = Path(__file__).resolve().parent


def main() -> None:
    for name, build in CASES.items():
        out = build()
        path = OUT / f"golden_{name}.parquet"
        out.to_parquet(path, index=False)
        print(f"wrote {path.name}: shape={out.shape}")


if __name__ == "__main__":
    main()
