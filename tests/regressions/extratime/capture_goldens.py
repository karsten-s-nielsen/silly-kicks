"""Capture RT-only converter goldens against the CURRENT (pre-4.0.0) behaviour.

RT-only output is unaffected by the ET guard (it never fires without ET periods),
so these pin "no regression". MUST be captured BEFORE any converter edit.
Re-runnable + deterministic. Committed in the single branch commit (kept for regen).

Run: python tests/regressions/extratime/capture_goldens.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Standalone-script path setup (pytest sets these via pythonpath=[".","tests"]).
_TESTS = Path(__file__).resolve().parents[1]
for p in (str(_TESTS.parent), str(_TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from regressions.extratime._builders import CASES, run_converter  # noqa: E402

OUT = Path(__file__).resolve().parent


def main() -> None:
    for case in CASES:
        out = run_converter(case, et=False, flag=None)  # RT-only: no ET, flag irrelevant
        path = OUT / f"golden_{case}_rt.parquet"
        out.to_parquet(path)
        print(f"wrote {path.name}: shape={out.shape}")


if __name__ == "__main__":
    main()
