"""The release size check (ADR-132; corpus widening spec 5.6, owner ruling r4).

PyPI caps each uploaded file at 100 MB by default. ``publish.yml``'s ``build`` job runs this after ``python -m build``
and before the upload: it prints the size of every file in ``dist/`` and fails when ``dist/`` lacks a wheel or an
sdist, or when either is ``LIMIT_BYTES`` or larger (20 MB headroom below the cap).

Usage: ``python scripts/check_dist_sizes.py dist``
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

#: 80 MB, decimal (plan ruling R2): below PyPI's default 100 MB per-file cap either way.
LIMIT_BYTES = 80_000_000


def check(dist: Path, limit: int = LIMIT_BYTES) -> list[str]:
    """The problems with ``dist`` (empty == pass). Prints each file's size."""
    dist = Path(dist)
    if not dist.is_dir():
        return [f"{dist}: no such directory"]
    files = sorted(p for p in dist.iterdir() if p.is_file())
    problems = []
    for p in files:
        size = p.stat().st_size
        print(f"{p.name}: {size} bytes ({size / 1_000_000:.2f} MB)")
        if size >= limit:
            problems.append(f"{p.name}: {size} bytes is at or above the {limit}-byte release limit")
    if not any(p.suffix == ".whl" for p in files):
        problems.append(f"{dist}: no wheel (*.whl)")
    if not any(p.name.endswith(".tar.gz") for p in files):
        problems.append(f"{dist}: no sdist (*.tar.gz)")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="The release size check (ADR-132): every dist/ file under the limit.")
    ap.add_argument("dist", type=Path, help="the build output directory (dist/)")
    ap.add_argument("--limit", type=int, default=LIMIT_BYTES, help="per-file limit in bytes (default 80_000_000)")
    args = ap.parse_args(argv)
    problems = check(args.dist, args.limit)
    for problem in problems:
        print(f"::error::{problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
