"""CI guard: the committed ``.test_durations`` must COVER the sharded suite, per directory.

pytest-split balances the ``--splits`` shards by the committed ``.test_durations``; a test with no
recorded duration is assigned the MEAN of the recorded ones, so a whole BLOCK of new tests landing with
zero recorded durations is silently mis-weighted and piles into one shard. That is exactly the TF-58
(#270) incident: ``.test_durations`` had 9823 entries and ZERO for ``tests/coordination`` (706 tests),
``tests/mcp`` (34) or ``tests/datasets`` -- every one fell back to the ~0.22 s mean, pytest-split
underweighted coordination ~40x, and one shard ran ~40 min.

This gate fails loud when any ``tests/`` subdirectory that contributes ``not e2e and not slow`` tests is
entirely (or mostly) absent from ``.test_durations`` -- forcing a ``--store-durations`` regen (CI-measured
per ADR-074) when a new block lands, so the split cannot silently drift again. Collection only (no run).
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DURATIONS = _ROOT / ".test_durations"

#: A subdirectory below this fraction of its collected tests present in ``.test_durations`` is a
#: regen-overdue block. 0.5 is deliberately loose: it catches a wholly- or mostly-untimed new block
#: (the failure mode) without flaking on a handful of individually-new tests in an otherwise-timed dir.
_MIN_COVERAGE = 0.5


def _collect_output() -> subprocess.CompletedProcess[str]:
    """Collect the EXACT ``not e2e and not slow`` selection the sharded CI step shards (``--collect-only``).

    ``check`` is deliberately NOT used: a collection error (e.g. a new optional dep missing on one leg)
    exits non-zero, and ``check=True`` would surface it as an opaque ``CalledProcessError`` that reads
    like a harness bug rather than "collection broke". The caller inspects ``returncode`` and reports the
    two failure modes -- collection-broke vs coverage-short -- distinctly.
    """
    return subprocess.run(  # noqa: S603 -- sys.executable + a fixed literal argv, no untrusted input
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-m",
            "not e2e and not slow",
            "--collect-only",
            "-q",
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _subdir(nodeid: str) -> str:
    # tests/<subdir>/... -> "tests/<subdir>"; a file directly under tests/ -> "tests"
    parts = nodeid.split("::", 1)[0].split("/")
    return "/".join(parts[:2]) if len(parts) > 2 else "tests"


def test_test_durations_covers_every_sharded_subdirectory() -> None:
    assert _DURATIONS.is_file(), ".test_durations is missing (pytest-split has nothing to balance on)"
    recorded = set(json.loads(_DURATIONS.read_text(encoding="utf-8")))

    proc = _collect_output()
    collected = [ln.strip() for ln in proc.stdout.splitlines() if "::" in ln and ln.strip().startswith("tests/")]
    # Distinguish a broken collection from a short coverage: a non-zero exit with no collected node IDs
    # means pytest could not even collect (import/dep error on this leg), NOT that durations are stale.
    if proc.returncode != 0 and not collected:
        tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-20:])
        raise AssertionError(
            f"pytest --collect-only FAILED (exit {proc.returncode}) -- collection broke, not a coverage "
            f"problem (fix the import/dependency error below, do not regen durations):\n{tail}"
        )
    assert collected, "collected no not-e2e/not-slow tests (collection or marker expression is wrong)"

    by_dir_total: dict[str, int] = {}
    by_dir_hit: dict[str, int] = {}
    for nid in collected:
        d = _subdir(nid)
        by_dir_total[d] = by_dir_total.get(d, 0) + 1
        by_dir_hit[d] = by_dir_hit.get(d, 0) + (1 if nid in recorded else 0)

    under = {
        d: (by_dir_hit[d], by_dir_total[d]) for d in by_dir_total if by_dir_hit[d] / by_dir_total[d] < _MIN_COVERAGE
    }
    assert not under, (
        "`.test_durations` under-covers sharded test dirs (regen overdue -- re-run the CI-measured "
        "`durations-capture` per ADR-074 / docs/context/ci.md, download the artifact, commit it): "
        + ", ".join(f"{d} {h}/{t}" for d, (h, t) in sorted(under.items()))
    )
