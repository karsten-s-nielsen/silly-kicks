"""Capture the space-creation opponent-mirror golden (ADR-041, plan Task 0).

Writes ``tests/tracking/fixtures/space_creation_mirror_golden.npz`` with the
``add_space_creation`` output on the shared mirror fixture, plus the git commit it was
captured at.

MUST be run on the PRE-change tree (before the axis=1 -> axis=(0,1) mirror upgrade in
``silly_kicks/tracking/_space_creation.py``). The resulting golden is what
``tests/tracking/test_space_creation_mirror.py`` asserts against, making that test a real
before/after identity check rather than a self-comparison.

Usage::

    python scripts/gen_space_creation_mirror_golden.py

Re-running it AFTER the mirror change would overwrite the golden with post-change values
and silently neuter the gate -- only re-run deliberately (e.g. if the fixture changes),
and re-review the diff.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from silly_kicks.tracking import features  # noqa: E402
from tests.tracking._space_creation_mirror_fixture import build_mirror_fixture  # noqa: E402

_OUT = _REPO_ROOT / "tests" / "tracking" / "fixtures" / "space_creation_mirror_golden.npz"


def _source_commit() -> str:
    try:
        git = shutil.which("git")
        if git is None:  # pragma: no cover - dev convenience
            return "unknown"
        return subprocess.check_output([git, "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True).strip()  # noqa: S603
    except (subprocess.CalledProcessError, OSError):  # pragma: no cover - dev convenience
        return "unknown"


def main() -> int:
    actions, frames = build_mirror_fixture()
    with warnings.catch_warnings():
        # The synthetic-EPV notice lands in a later task of this same PR; the golden is
        # about VALUES, not warnings.
        warnings.simplefilter("ignore")
        out = features.add_space_creation(actions, frames, home_team_id=5)

    created = out["space_created_m2"].to_numpy(dtype=float)
    denied = out["space_denied_m2_opponent"].to_numpy(dtype=float)
    # Non-vacuity: an all-NaN OR all-zero golden makes the gate pass under any change
    # (0 == 0). Both are refused. The first draft of the fixture hit exactly this --
    # actors positioned behind the ball produced [0., 0.].
    n_finite = int(np.isfinite(created).sum())
    if n_finite == 0:
        raise SystemExit(
            "refusing to write an all-NaN golden: the fixture produced no space_created_m2 "
            "values, so the gate would pass vacuously"
        )
    if not np.any(np.abs(created[np.isfinite(created)]) > 0.0):
        raise SystemExit(
            "refusing to write an all-ZERO space_created_m2 golden: the gate would pass "
            "vacuously (0 == 0). Reposition the fixture's actors into a region where their "
            "leave-one-out actually moves OBSO (ahead of the ball, not behind it)."
        )

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        _OUT,
        space_created_m2=created,
        space_denied_m2_opponent=denied,
        source_commit=np.array(_source_commit()),
    )
    print(f"wrote {_OUT.relative_to(_REPO_ROOT)}")
    print(f"  rows={len(created)} finite_created={n_finite}")
    print(f"  created={created}")
    print(f"  denied ={denied}")
    print(f"  commit={_source_commit()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
