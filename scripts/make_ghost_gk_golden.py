"""Capture a compute_ghost_gk output golden for the TF-19 PR-3 refactor equivalence gate.

Run on the UNMODIFIED tree BEFORE Task 2's body extraction; the npz is the pre-refactor
oracle that proves ``_serve_positions_core`` did not move ``compute_ghost_gk``'s output.

WHY THIS EXISTS: no output golden existed anywhere for the ghost path. The five test
modules that call ``compute_ghost_gk`` assert structure and behaviour (columns added, LTR
required, two-GK handling), not values, and ``test_weights_bundle_golden.py`` only
import-checks ``GhostGkModel``. Extracting a 79-line body without this gate would let a
numeric shift ship green.

SCOPE: this is a SAME-ENVIRONMENT oracle. It is captured and compared on one machine
within one cycle, which is all the equivalence gate needs. It is deliberately NOT a
cross-environment artifact -- the bundled ghost weights were fit under a different sklearn
than a given runtime may carry (serving is a sklearn-free numpy reconstruction per
ADR-016, so that is provenance noise, not a correctness issue). Do not repurpose this npz
as a cross-platform pin.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from silly_kicks.tracking import compute_ghost_gk
from tests.tracking.test_ghost_gk import _fitted_model, _make_multi_frame_fixture

OUT = pathlib.Path(__file__).resolve().parent.parent / "tests" / "tracking" / "data" / "ghost_gk_refactor_golden.npz"


def main() -> None:
    frames = _make_multi_frame_fixture(n_frames=5)
    out = compute_ghost_gk(frames, model=_fitted_model()[0], home_team_id=1)
    gk = out[out["is_goalkeeper"].astype(bool) & ~out["is_ball"].astype(bool)]
    gk = gk.sort_values(["game_id", "period_id", "frame_id", "team_id"])

    if len(gk) == 0:
        raise SystemExit("REFUSING to write a vacuous golden: 0 GK rows. Fix the fixture first.")

    # Provenance only -- never load-bearing, so a git failure must not lose the golden.
    # Mirrors scripts/train_ghost_gk.py:147 (same call, same noqa, same try/except).
    try:
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # noqa: S607
    except Exception:
        source_commit = "unknown"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        ghost_gk_x=gk["ghost_gk_x"].to_numpy(dtype=float),
        ghost_gk_y=gk["ghost_gk_y"].to_numpy(dtype=float),
        ghost_gk_density_spread=gk["ghost_gk_density_spread"].to_numpy(dtype=float),
        source_commit=np.array(source_commit),
    )
    print(f"wrote golden for {len(gk)} GK rows -> {OUT}")


if __name__ == "__main__":
    main()
