"""compute_ghost_gk output must not move when its body is extracted (TF-19 PR-3 Task 2).

This is the ONLY output-value gate on the ghost path. The behavioural suites
(test_ghost_gk, _serve_mean, _frame_restriction, _r3, _integration) assert structure --
columns added, LTR required, two-GK handling -- and test_weights_bundle_golden only
import-checks GhostGkModel. Without this gate, extracting the ~79-line shared body into
_serve_positions_core could ship a numeric shift green.

The golden is captured by scripts/make_ghost_gk_golden.py on the PRE-refactor tree and is
a SAME-ENVIRONMENT oracle (see that script's docstring) -- captured and compared on one
machine within one cycle, which is exactly what an equivalence gate needs.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from silly_kicks.tracking import compute_ghost_gk
from tests.tracking.test_ghost_gk import _fitted_model, _make_multi_frame_fixture

GOLDEN = pathlib.Path(__file__).parent / "data" / "ghost_gk_refactor_golden.npz"


def test_compute_ghost_gk_output_matches_the_pre_refactor_golden():
    if not GOLDEN.exists():  # pragma: no cover - operator error, not a code path
        pytest.fail(
            f"missing golden {GOLDEN.name}: run scripts/make_ghost_gk_golden.py on the "
            "PRE-refactor tree. Capturing it after the refactor would certify the "
            "refactor against itself."
        )
    ref = np.load(GOLDEN, allow_pickle=False)

    frames = _make_multi_frame_fixture(n_frames=5)
    out = compute_ghost_gk(frames, model=_fitted_model()[0], home_team_id=1)
    gk = out[out["is_goalkeeper"].astype(bool) & ~out["is_ball"].astype(bool)]
    gk = gk.sort_values(["game_id", "period_id", "frame_id", "team_id"])

    assert len(gk) == len(ref["ghost_gk_x"]) > 0, "golden is vacuous or the row count moved"
    for col in ("ghost_gk_x", "ghost_gk_y", "ghost_gk_density_spread"):
        np.testing.assert_allclose(
            gk[col].to_numpy(dtype=float),
            ref[col],
            rtol=1e-9,
            atol=0.0,
            err_msg=f"{col} moved: the body extraction was NOT behaviour-preserving",
        )
