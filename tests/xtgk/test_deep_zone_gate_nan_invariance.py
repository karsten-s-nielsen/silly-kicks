"""T4 -- the FIT path is invariant to NaN-coord rows, which is why the 4.42.0 deep-zone gate
verdict does NOT need re-running under the resolved-origin fix (ADR-036 amendment, non-goal #1).

If this ever fails, the gate MUST be re-run.

THREE fixture faults were found by EXECUTION in review. Do not "simplify" them back:
  1. Shots must sit in the ATTACKING half with margin -- the ADR-028 orientation guard counts a
     NaN-start_x shot as own-half (NaN > 52.5 is False), so a naive fixture makes fit(contaminated)
     raise "only 43% of shots are in the attacking half" and the escalation clause misfires on an
     artifact.
  2. EVERY shot must carry an xg -- drawing xg independently of type_id gave 0/18 shots a reward, so
     all three surfaces were identically ZERO and the assertion was allclose(0, 0). A vacuous test
     cited as THE evidence for non-goal #1 is worse than no test. The meta-assertion guards it.
  3. ONE shared PressureLevels -- refitting per leg moves the tercile cutpoints and flips rows'
     terciles, so the surfaces differ for a reason unrelated to the property under test.
"""

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xtgk import MarkovPossessionValue, PressureLevels

PASS = spadlconfig.actiontype_id["pass"]
SHOT = spadlconfig.actiontype_id["shot"]
GOALKICK = spadlconfig.actiontype_id["goalkick"]
SUCCESS = spadlconfig.result_id["success"]


def _cohort(n=240, seed=0):
    rng = np.random.default_rng(seed)
    is_shot = rng.random(n) < 0.12
    # FIXTURE FAULT 1: shots MUST be in the attacking half with margin.
    start_x = np.where(is_shot, rng.uniform(70.0, 100.0, n), rng.uniform(0.0, 105.0, n))
    return pd.DataFrame(
        {
            "game_id": rng.integers(1, 5, n),
            "period_id": np.ones(n, dtype=int),
            "action_id": np.arange(n),
            "time_seconds": np.arange(n, dtype=float),
            "team_id": rng.integers(1, 3, n),
            "player_id": rng.integers(1, 23, n),
            "type_id": np.where(is_shot, SHOT, PASS),
            "result_id": np.full(n, SUCCESS),
            "possession_id": rng.integers(1, 40, n),
            "start_x": start_x,
            "start_y": rng.uniform(0, 68, n),
            "end_x": rng.uniform(0, 105, n),
            "end_y": rng.uniform(0, 68, n),
            "pressure": rng.uniform(0, 1, n),
            # FIXTURE FAULT 2: EVERY shot carries an xg, else every surface is identically zero.
            "xg": np.where(is_shot, rng.uniform(0.05, 0.4, n), np.nan),
        }
    )


def _nan_rows(n=60, seed=1):
    """The real defect's shape: goal-kicks / passes whose ORIGIN is NaN."""
    rng = np.random.default_rng(seed)
    bad = _cohort(n, seed=seed)
    bad["type_id"] = np.where(rng.random(n) < 0.5, GOALKICK, PASS)
    bad["xg"] = np.nan
    bad["start_x"] = np.nan
    bad["start_y"] = np.nan
    bad["action_id"] = np.arange(1000, 1000 + n)
    return bad


def test_fitted_surfaces_and_support_are_invariant_to_added_nan_coord_rows():
    clean = _cohort()
    contaminated = pd.concat([clean, _nan_rows()], ignore_index=True)

    # FIXTURE FAULT 3: ONE shared PressureLevels. The property under test is NaN-row-drop
    # invariance of the FIT SEAMS, not quantile stability under a changed row count.
    pl = PressureLevels().fit(clean["pressure"])

    def _fit(actions):
        return MarkovPossessionValue().fit(actions, xg_column="xg", pressure_column="pressure", pressure_levels=pl)

    v_clean = _fit(clean)
    v_contaminated = _fit(contaminated)

    # META-ASSERTION (non-vacuity): an all-zero surface would make the invariance check below
    # allclose(0, 0) -- it would PASS while proving nothing. That exact defect shipped in the first
    # draft of this test. Guard it.
    for p in (1, 2, 3):
        surface = v_clean.surface(p)  # type: ignore[arg-type]
        assert (surface != 0).sum() > 0, (
            f"pressure tercile {p}: the fitted V surface is ALL ZERO, so the invariance assertion "
            "below would be vacuous. Fix the fixture (shots need xg); do not weaken the test."
        )

    for p in (1, 2, 3):
        np.testing.assert_allclose(
            v_clean.surface(p),  # type: ignore[arg-type]
            v_contaminated.surface(p),  # type: ignore[arg-type]
            atol=1e-12,
            err_msg=(
                f"pressure tercile {p}: the fitted V surface MOVED when NaN-coord rows were added. "
                "The deep-zone gate would then be contaminated and MUST be re-run -- ADR-036 "
                "non-goal #1 no longer holds."
            ),
        )
        np.testing.assert_array_equal(
            v_clean.support(p),  # type: ignore[arg-type]
            v_contaminated.support(p),  # type: ignore[arg-type]
            err_msg=f"pressure tercile {p}: the fitted support counts moved (spec: surfaces AND support).",
        )
