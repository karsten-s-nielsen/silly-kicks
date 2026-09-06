"""compute_territorial_dominance -- ``completed_failed`` BYTE-IDENTITY golden regression (Task 10).

Freezes v1 (``method="completed_failed"``) output on a fixed multi-game/multi-player fixture so later
tasks (T11/T13/T14, which touch the territory area) cannot silently drift the untouched v1 path. Also
proves the new ``method``/``completion_model`` kwargs are inert on the default leg, and that the 5
counterfactual-only columns are absent from v1 output / present under ``method="counterfactual"``
(SPEC-04 sec 5.3) -- the schema-exclusion companion to ``tests/territory/test_columns_by_method.py``
(which pins the *schema dict*; this pins the *real compute output*).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from silly_kicks.territory import TERRITORY_COLUMNS, compute_territorial_dominance

from .test_compute import _actions, _def, _pass, _toy_xt
from .test_counterfactual_compute import _ToyUniformXt

_GOLDEN = Path(__file__).parent / "_golden" / "completed_failed_v1.parquet"

_CF_ONLY_COLUMNS = [
    "territory_xt_prevented_above_expectation",
    "territory_passes_aimed_into_hull",
    "territory_expected_threat_faced",
    "territory_mean_completion_faced",
    "territory_target_source",
]


class _ConstCompletion:
    """A trivial, never-fitted completion model -- constant c=0.6 -- for the schema-exclusion probe.

    silly-kicks ships no pass-completion model (port pattern, ADR-022 idiom); this stands in only to
    exercise the ``method="counterfactual"`` schema, not to make any claim about real completion rates.
    """

    def predict_completion(self, ox, oy, tx, ty):
        return np.full(np.asarray(tx, dtype=float).shape, 0.6)


def _fixture() -> pd.DataFrame:
    """A fixed, deterministic multi-game/multi-player scene -- the golden's source of truth.

    Reuses the exact row builders from ``test_compute.py`` (``_def``/``_pass``/``_actions``) so the
    golden is built from the same primitives every other territory test already trusts. Touches:
    conceded + prevented + an out-of-hull ignored pass + a forward pass (game 1, player 1); a degenerate
    hull, dropped-and-counted (game 1, player 2); and a resolved hull with zero opponent passes ->
    NaN rate (game 2, player 1).
    """
    rows = [
        # game 1, player 1, team 10: rectangle hull (trims to a triangle at the default trim_fraction=0.70).
        _def(1, 1, 10, 5, 20),
        _def(1, 1, 10, 15, 20),
        _def(1, 1, 10, 15, 48),
        _def(1, 1, 10, 5, 48),
        _pass(1, 20, 80, 40, 95, 40, completed=True),  # reflected (10,28) -> conceded
        _pass(1, 20, 80, 30, 98, 30, completed=False),  # reflected (7,38)  -> prevented
        _pass(1, 20, 40, 40, 50, 40, completed=True),  # reflected (55,28) -> far out -> ignored
        _pass(1, 20, 70, 40, 95, 40, completed=True),  # forward (end_x 95 > start_x 70), into hull
        # game 1, player 2, team 10: only 2 defensive actions -> degenerate hull, dropped-and-counted.
        _def(1, 2, 10, 5, 20),
        _def(1, 2, 10, 15, 20),
        # game 2, player 1, team 10: hull present, zero opponent passes -> NaN rate.
        _def(2, 1, 10, 5, 20),
        _def(2, 1, 10, 15, 20),
        _def(2, 1, 10, 15, 48),
        _def(2, 1, 10, 5, 48),
    ]
    return _actions(rows)


_FIXTURE_ACTIONS = _fixture()
_FITTED_XT = _toy_xt(0.1)


def _read_golden() -> pd.DataFrame:
    """Read the committed golden and restore the canonical id-column dtype.

    Parquet (pyarrow) cannot round-trip an ``object`` column holding homogeneous plain ints -- it infers
    ``int64`` on write and comes back ``int64``, not ``object``. Re-applying ``TERRITORY_COLUMNS`` (the
    single-sourced output schema ``compute_territorial_dominance`` itself casts to) repairs that
    storage-format artifact without masking any real value drift: every non-id column, and every id
    *value*, is still compared byte-for-byte below.
    """
    return pd.read_parquet(_GOLDEN).astype(dict(TERRITORY_COLUMNS))


def test_completed_failed_matches_the_pre_change_golden():
    out, _ = compute_territorial_dominance(_FIXTURE_ACTIONS, xt=_FITTED_XT)  # default method
    golden = _read_golden()
    assert list(golden.columns) == list(TERRITORY_COLUMNS)  # exactly the v1 15 columns
    assert_frame_equal(out, golden)  # v1 did not drift

    # New kwargs are inert on the default leg: explicit completed_failed + completion_model=None.
    out2, _ = compute_territorial_dominance(
        _FIXTURE_ACTIONS, xt=_FITTED_XT, method="completed_failed", completion_model=None
    )
    assert_frame_equal(out2, golden)
    assert_frame_equal(out2, out)


def test_completed_failed_output_excludes_counterfactual_only_columns():
    out, _ = compute_territorial_dominance(_FIXTURE_ACTIONS, xt=_FITTED_XT)
    for col in _CF_ONLY_COLUMNS:
        assert col not in out.columns


def test_counterfactual_output_includes_the_five_cf_only_columns():
    # completed_failed's _toy_xt has no `.transition_matrix` (unfitted otherwise); the counterfactual
    # path needs one (`destination_profiles`), so this leg uses `_ToyUniformXt` (a real, if uniform,
    # transition matrix) rather than the golden's `_FITTED_XT`.
    out, _ = compute_territorial_dominance(
        _FIXTURE_ACTIONS,
        xt=_ToyUniformXt(),  # type: ignore[arg-type]  -- duck-typed fitted xT (ADR-022)
        method="counterfactual",
        completion_model=_ConstCompletion(),  # type: ignore[arg-type]  -- duck-typed completion model
    )
    for col in _CF_ONLY_COLUMNS:
        assert col in out.columns
    # the v1 columns are still present alongside the cf-only ones.
    for col in TERRITORY_COLUMNS:
        assert col in out.columns
