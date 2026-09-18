"""Byte-identity gate for the batched packing kernel (perf hoist, ADR-039 no-duplicate).

``compute_packing_metrics_batch`` hoists the loop-invariant defender-extraction + back-line
selection + goal_map lookups ONCE and vectorizes only the cheap per-receiver counts; the scalar
``compute_packing_metrics`` DELEGATES to it at n=1. This gate proves the batch is byte-identical to
the scalar element-by-element across every branch (finite/non-finite receiver, home/away mirror,
no-back-line, empty defenders), so the ``gk_decision`` reconstruction consumer that switched to the
batch produces identical ``packing_made`` -- no metric change, no re-validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.tracking import compute_packing_metrics, compute_packing_metrics_batch
from silly_kicks.tracking._packing import GoalEndUnresolvedError
from tests.tracking._goal_map_helpers import goal_map_like_home_team_id
from tests.tracking.test_defensive_line import _make_frame_rows

_COLS = ("packing_made", "packing_net", "packing_goal_threat", "line_x")


def _frame(**kw):
    return _make_frame_rows(
        home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
        home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
        away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
        away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
        **kw,
    )


def _assert_batch_matches_scalar(frame, *, attacking_team_id, passer_xy, receivers):
    gm = goal_map_like_home_team_id(frame, 1)
    batch = compute_packing_metrics_batch(
        frame,
        attacking_team_id=attacking_team_id,
        goal_map=gm,
        passer_xy=passer_xy,
        receivers=np.asarray(receivers, dtype=float),
    )
    for i, rec in enumerate(receivers):
        scalar = compute_packing_metrics(
            frame,
            attacking_team_id=attacking_team_id,
            goal_map=gm,
            passer_xy=passer_xy,
            receiver_xy=(float(rec[0]), float(rec[1])),
        )
        for k in _COLS:
            b, s = batch[k][i], scalar[k]
            assert (np.isnan(b) and np.isnan(s)) or b == s, f"row {i} col {k}: batch {b!r} != scalar {s!r}"


def test_batch_matches_scalar_home_action_multiple_receivers():
    # HOME action (attacks x=105); several receivers spanning the defender x-interval.
    _assert_batch_matches_scalar(
        _frame(),
        attacking_team_id=1,
        passer_xy=(40.0, 34.0),
        receivers=[(70.0, 34.0), (55.0, 20.0), (90.0, 40.0), (45.0, 34.0)],
    )


def test_batch_matches_scalar_away_action_mirror():
    # AWAY action (attacks x=0) -> mirror branch (dx_ = 105 - x, bx = 105 - bx).
    _assert_batch_matches_scalar(
        _frame(),
        attacking_team_id=2,
        passer_xy=(65.0, 34.0),
        receivers=[(35.0, 34.0), (50.0, 20.0), (15.0, 40.0)],
    )


def test_batch_non_finite_receiver_is_nan_row_others_unaffected():
    # A non-finite receiver -> that row all-NaN; the finite rows are byte-identical to the scalar.
    _assert_batch_matches_scalar(
        _frame(),
        attacking_team_id=1,
        passer_xy=(40.0, 34.0),
        receivers=[(70.0, 34.0), (np.nan, 34.0), (55.0, 20.0), (60.0, np.nan)],
    )


def test_batch_non_finite_passer_all_nan():
    gm = goal_map_like_home_team_id(_frame(), 1)
    out = compute_packing_metrics_batch(
        _frame(), attacking_team_id=1, goal_map=gm, passer_xy=(np.nan, 34.0), receivers=[(70.0, 34.0), (55.0, 20.0)]
    )
    for k in _COLS:
        assert np.isnan(out[k]).all()


def test_batch_empty_frame_all_nan():
    gm = goal_map_like_home_team_id(_frame(), 1)
    empty = _frame().iloc[0:0]
    out = compute_packing_metrics_batch(
        empty, attacking_team_id=1, goal_map=gm, passer_xy=(40.0, 34.0), receivers=[(70.0, 34.0)]
    )
    for k in _COLS:
        assert np.isnan(out[k]).all()


def test_batch_unresolvable_goal_map_raises_once():
    # A non-finite receiver must NOT suppress the raise in the batch (batch contract: raises on an
    # unresolvable map given a non-empty frame + finite passer, regardless of receiver finiteness).
    # The scalar keeps its receiver-finiteness early guard, so the scalar path is unchanged.
    class _Unresolvable:
        def attacked_goal(self, *a, **k):
            return None

        def get(self, *a, **k):
            return None

    with pytest.raises(GoalEndUnresolvedError):
        compute_packing_metrics_batch(
            _frame(),
            attacking_team_id=1,
            goal_map=_Unresolvable(),  # type: ignore[arg-type]  # minimal duck-typed unresolvable map
            passer_xy=(40.0, 34.0),
            receivers=[(70.0, 34.0)],
        )
