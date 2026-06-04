"""Tests for the time-base contract + low-coverage guard (spec 2026-06-04)."""

import warnings

import pandas as pd
import pytest

from silly_kicks.tracking import TimeBaseDiagnosis, validate_time_base
from silly_kicks.tracking.utils import (
    MISMATCH_OVERLAP_FLOOR,
    _diagnose_time_base,
    link_actions_to_frames,
)


def _frame_row(period_id, frame_id, t):
    return {
        "game_id": 1,
        "period_id": period_id,
        "frame_id": frame_id,
        "time_seconds": t,
        "frame_rate": 25.0,
        "player_id": 7,
        "team_id": 100,
        "is_ball": False,
        "is_goalkeeper": False,
        "x": 50.0,
        "y": 34.0,
        "z": float("nan"),
        "speed": 5.0,
        "speed_source": "native",
        "ball_state": "alive",
        "team_attacking_direction": "ltr",
        "confidence": None,
        "visibility": None,
        "source_provider": "gradientsports",
    }


def _action_row(action_id, period_id, t):
    return {
        "game_id": 1,
        "action_id": action_id,
        "period_id": period_id,
        "time_seconds": t,
        "team_id": 100,
        "player_id": 7,
        "type_id": 0,
        "result_id": 1,
        "bodypart_id": 0,
        "start_x": 50.0,
        "start_y": 34.0,
        "end_x": 60.0,
        "end_y": 34.0,
    }


def test_per_period_link_rate_populated():
    # p1: 2 actions both on a frame (linked). p2: 2 actions, both on frames (linked).
    # Healthy match -> no warning, no kwargs that don't exist yet.
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(1, 1, 1.0), _frame_row(2, 2, 0.0), _frame_row(2, 3, 1.0)])
    actions = pd.DataFrame(
        [_action_row(0, 1, 0.0), _action_row(1, 1, 1.0), _action_row(2, 2, 0.0), _action_row(3, 2, 1.0)]
    )
    _, report = link_actions_to_frames(actions, frames)
    assert report.per_period_link_rate == {1: 1.0, 2: 1.0}


def test_link_report_positional_construction_backcompat():
    """LOW: lock that the existing 7-positional-arg LinkReport construction still works
    (the new per_period_link_rate field is added LAST with a default). The empty-actions
    early return in link_actions_to_frames relies on this."""
    from silly_kicks.tracking.schema import LinkReport

    rpt = LinkReport(3, 2, 1, 0, {"gradientsports": 0.67}, 0.05, 0.2)  # 7 positional args
    assert rpt.per_period_link_rate == {}
    assert rpt.link_rate == 2 / 3


def test_diagnosis_flags_disjoint_period():
    # p1 conforming (overlap ~1); p2 actions absolute [2700,5835], frames relative [0,3142].
    frames = pd.DataFrame(
        [_frame_row(1, 0, 0.0), _frame_row(1, 1, 2823.0), _frame_row(2, 2, 0.0), _frame_row(2, 3, 3142.0)]
    )
    actions = pd.DataFrame(
        [_action_row(0, 1, 1.0), _action_row(1, 1, 2822.0), _action_row(2, 2, 2700.0), _action_row(3, 2, 5835.0)]
    )
    diag = _diagnose_time_base(actions, frames)
    assert diag.suspected_mismatch_periods == (2,)
    assert diag.per_period_overlap_fraction[2] == pytest.approx((3142.0 - 2700.0) / (5835.0 - 2700.0), rel=1e-6)
    assert diag.per_period_overlap_fraction[2] < MISMATCH_OVERLAP_FLOOR
    assert diag.per_period_overlap_fraction[1] >= MISMATCH_OVERLAP_FLOOR
    assert "period 2" in diag.message


def test_diagnosis_no_mismatch_when_overlapping():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(1, 1, 100.0)])
    actions = pd.DataFrame([_action_row(0, 1, 10.0), _action_row(1, 1, 90.0)])
    diag = _diagnose_time_base(actions, frames)
    assert diag.suspected_mismatch_periods == ()


def test_diagnosis_worst_first_ordering():
    # p2 fully disjoint (overlap 0), p3 partial-but-below-floor.
    frames = pd.DataFrame(
        [
            _frame_row(1, 0, 0.0),
            _frame_row(1, 1, 100.0),
            _frame_row(2, 2, 0.0),
            _frame_row(2, 3, 10.0),
            _frame_row(3, 4, 0.0),
            _frame_row(3, 5, 100.0),
        ]
    )
    actions = pd.DataFrame(
        [
            _action_row(0, 1, 10.0),
            _action_row(1, 1, 90.0),  # p1 fine
            _action_row(2, 2, 900.0),
            _action_row(3, 2, 1000.0),  # p2 fully disjoint
            _action_row(4, 3, 95.0),
            _action_row(5, 3, 600.0),
        ]  # p3 small overlap < floor
    )
    diag = _diagnose_time_base(actions, frames)
    assert diag.suspected_mismatch_periods[0] == 2  # worst (overlap 0) first
    assert set(diag.suspected_mismatch_periods) == {2, 3}


def test_diagnosis_single_action_degenerate_span():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(1, 1, 100.0)])
    in_range = pd.DataFrame([_action_row(0, 1, 50.0)])
    out_range = pd.DataFrame([_action_row(0, 1, 500.0)])
    assert _diagnose_time_base(in_range, frames).per_period_overlap_fraction[1] == 1.0
    assert _diagnose_time_base(out_range, frames).per_period_overlap_fraction[1] == 0.0


def test_diagnosis_nan_tolerant():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(1, 1, 100.0)])
    actions = pd.DataFrame([_action_row(0, 1, 10.0), _action_row(1, 1, float("nan"))])
    diag = _diagnose_time_base(actions, frames)  # must not raise
    assert diag.per_period_action_range[1] == (10.0, 10.0)


def _mismatch_inputs():
    frames = pd.DataFrame([_frame_row(2, 0, 0.0), _frame_row(2, 1, 3142.0)])
    actions = pd.DataFrame([_action_row(0, 2, 2700.0), _action_row(1, 2, 5835.0)])
    return actions, frames


def test_validate_time_base_raises_by_default():
    actions, frames = _mismatch_inputs()
    with pytest.raises(ValueError, match="time-base mismatch"):
        validate_time_base(actions, frames)


def test_validate_time_base_warn_returns_diagnosis():
    actions, frames = _mismatch_inputs()
    with pytest.warns(UserWarning, match="time-base mismatch"):
        diag = validate_time_base(actions, frames, on_mismatch="warn")
    assert isinstance(diag, TimeBaseDiagnosis)
    assert diag.suspected_mismatch_periods == (2,)


def test_validate_time_base_ignore_silent():
    actions, frames = _mismatch_inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail
        diag = validate_time_base(actions, frames, on_mismatch="ignore")
    assert diag.has_suspected_mismatch


def test_validate_time_base_clean_inputs_no_raise():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(1, 1, 100.0)])
    actions = pd.DataFrame([_action_row(0, 1, 10.0), _action_row(1, 1, 90.0)])
    diag = validate_time_base(actions, frames)  # default raise, but nothing to raise on
    assert not diag.has_suspected_mismatch


def test_low_coverage_warns_by_default():
    # p2 fully unlinked (actions far from p2 frame).
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(2, 1, 0.0)])
    actions = pd.DataFrame([_action_row(0, 1, 0.0), _action_row(1, 2, 5000.0)])
    with pytest.warns(UserWarning, match="period 2"):
        link_actions_to_frames(actions, frames)


def test_laundering_keystone_per_period_not_aggregate():
    """GS 10503 shape: aggregate > 0.5 but p2 < 0.5. Must fire on p2.

    p1: 100 actions all linked. p2: 100 actions, 19 linked. Aggregate = 59.5%
    (> 0.5) -- a match-aggregate guard would stay SILENT. Per-period p2 = 19%.
    This test fails if anyone reduces the guard to the match aggregate.
    """
    frame_rows = [_frame_row(1, i, float(i)) for i in range(100)]
    frame_rows += [_frame_row(2, 100 + i, float(i)) for i in range(100)]
    frames = pd.DataFrame(frame_rows)
    action_rows = [_action_row(i, 1, float(i)) for i in range(100)]  # p1 all on frames
    action_rows += [_action_row(100 + i, 2, float(i)) for i in range(19)]  # p2 first 19 on frames
    action_rows += [_action_row(200 + i, 2, 9000.0 + i) for i in range(81)]  # p2 other 81 far away
    actions = pd.DataFrame(action_rows)

    _, report = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    aggregate = report.n_actions_linked / report.n_actions_in
    assert aggregate > 0.5  # would launder under a match-agg guard
    assert report.per_period_link_rate[2] < 0.5

    with pytest.warns(UserWarning, match="period 2"):
        link_actions_to_frames(actions, frames)  # default warn fires on p2


def test_low_coverage_raise():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(2, 1, 0.0)])
    actions = pd.DataFrame([_action_row(0, 1, 0.0), _action_row(1, 2, 5000.0)])
    with pytest.raises(ValueError, match="period 2"):
        link_actions_to_frames(actions, frames, on_low_coverage="raise")


def test_low_coverage_ignore_silent_but_report_populated():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(2, 1, 0.0)])
    actions = pd.DataFrame([_action_row(0, 1, 0.0), _action_row(1, 2, 5000.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, report = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    assert report.per_period_link_rate[2] == 0.0


def test_healthy_match_silent():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(1, 1, 1.0), _frame_row(2, 2, 0.0)])
    actions = pd.DataFrame([_action_row(0, 1, 0.0), _action_row(1, 1, 1.0), _action_row(2, 2, 0.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails
        link_actions_to_frames(actions, frames)  # default warn, but healthy -> silent


def test_low_coverage_message_carries_time_base_hint():
    # p2 low coverage AND near-disjoint ranges -> message names the mismatch.
    frames = pd.DataFrame([_frame_row(2, 0, 0.0), _frame_row(2, 1, 3142.0)])
    actions = pd.DataFrame([_action_row(i, 2, 2700.0 + 50.0 * i) for i in range(50)])
    with pytest.warns(UserWarning, match="time-base mismatch"):
        link_actions_to_frames(actions, frames)


def test_sparsity_warns_without_mismatch_hint():
    # Tighten min_link_rate; ranges overlap (uniform sparsity) -> warn but NO mismatch claim.
    frames = pd.DataFrame([_frame_row(1, i, float(i)) for i in range(0, 100, 10)])  # every 10s
    actions = pd.DataFrame([_action_row(i, 1, float(i)) for i in range(100)])  # every 1s
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        link_actions_to_frames(actions, frames, min_link_rate=0.8, tolerance_seconds=0.2)
    msgs = [str(w.message) for w in caught]
    assert any("period 1" in m for m in msgs)
    assert all("time-base mismatch" not in m for m in msgs)


def test_per_period_link_rate_is_not_laundered_field():
    # p1 fully linked, p2 fully UNLINKED. Field reflects per-period truth (moved from Task 1).
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(2, 1, 0.0)])
    actions = pd.DataFrame([_action_row(0, 1, 0.0), _action_row(1, 2, 5000.0)])
    _, report = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    assert report.per_period_link_rate[1] == 1.0
    assert report.per_period_link_rate[2] == 0.0


def test_low_coverage_warning_blames_caller_not_linker_internals():
    """HIGH (lakehouse): stacklevel must point at THIS call site, not utils.py.

    The warn lives in the _enforce_link_coverage helper (one frame below
    link_actions_to_frames), so stacklevel must be 3. A value of 2 would set
    w.filename to silly_kicks/tracking/utils.py -- the exact 'into the linker
    internals' outcome the spec forbids. Message-only assertions cannot catch
    this; assert the warning's filename is this test module.
    """
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(2, 1, 0.0)])
    actions = pd.DataFrame([_action_row(0, 1, 0.0), _action_row(1, 2, 5000.0)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        link_actions_to_frames(actions, frames)  # default warn
    assert caught, "expected a low-coverage UserWarning"
    assert caught[0].filename == __file__, (
        f"warning blamed {caught[0].filename}, expected the caller ({__file__}); stacklevel is wrong"
    )


def test_guard_integration_fidelity_realistic_two_period_match():
    """LOW (lakehouse): one 'as e2e as possible' test exercising the full link +
    guard on a realistic 2-period fixture -- p1 healthy, p2 a period-relative-vs-
    absolute time-base mismatch (frames reset to 0, actions on the match clock)."""
    frames = pd.DataFrame(
        [_frame_row(1, i, float(i) * 30.0) for i in range(40)]  # p1 frames 0..1170s
        + [_frame_row(2, 40 + i, float(i) * 30.0) for i in range(40)]  # p2 frames RESET to 0..1170s
    )
    actions = pd.DataFrame(
        [_action_row(i, 1, float(i) * 30.0) for i in range(40)]  # p1 actions on frames
        + [_action_row(40 + i, 2, 1500.0 + float(i) * 30.0) for i in range(40)]  # p2 actions ABSOLUTE
    )
    _, report = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    assert report.per_period_link_rate[1] == pytest.approx(1.0)
    assert report.per_period_link_rate[2] < 0.2
    with pytest.warns(UserWarning, match="time-base mismatch"):
        link_actions_to_frames(actions, frames)
