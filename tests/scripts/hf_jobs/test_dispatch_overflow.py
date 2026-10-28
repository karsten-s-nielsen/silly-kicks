"""T6 (spec §D): dispatch-concurrency overflow is bounded + stateless. A job whose container NEVER STARTED
and then timed out is a dispatch overflow (back off, do NOT count toward error_resumes); a STARTED job that
timed out is a running-timeout (counts, today's behaviour). The back-off floors at ``dispatch_floor`` and,
after ``dispatch_halt_after`` more never-started timeouts at the floor, halts + surfaces the real ceiling
(owner ruling B24). A capacity reject degrades; a non-capacity or ambiguous reject fails loud.
"""

from __future__ import annotations

import datetime as dt
import types
from typing import Any

from scripts.hf_jobs import campaign as C


def _dt(s):
    return dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=s)


def _job(*, status="ERROR", message=None, created_at: Any = 1, started_at: Any = "UNSET", finished_at: Any = None):
    ns = types.SimpleNamespace(
        id="j",
        labels={"run_key": "rk", "variant": "default", "stage": "fit"},
        status=types.SimpleNamespace(stage=status, message=message),
        created_at=created_at,
    )
    if started_at != "UNSET":  # "UNSET" models a JobInfo WITHOUT the attribute (B23 fail-safe)
        ns.started_at = started_at
    ns.finished_at = finished_at
    return ns


# ---- the never-started classifier (B23 fail-safe) ----


def test_timed_out_and_never_started_is_a_dispatch_overflow():
    assert C.job_never_started_timeout(_job(message="Job timeout", started_at=None)) is True


def test_timed_out_but_started_is_a_running_timeout_not_an_overflow():
    assert C.job_never_started_timeout(_job(message="Job timeout", started_at=_dt(10))) is False


def test_missing_started_at_attribute_is_treated_as_started():
    # B23: absence of the attribute must NOT be exempted -- fail safe to "started" (counts)
    assert C.job_never_started_timeout(_job(message="Job timeout", started_at="UNSET")) is False


def test_a_non_timeout_error_is_never_a_dispatch_overflow():
    assert C.job_never_started_timeout(_job(message="boom", started_at=None)) is False


# ---- the stateless back-off, floor and halt ----


def test_consecutive_never_started_counts_leading_overflows_only():
    jobs = [
        _job(created_at=3, message="Job timeout", started_at=None),
        _job(created_at=2, message="Job timeout", started_at=None),
        _job(created_at=1, status="COMPLETED"),  # a success ends the run
    ]
    assert C.consecutive_never_started(jobs) == 2


def test_a_started_timeout_breaks_the_never_started_run():
    jobs = [
        _job(created_at=2, message="Job timeout", started_at=None),
        _job(created_at=1, message="Job timeout", started_at=_dt(5)),  # started -> not an overflow
    ]
    assert C.consecutive_never_started(jobs) == 1


def test_effective_dispatch_cap_backs_off_to_the_floor():
    assert C.effective_dispatch_cap(dispatch_cap=25, dispatch_floor=20, never_started=0) == 25
    assert C.effective_dispatch_cap(dispatch_cap=25, dispatch_floor=20, never_started=3) == 22
    assert C.effective_dispatch_cap(dispatch_cap=25, dispatch_floor=20, never_started=99) == 20  # never below floor


def test_overflow_halts_after_m_never_started_timeouts_at_the_floor():
    # cap 25, floor 20 -> 5 back-offs reach the floor; m=2 more at the floor -> halt (B24)
    assert C.dispatch_overflow_halt(never_started=6, dispatch_cap=25, dispatch_floor=20, m=2) is False
    assert C.dispatch_overflow_halt(never_started=7, dispatch_cap=25, dispatch_floor=20, m=2) is True


def test_observed_max_concurrency_is_the_peak_overlap():
    jobs = [
        _job(started_at=_dt(0), finished_at=_dt(100)),
        _job(started_at=_dt(10), finished_at=_dt(50)),  # overlaps the first -> 2
        _job(started_at=_dt(200), finished_at=_dt(300)),  # disjoint -> back to 1
    ]
    assert C.observed_max_concurrency(jobs) == 2


# ---- the reject classifier (B26 g/h, consider-09) ----


def _http_exc(status):
    return types.SimpleNamespace(response=types.SimpleNamespace(status_code=status))


def test_capacity_reject_is_recognized():
    assert C._is_capacity_reject(RuntimeError("You have too many concurrent jobs running")) is True


def test_a_429_is_a_capacity_reject():
    exc = Exception("rejected")
    exc.response = types.SimpleNamespace(status_code=429)  # type: ignore[attr-defined]
    assert C._is_capacity_reject(exc) is True


def test_a_non_capacity_error_is_not_a_capacity_reject():
    assert C._is_capacity_reject(RuntimeError("image pull failed: manifest unknown")) is False


# ---- the error_resumes exemption (§D): a never-started timeout does not count ----


def test_trailing_errors_skips_never_started_timeouts_but_counts_started_ones():
    overflow = _job(created_at=2, message="Job timeout", started_at=None)  # exempt
    real = _job(created_at=1, message="boom", started_at=_dt(5))  # a started ERROR counts
    assert C._trailing_errors([overflow, real]) == 1  # the overflow is skipped, the real error counts
    started_timeout = _job(message="Job timeout", started_at=_dt(5))  # started -> running-timeout counts
    assert C._trailing_errors([started_timeout]) == 1
    assert C._trailing_errors([overflow, overflow]) == 0  # only overflows -> nothing counts


def test_a_quota_or_rate_limit_reject_fails_loud():
    # B48: a permanent quota/billing/rate-limit reject is NOT a transient capacity reject -> must fail loud
    # (degrading it would silent-loop: no slot ever frees, no counter, no halt)
    assert C._is_capacity_reject(RuntimeError("Monthly quota exceeded for your account")) is False
    assert C._is_capacity_reject(RuntimeError("rate limit exceeded")) is False
    assert C._is_capacity_reject(RuntimeError("resource limit reached")) is False


def test_consecutive_never_started_tolerates_a_none_created_at():
    # B49/B55: created_at is Optional (hf_hub 2.2.0). The jobs mix a DATETIME created_at with a None so the
    # old `created_at or 0` would compare a datetime against int 0 -> TypeError (discriminating: a reverted
    # `or 0` fails here, killing mutant M13). _created_epoch sorts None oldest with no crash.
    jobs = [
        _job(created_at=None, message="Job timeout", started_at=None),
        _job(created_at=_dt(30), message="Job timeout", started_at=None),
    ]
    assert C.consecutive_never_started(jobs) == 2  # both are overflows; no TypeError
