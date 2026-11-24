"""W1: the public numba backend query (spec 2026-11-08 tracking-batch §4.1, plan Task 6)."""

from __future__ import annotations

import pytest

from silly_kicks import _numba_dispatch, backends


def _clear_switches(monkeypatch) -> None:
    for family in _numba_dispatch.FAMILIES:
        monkeypatch.delenv(f"SILLY_KICKS_{family}_FORCE_NUMPY", raising=False)


def _numba_missing(monkeypatch) -> None:
    def broken():
        raise _numba_dispatch.NumbaUnavailableError("numba cannot be imported (simulated)")

    monkeypatch.setattr(_numba_dispatch, "_import_numba", broken)


def test_reexports_are_the_dispatch_objects():
    assert backends.FAMILIES is _numba_dispatch.FAMILIES
    assert backends.NumbaUnavailableError is _numba_dispatch.NumbaUnavailableError
    assert len(backends.FAMILIES) >= 6  # non-vacuity: the #299 census


@pytest.mark.parametrize("family", _numba_dispatch.FAMILIES)
def test_active_backend_default_is_numba(family, monkeypatch):
    _clear_switches(monkeypatch)
    assert backends.active_backend(family) == "numba"


@pytest.mark.parametrize("family", _numba_dispatch.FAMILIES)
def test_active_backend_force_numpy(family, monkeypatch):
    _clear_switches(monkeypatch)
    monkeypatch.setenv(f"SILLY_KICKS_{family}_FORCE_NUMPY", "1")
    assert backends.active_backend(family) == "numpy"
    others = [f for f in _numba_dispatch.FAMILIES if f != family]
    assert all(backends.active_backend(f) == "numba" for f in others)  # one switch moves one family


@pytest.mark.parametrize("family", _numba_dispatch.FAMILIES)
def test_active_backend_force_numpy_even_without_numba(family, monkeypatch):
    _clear_switches(monkeypatch)
    _numba_missing(monkeypatch)
    monkeypatch.setenv(f"SILLY_KICKS_{family}_FORCE_NUMPY", "1")
    assert backends.active_backend(family) == "numpy"


def test_active_backend_unknown_family_raises(monkeypatch):
    _clear_switches(monkeypatch)
    with pytest.raises(ValueError, match="unknown numba kernel family"):
        backends.active_backend("NOT_A_FAMILY")


@pytest.mark.parametrize("family", _numba_dispatch.FAMILIES)
def test_active_backend_missing_numba_raises(family, monkeypatch):
    _clear_switches(monkeypatch)
    _numba_missing(monkeypatch)
    with pytest.raises(backends.NumbaUnavailableError):
        backends.active_backend(family)


def test_backend_report_lists_every_family_and_the_version(monkeypatch):
    _clear_switches(monkeypatch)
    report = backends.backend_report()
    assert set(report) == {*_numba_dispatch.FAMILIES, "numba_version"}
    assert all(report[f] == "numba" for f in _numba_dispatch.FAMILIES)
    import numba

    assert report["numba_version"] == numba.__version__


def test_backend_report_shows_a_forced_family(monkeypatch):
    _clear_switches(monkeypatch)
    family = _numba_dispatch.FAMILIES[0]
    monkeypatch.setenv(f"SILLY_KICKS_{family}_FORCE_NUMPY", "1")
    report = backends.backend_report()
    assert report[family] == "numpy"
    assert all(report[f] == "numba" for f in _numba_dispatch.FAMILIES[1:])


def test_backend_report_never_raises_when_numba_missing(monkeypatch):
    _clear_switches(monkeypatch)
    _numba_missing(monkeypatch)
    forced = _numba_dispatch.FAMILIES[-1]
    monkeypatch.setenv(f"SILLY_KICKS_{forced}_FORCE_NUMPY", "1")
    report = backends.backend_report()
    assert report[forced] == "numpy"  # a forced family needs no numba
    assert all(report[f] == "unavailable" for f in _numba_dispatch.FAMILIES if f != forced)
    assert report["numba_version"] == "unavailable"


@pytest.mark.parametrize("value", [None, "1", "0", "true"])
def test_report_and_dispatch_agree_on_every_switch_value(value, monkeypatch):
    """Review A-04: the report and the kernels read one force-numpy rule, so they never disagree."""
    _clear_switches(monkeypatch)
    if value is not None:
        for family in _numba_dispatch.FAMILIES:
            monkeypatch.setenv(f"SILLY_KICKS_{family}_FORCE_NUMPY", value)
    report = backends.backend_report()
    assert {f: report[f] for f in _numba_dispatch.FAMILIES} == {
        f: backends.active_backend(f) for f in _numba_dispatch.FAMILIES
    }


def test_report_reads_the_dispatch_rule(monkeypatch):
    """Review A-04: ``backend_report`` has no copy of the rule; it calls the dispatch's ``_forced_numpy``."""
    _clear_switches(monkeypatch)
    monkeypatch.setattr(_numba_dispatch, "_forced_numpy", lambda family: family == "DAS")
    report = backends.backend_report()
    assert report["DAS"] == "numpy" and backends.active_backend("DAS") == "numpy"
    assert report["PITCH_CONTROL"] == "numba" and backends.active_backend("PITCH_CONTROL") == "numba"


def test_backend_report_does_not_swallow_a_non_import_error(monkeypatch):
    """Only a missing numba is a report value; any other error (a numba cache-locator ``RuntimeError``) is a real
    failure and propagates, as in ``_numba_dispatch``."""
    _clear_switches(monkeypatch)

    def cache_failure():
        raise RuntimeError("cannot cache function: no locator available")

    monkeypatch.setattr(_numba_dispatch, "_import_numba", cache_failure)
    with pytest.raises(RuntimeError, match="no locator"):
        backends.backend_report()
