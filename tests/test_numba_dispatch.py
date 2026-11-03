"""The single numba dispatch point (HFU spec §13.2, owner 2026-11-06): numba is a core dependency, a numpy
reference kernel runs only under an explicit ``SILLY_KICKS_<FAMILY>_FORCE_NUMPY=1`` switch, and a missing numba
raises instead of falling back silently."""

from __future__ import annotations

import sys

import pytest

from silly_kicks import _numba_dispatch as nd


@pytest.fixture(autouse=True)
def _fresh_probe(monkeypatch):
    """Each test sees an unprobed dispatch module and no force switch."""
    monkeypatch.setattr(nd, "_NUMBA", None)
    for fam in nd.FAMILIES:
        monkeypatch.delenv(f"SILLY_KICKS_{fam}_FORCE_NUMPY", raising=False)


def test_families_are_the_census():
    assert nd.FAMILIES == ("GHOST", "DAS", "COORDINATION", "PITCH_CONTROL", "BALL_CARRIER", "XTGK")


def test_numba_installed_means_numba():
    assert nd.use_numba("GHOST") is True


@pytest.mark.parametrize("family", ["GHOST", "DAS", "COORDINATION", "PITCH_CONTROL", "BALL_CARRIER", "XTGK"])
def test_force_switch_skips_numba(monkeypatch, family):
    monkeypatch.setenv(f"SILLY_KICKS_{family}_FORCE_NUMPY", "1")
    monkeypatch.setitem(sys.modules, "numba", None)  # an import would raise: the switch must not touch numba
    assert nd.use_numba(family) is False


def test_switch_only_honours_exact_one(monkeypatch):
    monkeypatch.setenv("SILLY_KICKS_DAS_FORCE_NUMPY", "true")
    assert nd.use_numba("DAS") is True


def test_missing_numba_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "numba", None)
    with pytest.raises(nd.NumbaUnavailableError) as info:
        nd.use_numba("GHOST")
    assert isinstance(info.value, ImportError)
    assert "pip install --force-reinstall silly-kicks" in str(info.value)


def test_unknown_family_raises():
    with pytest.raises(ValueError, match="NOPE"):
        nd.use_numba("NOPE")


def test_kernel_runtime_error_is_not_swallowed(monkeypatch):
    """Review Focus 1: a numba whose import raises RuntimeError (the cache-locator trap) propagates."""

    class _BoomFinder:
        def find_spec(self, name, path=None, target=None):
            if name == "numba":
                raise RuntimeError("cannot cache function: no locator available")
            return None

    monkeypatch.delitem(sys.modules, "numba", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BoomFinder(), *sys.meta_path])
    with pytest.raises(RuntimeError, match="no locator"):
        nd.use_numba("GHOST")


def test_require_numba_ignores_the_switch(monkeypatch):
    monkeypatch.setenv("SILLY_KICKS_DAS_FORCE_NUMPY", "1")
    nd.require_numba()  # numba installed: no error even with a switch set


def test_require_numba_raises_without_numba(monkeypatch):
    monkeypatch.setitem(sys.modules, "numba", None)
    with pytest.raises(nd.NumbaUnavailableError):
        nd.require_numba()


def test_numba_version_present():
    import numba

    assert nd.numba_version() == numba.__version__


def test_numba_version_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "numba", None)
    assert nd.numba_version() == "absent"
