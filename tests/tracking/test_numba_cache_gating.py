"""Regression: numba's on-disk cache (``cache=True``) must not be unconditionally
enabled, or import hard-fails on read-only / ephemeral install paths.

Background
----------
``@njit(cache=True)`` makes numba persist compiled code to disk, which requires a
writable cache *locator* to be resolved AT DECORATION TIME (module import). On
read-only / ephemeral installs (e.g. Databricks serverless: wheel on a read-only
ephemeral NFS path with no writable ``__pycache__`` beside the source and no
writable user-wide cache dir) all locators fail and numba raises ``RuntimeError``
*from inside a successful import* — taking down all of ``silly_kicks.tracking``,
not just the cached function. The existing ``try/except ImportError`` guards in the
consumer modules do NOT catch this (the exception is ``RuntimeError``, not
``ImportError``).

The fix gates ``cache`` on a module-level ``_NUMBA_CACHE`` flag, default OFF, opt-in
via ``SILLY_KICKS_NUMBA_CACHE=1`` OR numba's own ``NUMBA_CACHE_DIR``. With the
default (no env vars) ``cache=False`` → numba never resolves a locator → import is
safe everywhere. ``cache=False`` keeps full native JIT speed; it only drops
cross-process cache persistence (a one-time per-process recompile).

numba is a ``[test]`` dependency, so these always run in CI.
"""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "silly_kicks.tracking._ball_carrier_numba",
    "silly_kicks.tracking.pitch_control._numba_kernels",
]

# (module, attribute) pairs for every @njit-decorated kernel.
KERNELS = [
    ("silly_kicks.tracking._ball_carrier_numba", "_carrier_loop_numba"),
    ("silly_kicks.tracking.pitch_control._numba_kernels", "tti_numba"),
    ("silly_kicks.tracking.pitch_control._numba_kernels", "influence_numba"),
    ("silly_kicks.tracking.pitch_control._numba_kernels", "gaussian_influence_numba"),
]


def _reload(mod_name: str):
    return importlib.reload(importlib.import_module(mod_name))


@pytest.fixture(autouse=True)
def _restore_modules():
    """Reload both modules under the (restored) ambient env after each test so
    monkeypatched env state does not leak module-level globals into other tests."""
    yield
    for mod_name in MODULES:
        _reload(mod_name)


def _clear_cache_env(monkeypatch):
    monkeypatch.delenv("SILLY_KICKS_NUMBA_CACHE", raising=False)
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)


@pytest.mark.parametrize("mod_name", MODULES)
def test_default_disables_cache(monkeypatch, mod_name):
    """No env vars → cache OFF. This is the load-bearing import-safety guarantee:
    with cache disabled numba never resolves a writable locator at decoration."""
    _clear_cache_env(monkeypatch)
    mod = _reload(mod_name)
    assert mod._NUMBA_CACHE is False


@pytest.mark.parametrize("mod_name", MODULES)
def test_silly_kicks_var_enables_cache(monkeypatch, mod_name):
    """SILLY_KICKS_NUMBA_CACHE=1 opts back in (stable env / local dev)."""
    _clear_cache_env(monkeypatch)
    monkeypatch.setenv("SILLY_KICKS_NUMBA_CACHE", "1")
    mod = _reload(mod_name)
    assert mod._NUMBA_CACHE is True


@pytest.mark.parametrize("mod_name", MODULES)
def test_silly_kicks_var_falsey_keeps_cache_off(monkeypatch, mod_name):
    """Only the literal "1" enables — "0"/anything else stays off."""
    _clear_cache_env(monkeypatch)
    monkeypatch.setenv("SILLY_KICKS_NUMBA_CACHE", "0")
    mod = _reload(mod_name)
    assert mod._NUMBA_CACHE is False


@pytest.mark.parametrize("mod_name", MODULES)
def test_numba_cache_dir_enables_cache(monkeypatch, tmp_path, mod_name):
    """Setting numba's own NUMBA_CACHE_DIR (a writable path) opts back in, so the
    lakehouse gets caching just by pointing it at a writable local-disk dir."""
    _clear_cache_env(monkeypatch)
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path))
    mod = _reload(mod_name)
    assert mod._NUMBA_CACHE is True


@pytest.mark.parametrize(("mod_name", "attr"), KERNELS)
def test_decorated_kernel_reflects_disabled_cache(monkeypatch, mod_name, attr):
    """Every @njit kernel actually carries cache=False under the default env —
    proving the flag reaches the decoration, not just the module global.

    numba resolves a writable cache *locator* inside ``enable_caching()`` at
    decoration time when ``cache=True`` (that locator failure is the original
    import-time crash); with ``cache=False`` the dispatcher keeps the default
    ``NullCache`` and never touches the filesystem.
    """
    _clear_cache_env(monkeypatch)
    mod = _reload(mod_name)
    dispatcher = getattr(mod, attr)
    assert type(dispatcher._cache).__name__ == "NullCache"
