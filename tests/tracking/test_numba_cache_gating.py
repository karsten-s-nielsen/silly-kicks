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

import ast
import importlib
import pathlib

import pytest

import silly_kicks

MODULES = [
    "silly_kicks.tracking._ball_carrier_numba",
    "silly_kicks.tracking.pitch_control._numba_kernels",
    # TF-58 coordination: the SampEn pair counters + the spec 7.9 surrogate-identity kernels (eager signatures,
    # so they decorate -- and would resolve a cache locator -- at `import silly_kicks.coordination`).
    "silly_kicks.coordination._kernels._numba",
    "silly_kicks.tracking._das_numba",
    "silly_kicks.tracking._ghost_gk_numba",
    # xT-GK turnover scan: a lazy `_njit(cache=True)` inside `except Exception` did not crash the import -- it
    # silently fell back to the ~100x slower pure-Python scan on a read-only install (fixed 2026-10-02).
    "silly_kicks.xtgk._turnover",
]

# (module, attribute) pairs for every @njit-decorated kernel.
KERNELS = [
    ("silly_kicks.tracking._ball_carrier_numba", "_carrier_loop_numba"),
    ("silly_kicks.tracking.pitch_control._numba_kernels", "tti_numba"),
    ("silly_kicks.tracking.pitch_control._numba_kernels", "influence_numba"),
    ("silly_kicks.tracking.pitch_control._numba_kernels", "gaussian_influence_numba"),
    ("silly_kicks.coordination._kernels._numba", "_pairs_within_sorted_nb"),
    ("silly_kicks.coordination._kernels._numba", "_dominance_count_self_nb"),
    ("silly_kicks.coordination._kernels._numba", "_dominance_count_cross_nb"),
    ("silly_kicks.coordination._kernels._numba", "near_in_phase_counts_at_nb"),
    ("silly_kicks.coordination._kernels._numba", "xcorr_edge_correction_nb"),
    ("silly_kicks.coordination._kernels._numba", "shifted_rho_group_means_nb"),
    ("silly_kicks.coordination._kernels._numba", "pearson_rows_nb"),
    ("silly_kicks.tracking._das_numba", "_approx_sigmoid"),
    ("silly_kicks.tracking._das_numba", "_one_frame"),
    ("silly_kicks.tracking._ghost_gk_numba", "_kde_numba_loop"),
    ("silly_kicks.tracking._ghost_gk_numba", "_leaf_values_numba"),
    ("silly_kicks.tracking._ghost_gk_numba", "_leaf_indices_numba"),
    ("silly_kicks.xtgk._turnover", "_opp_first_shot_scan_fast"),
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


# --------------------------------------------------------------------------- derived population (ADR-056)
#: Every numba decorator that builds a CACHED dispatcher (cache=True touches the filesystem at decoration time). Beyond
#: njit/jit: vectorize/guvectorize/cfunc do too (review B m15). `_njit` is the repo's own gated wrapper.
_NUMBA_JIT_BASE = frozenset({"njit", "_njit", "jit", "vectorize", "guvectorize", "cfunc"})


def _numba_names(tree: ast.AST) -> frozenset[str]:
    """The LOCAL names bound to a numba jit decorator in this module: the base names plus any
    ``from numba[...] import <base> as <alias>`` alias (review B m15 -- an aliased import must not escape the scan).
    ``import numba as x`` needs no entry: ``x.njit`` is caught by the attribute name."""
    names = set(_NUMBA_JIT_BASE)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "numba":
            names.update(a.asname or a.name for a in node.names if a.name in _NUMBA_JIT_BASE)
    return frozenset(names)


def _applies_njit(tree: ast.AST) -> bool:
    names = _numba_names(tree)
    for node in ast.walk(tree):
        func = (
            node.func if isinstance(node, ast.Call) else node if isinstance(node, (ast.Name, ast.Attribute)) else None
        )
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name in names and (isinstance(node, ast.Call) or _is_decorator(tree, node)):
            return True
    return False


def _is_decorator(tree: ast.AST, node: ast.AST) -> bool:
    return any(
        node in fn.decorator_list for fn in ast.walk(tree) if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def _literal_cache_true(tree: ast.AST) -> list[int]:
    """Line numbers of every ``njit(..., cache=True)`` / ``jit(..., cache=True)`` with a LITERAL True."""
    bad = []
    names = _numba_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name in names and any(
                kw.arg == "cache" and isinstance(kw.value, ast.Constant) and kw.value.value is True
                for kw in node.keywords
            ):
                bad.append(node.lineno)
    return bad


def _njit_modules(root: pathlib.Path, package: str) -> dict[str, ast.AST]:
    out = {}
    for py in sorted(root.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        if _applies_njit(tree):
            out[".".join((package, *py.relative_to(root).with_suffix("").parts))] = tree
    return out


def test_every_numba_module_is_enrolled():
    """The population is DERIVED from the source, never hand-kept: a new ``@njit`` module joins the gate or fails
    here (the coordination kernels and the xT-GK turnover scan were both missing from the hand-kept list)."""
    root = pathlib.Path(silly_kicks.__file__).parent
    derived = set(_njit_modules(root, "silly_kicks"))
    assert derived == set(MODULES), {"unenrolled": derived - set(MODULES), "stale": set(MODULES) - derived}
    assert {m for m, _a in KERNELS} == set(MODULES)  # every enrolled module has its kernels checked


def test_no_literal_cache_true_anywhere():
    root = pathlib.Path(silly_kicks.__file__).parent
    offenders = {
        m: lines for m, tree in _njit_modules(root, "silly_kicks").items() if (lines := _literal_cache_true(tree))
    }
    assert not offenders, f"cache must be the env-gated _NUMBA_CACHE flag, never a literal True: {offenders}"


def test_population_scan_catches_a_planted_module(tmp_path):
    """Planted violations: an unenrolled ``@njit`` module is found, and a literal ``cache=True`` is flagged."""
    planted = "from numba import njit\n\n@njit(cache=True)\ndef f(x):\n    return x\n"
    (tmp_path / "planted.py").write_text(planted, encoding="utf-8")
    plain = "def g(x):\n    return x  # njit is mentioned only in a comment\n"
    (tmp_path / "plain.py").write_text(plain, encoding="utf-8")
    found = _njit_modules(tmp_path, "pkg")
    assert set(found) == {"pkg.planted"}
    assert _literal_cache_true(found["pkg.planted"]) == [3]


@pytest.mark.parametrize(
    "src",
    [
        "from numba import njit as nb\n@nb(cache=True)\ndef f(): return 1\n",  # ALIASED import
        "from numba import vectorize\n@vectorize(cache=True)\ndef g(): return 1\n",  # another numba jit decorator
        "from numba import guvectorize\n@guvectorize(cache=True)\ndef h(): return 1\n",
        "from numba import cfunc\n@cfunc(cache=True)\ndef k(): return 1\n",
    ],
)
def test_detection_catches_aliases_and_other_numba_decorators(src):
    # B m15: a `from numba import njit as nb` alias, or vectorize/guvectorize/cfunc with cache=True, must NOT escape the
    # population scan or the literal-cache scan (all build a cached dispatcher). Population is complete today; this
    # hardens the detector against a future escapee.
    tree = ast.parse(src)
    assert _applies_njit(tree)  # the module is enrolled
    assert _literal_cache_true(tree)  # the literal cache=True is flagged
