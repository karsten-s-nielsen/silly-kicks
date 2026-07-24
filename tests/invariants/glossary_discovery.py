"""Discover public derived-column producers (``add_*`` / ``*_xfns``) by inspection, ``__all__``-less-safe.

Mirrors ``tests/invariants/conftest_id_scalar.py::_public_names`` (surface = ``__all__`` if declared,
else public callables defined in-module). The ``__all__``-less fallback is defensive Chesterton's-fence
robustness; the load-bearing dependency for THIS surface is the ``add_*``/``*_xfns`` NAME SHAPE. vaep's
``fs.*`` feature functions are the recorded exception (enumerated by the default-list run-and-diff leg
in ``glossary_emitted_columns``, not here).
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

PACKAGES = [
    "silly_kicks.tracking",
    "silly_kicks.atomic.tracking",
    "silly_kicks.spadl",
    "silly_kicks.atomic.spadl",
]  # vaep handled by the list-invocation leg (fs.* aren't name-shape producers)


def _is_producer_name(name: str) -> bool:
    return name.startswith("add_") or name.endswith("_xfns")


# NOTE: deliberate COPY of tests/invariants/conftest_id_scalar.py::_public_names (a conftest is awkward
# to import cross-directory). Keep the two in sync -- both encode the __all__-else-vars rule.
def _public_names(mod) -> list[str]:
    declared = getattr(mod, "__all__", None)
    if declared:
        return list(declared)
    return [
        n
        for n, o in vars(mod).items()
        if not n.startswith("_")
        and (inspect.isfunction(o) or inspect.isclass(o))
        and getattr(o, "__module__", None) == mod.__name__
    ]


# Modules that legitimately require an optional extra to import (accessible-space, xgboost, ...).
# Populate from the ACTUAL failures the first run surfaces -- do NOT pre-guess; an over-broad allowlist
# re-hides the coverage hole. A failing module drops ALL its columns from the gate, so keep this tight.
_OPTIONAL_IMPORT_MODULES: set[str] = set()
_import_failures: dict[str, str] = {}


def _iter_modules(pkg_name):
    pkg = importlib.import_module(pkg_name)
    yield pkg
    if hasattr(pkg, "__path__"):
        for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            try:
                yield importlib.import_module(info.name)
            except Exception as exc:  # RECORD, don't silently drop -- a failing module drops its columns
                _import_failures[info.name] = repr(exc)


def unexpected_import_failures() -> dict[str, str]:
    """Import failures NOT explained by a recorded optional extra -- each silently drops a module's columns."""
    return {m: e for m, e in _import_failures.items() if m not in _OPTIONAL_IMPORT_MODULES}


def discover_public_column_producers(*, extra_modules=None) -> dict[str, str]:
    """``{defining module.qualname: function-name}`` for public ``add_*``/``*_xfns`` across PACKAGES."""
    found: dict[str, str] = {}
    mods = [m for pkg in PACKAGES for m in _iter_modules(pkg)]
    if extra_modules:
        mods += list(extra_modules)
    for mod in mods:
        for name in _public_names(mod):
            obj = getattr(mod, name, None)
            if not inspect.isfunction(obj) or not _is_producer_name(name):
                continue
            found[f"{obj.__module__}.{obj.__qualname__}"] = name
    return found
