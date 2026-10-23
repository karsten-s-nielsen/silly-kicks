"""A package attribute named like one of its modules IS that module (B-16 class, TF-46 C2 r2).

When a package re-exports a function under the name of the module it lives in, ``package.name`` is the function
or the module depending on import order: ``mock.patch("package.name.attr")`` and ``import package.name as m``
then resolve to the wrong object. The convention instead is a public function in a PRIVATE module
(``tracking._kick_resolver.resolve_kicks``), re-exported by name. This gate enumerates every package under
``silly_kicks`` and every ``<pkg>/<name>.py`` / ``<pkg>/<name>/`` sub-module, so a new shadowing export fails CI.
"""

from __future__ import annotations

import importlib
import pkgutil
import types

import silly_kicks

#: Sub-packages whose import needs an optional extra; enumeration skips them when the extra is absent.
_OPTIONAL = ("silly_kicks.mcp",)

#: Shadowing that SHIPPED before this gate, each with its reason. Renaming a shipped public module path breaks its
#: consumers (Hyrum), so these stay; adding an entry is a visible review decision, never a convenience.
#: The list is CLOSED (owner confirmation 2026-10-22, C2 r3): it can only shrink -- a new shadow fails the invariant,
#: and a rename of these is a deliberate breaking change (a major version with a deprecation path), never a fix-up.
_SHIPPED_EXCEPTIONS = {
    "silly_kicks.vaep.features.actiontype": "shipped public submodule path + socceraction-compatible function",
    "silly_kicks.vaep.features.bodypart": "shipped public submodule path + socceraction-compatible function",
    "silly_kicks.vaep.features.result": "shipped public submodule path + socceraction-compatible function",
}


def _packages() -> list[types.ModuleType]:
    out = [silly_kicks]
    for info in pkgutil.walk_packages(silly_kicks.__path__, prefix="silly_kicks."):
        if not info.ispkg:
            continue
        try:
            out.append(importlib.import_module(info.name))
        except ImportError:
            if info.name.startswith(_OPTIONAL):
                continue
            raise
    return out


def test_no_package_attribute_shadows_its_submodule():
    shadowed = []
    for pkg in _packages():
        for info in pkgutil.iter_modules(pkg.__path__):
            full = f"{pkg.__name__}.{info.name}"
            try:
                importlib.import_module(full)
            except ImportError:
                continue  # an optional-extra module; its absence cannot shadow anything
            attr = getattr(pkg, info.name, None)
            if full in _SHIPPED_EXCEPTIONS:
                continue
            if attr is not None and not isinstance(attr, types.ModuleType):
                shadowed.append(f"{full} is shadowed by a {type(attr).__name__}")
    assert not shadowed, (
        "A package attribute shadows its own sub-module; move the function into a private module "
        f"(`_<name>.py`) and re-export it by name: {shadowed}"
    )


def test_the_gate_sees_the_packages_it_must_cover():
    # Anti-rot: the enumeration reaches the restart-stage packages (an empty walk would pass vacuously).
    names = {p.__name__ for p in _packages()}
    assert {"silly_kicks", "silly_kicks.spadl", "silly_kicks.tracking"} <= names


def test_every_shipped_exception_is_still_a_shadow():
    # Anti-rot: an exception that no longer shadows (renamed / fixed) must leave the list.
    for full in _SHIPPED_EXCEPTIONS:
        pkg_name, _, name = full.rpartition(".")
        attr = getattr(importlib.import_module(pkg_name), name, None)
        assert attr is not None and not isinstance(attr, types.ModuleType), f"{full} no longer shadows; drop it"
        assert callable(attr), f"{full}: the shadowing attribute is expected to be the public function"
        assert isinstance(importlib.import_module(full), types.ModuleType)  # the shipped submodule path still loads


def test_the_exception_list_is_closed():
    assert set(_SHIPPED_EXCEPTIONS) == {
        "silly_kicks.vaep.features.actiontype",
        "silly_kicks.vaep.features.bodypart",
        "silly_kicks.vaep.features.result",
    }
