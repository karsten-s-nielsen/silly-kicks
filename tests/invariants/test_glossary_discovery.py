import sys
import types

from tests.invariants import glossary_discovery as G


def test_finds_known_producers():
    prods = G.discover_public_column_producers()
    names = {q.rsplit(".", 1)[1] for q in prods}
    assert "add_obso" in names and "obso_xfns" in names and "add_packing" in names


def test_no_unexpected_import_failures():
    # A module that fails to import in CI silently drops ALL its columns -> coverage hole. Surface it.
    G.discover_public_column_producers()  # walks all packages, populating _import_failures
    bad = G.unexpected_import_failures()
    assert not bad, f"modules failing to import (columns silently dropped), not in the optional-extra allowlist: {bad}"


def test_all_less_module_is_discovered(monkeypatch):
    # META, red-first: the __all__-less fallback must actually DISCOVER an add_*-named producer in a
    # module that declares no __all__.
    mod = types.ModuleType("silly_kicks.tracking._planted_glossary_probe")

    def add_planted(actions, frames):  # add_* name shape; module has no __all__
        return actions

    add_planted.__module__ = mod.__name__
    add_planted.__qualname__ = "add_planted"  # module-level function (not the test's <locals> nesting)
    setattr(mod, "add_planted", add_planted)
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    found = G.discover_public_column_producers(extra_modules=[mod])
    assert "silly_kicks.tracking._planted_glossary_probe.add_planted" in found
