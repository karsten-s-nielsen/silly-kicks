"""provider keeper-appearance extractors: event-only, NEVER `tracking`/`shot_stopping` (TF-59 PR1).

Spec §5.6 (blast radius / import boundary): the four ``providers/<p>/appearances.py`` extractors
produce the injected ``KeeperAppearances`` port (``silly_kicks.keeper_identity``) from RAW provider
data ALONE. An event-only metric (TF-59 PR2 ``shot_stopping``) must be able to reach these extractors
without dragging in the heavy ``silly_kicks.tracking`` import chain -- the whole reason the resolver
was promoted OUT of ``tracking/`` (amends ADR-078). So each extractor's OWN module-level imports must
stay confined to the port + ``id_compat`` + pandas/stdlib.

This is an **AST module-level** check, NOT a runtime package import: the provider packages
transitively pull ``tracking`` via the pre-existing sibling ``parse.py`` (e.g. sportec's DFL parser),
so importing the package at runtime would always "see" tracking. Only the ``appearances.py`` module's
OWN ``import`` / ``from ... import`` statements matter, and those are exactly what the AST exposes.

``from silly_kicks.providers.<self>.parse import ...`` is PERMITTED (sportec imports ``MatchInfo``
under ``TYPE_CHECKING`` for typing only) -- it is a sibling provider module, not ``tracking``, and it
is never executed at import time.

Every detector carries a planted-violation meta-test so the gate cannot pass vacuously, and a
completeness meta-assertion pins the discovered set to the four expected providers -- a NEW provider
extractor (a 5th ``appearances.py``) fails CI until the expectation is updated deliberately.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import silly_kicks

_ROOT = pathlib.Path(silly_kicks.__file__).resolve().parent
_PROVIDERS = _ROOT / "providers"

#: The providers that ship a keeper-appearance extractor this cycle (spec §5.5). A 5th
#: ``appearances.py`` must be added here deliberately -- see the completeness meta-assertion.
_EXPECTED_PROVIDERS = {"statsbomb", "sportec", "gradientsports", "skillcorner"}

#: Module prefixes an event-only extractor must NEVER import at module level: the heavy tracking
#: package (the promotion's whole point) and the PR2 shot-stopping metric (a consumer, never a dep).
_BANNED_PREFIXES = ("silly_kicks.tracking", "silly_kicks.shot_stopping")

#: The port module every extractor must import (it produces the port via the shared builder).
_REQUIRED_IMPORT = "silly_kicks.keeper_identity"


def _appearance_modules() -> list[pathlib.Path]:
    """Every ``silly_kicks/providers/*/appearances.py`` -- discovered, never hand-listed."""
    return sorted(_PROVIDERS.glob("*/appearances.py"))


def _imported_modules(path: pathlib.Path) -> list[str]:
    """The module names named by *path*'s OWN ``import`` / ``from ... import`` statements (AST).

    ``ast.walk`` sees imports inside an ``if TYPE_CHECKING:`` block too, which is intended: the
    sportec ``providers.sportec.parse`` type-only import is a permitted sibling module, and a real
    banned import hidden under ``TYPE_CHECKING`` must still be caught.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods.extend(alias.name for alias in node.names)
    return mods


def _is_banned(module: str) -> bool:
    return any(module == p or module.startswith(p + ".") for p in _BANNED_PREFIXES)


# --- discovery / completeness --------------------------------------------------------------


def test_discovered_providers_match_the_expected_four():
    """META: the discovered ``appearances.py`` set must equal the four in-scope providers.

    A NEW provider extractor (a 5th ``appearances.py``) fails here until ``_EXPECTED_PROVIDERS``
    is updated deliberately -- so a new boundary cannot slip in un-gated -- and a DELETED one fails
    too (the scans below would silently cover fewer files).
    """
    discovered = {p.parent.name for p in _appearance_modules()}
    assert discovered == _EXPECTED_PROVIDERS, (
        f"discovered provider appearance extractors {sorted(discovered)} != expected "
        f"{sorted(_EXPECTED_PROVIDERS)}; a new/removed appearances.py needs a deliberate update here."
    )


# --- the two content gates, one per extractor ----------------------------------------------


@pytest.mark.parametrize(
    "path",
    _appearance_modules(),
    ids=lambda p: p.parent.name,
)
def test_extractor_never_imports_tracking_or_shot_stopping(path):
    hits = [m for m in _imported_modules(path) if _is_banned(m)]
    assert not hits, (
        f"{path.parent.name}/appearances.py imports banned module(s) {hits}. An event-only "
        "keeper-appearance extractor must NOT drag in silly_kicks.tracking (the promotion's whole "
        "point, amends ADR-078) nor depend on the shot_stopping consumer. Import the port "
        "(silly_kicks.keeper_identity) + id_compat + pandas/stdlib only."
    )


@pytest.mark.parametrize(
    "path",
    _appearance_modules(),
    ids=lambda p: p.parent.name,
)
def test_extractor_imports_the_keeper_identity_port(path):
    mods = _imported_modules(path)
    assert _REQUIRED_IMPORT in mods, (
        f"{path.parent.name}/appearances.py must import {_REQUIRED_IMPORT} -- it produces the "
        "KeeperAppearances port via the shared build_keeper_appearances_from_segments builder."
    )


def test_sibling_provider_parse_import_is_permitted():
    """A ``providers.<self>.parse`` import is a sibling module, NOT tracking -- never flagged.

    sportec imports ``MatchInfo`` from its own ``parse`` under ``TYPE_CHECKING`` (typing only); that
    module transitively pulls tracking, but the AST module-level check never follows it, so it must
    pass the banned-prefix gate.
    """
    sportec = _PROVIDERS / "sportec" / "appearances.py"
    if not sportec.is_file():  # pragma: no cover - guarded by the completeness gate
        pytest.skip("sportec appearances extractor absent")
    mods = _imported_modules(sportec)
    assert "silly_kicks.providers.sportec.parse" in mods, "expected the TYPE_CHECKING MatchInfo import"
    assert not any(_is_banned(m) for m in mods), "the sibling parse import must not be treated as banned"


# --- planted-violation meta-tests (the detectors must actually detect) ----------------------


def test_banned_detector_fires_on_a_planted_violation(tmp_path):
    """META: ``_is_banned`` must flag a real tracking/shot_stopping import and pass a safe one."""
    assert _is_banned("silly_kicks.tracking")
    assert _is_banned("silly_kicks.tracking.features")
    assert _is_banned("silly_kicks.shot_stopping")
    assert _is_banned("silly_kicks.shot_stopping._compute")
    # A prefix collision must NOT false-positive: `silly_kicks.tracking_foo` is a different package.
    assert not _is_banned("silly_kicks.tracking_helpers")
    assert not _is_banned("silly_kicks.keeper_identity")
    assert not _is_banned("silly_kicks.providers.sportec.parse")
    assert not _is_banned("silly_kicks.id_compat")

    planted = tmp_path / "_planted.py"
    planted.write_text("from silly_kicks.tracking import resolve_keeper_identities\n", encoding="utf-8")
    assert any(_is_banned(m) for m in _imported_modules(planted))
    planted.write_text("import silly_kicks.shot_stopping\n", encoding="utf-8")
    assert any(_is_banned(m) for m in _imported_modules(planted))
    planted.write_text(
        "from silly_kicks.keeper_identity import build_keeper_appearances_from_segments\n",
        encoding="utf-8",
    )
    assert not any(_is_banned(m) for m in _imported_modules(planted))


def test_required_import_detector_fires_on_a_planted_violation(tmp_path):
    """META: the keeper_identity requirement must actually notice an absence."""
    planted = tmp_path / "_planted.py"
    planted.write_text("import pandas as pd\n", encoding="utf-8")
    assert _REQUIRED_IMPORT not in _imported_modules(planted)
    planted.write_text(f"from {_REQUIRED_IMPORT} import KeeperSegment\n", encoding="utf-8")
    assert _REQUIRED_IMPORT in _imported_modules(planted)
