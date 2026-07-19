"""gkdv -> tracking: PUBLIC SEAMS ONLY, and never the reverse (ADR-037)."""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2] / "silly_kicks"
GKDV = ROOT / "gkdv"
TRACKING = ROOT / "tracking"

#: Private tracking modules gkdv is permitted to import, each with a recorded reason.
#: Additive-only; every entry is a deliberate decision.
#
# EVERY entry must be a NAMED, CONFINED PORT: a private with no public meaning, reached from
# exactly ONE gkdv module whose job is to be that boundary. This is deliberately NOT a debt
# register -- an entry is not "a private we have not promoted yet, see the exit condition",
# because that framing let two genuinely-public seams sit here indefinitely while the package
# advertised "public tracking seams only". A private that DOES have public meaning gets
# promoted instead of listed (4.53.0 promoted both: `_id_compat` -> `silly_kicks.id_compat`,
# `_gk_resolve.defended_goal_x` -> `silly_kicks.tracking.defended_goal_x`).
#
# So the review question for an addition is not "is the debt documented?" but "is this symbol
# genuinely internal, and is it confined to one port module?" If the answer to either half is
# no, promote it. `CONFINED_TO` below makes the second half structural, not prose.
ALLOW_PRIVATE: dict[str, str] = {
    # `_pin_attacking_direction` is a DAS internal with no public meaning -- it pins the
    # direction convention `accessible-space` expects, which is a fact about that library's
    # input contract, not a concept gkdv or any other consumer should be reasoning about.
    # Promoting it would export an implementation detail of an optional dependency.
    #
    # It is CONFINED to `_das_port.py`, and the port exists so the structural direction-pinning
    # guard runs on every CI leg WITHOUT the optional `accessible-space` extra installed. An
    # import of this module from any other gkdv file is a real violation -- route it via the
    # port. The sibling `get_individual_das` is NOT covered here: it is already public and the
    # port imports it as such.
    "silly_kicks.tracking._das": (
        "_pin_attacking_direction is a DAS-internal convention with no public meaning; "
        "CONFINED to gkdv/_das_port.py, which exists so the direction-pinning guard runs "
        "without the optional accessible-space extra"
    ),
}

#: The one module each allowlisted private is confined TO. Every ALLOW_PRIVATE entry must
#: appear here -- see `test_every_allowlisted_private_declares_its_confinement`.
CONFINED_TO: dict[str, str] = {"silly_kicks.tracking._das": "_das_port.py"}


def _imported_tracking_symbols(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        # ALLOW_PRIVATE is keyed on the FULL dotted path, and the check matches on the full
        # path too: a tail-only match ("_id_compat") would also wave through some unrelated
        # `other.package.tracking._id_compat`.
        if isinstance(node, ast.ImportFrom) and node.module and "tracking" in node.module:
            tail = node.module.split(".")[-1]
            if tail.startswith("_") and node.module not in ALLOW_PRIVATE:
                hits.append(node.module)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if "tracking._" in a.name and a.name not in ALLOW_PRIVATE:
                    hits.append(a.name)
    return hits


def _all_private_tracking_imports(path: pathlib.Path) -> list[str]:
    """Every private tracking module imported by *path*, allowlisted or not.

    ``_imported_tracking_symbols`` deliberately filters the allowlist out, so it cannot see
    where an ALLOWED private is used -- which is exactly what the confinement rule is about.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "tracking" in node.module:
            if node.module.split(".")[-1].startswith("_"):
                hits.append(node.module)
        elif isinstance(node, ast.Import):
            hits.extend(a.name for a in node.names if "tracking._" in a.name)
    return hits


@pytest.mark.parametrize(
    "path",
    # rglob, NOT glob -- for the same reason the tracking scan below is recursive: a flat
    # glob silently stops scanning the moment gkdv grows its first subpackage, and an
    # unscanned module is indistinguishable from a compliant one. Enumeration, not heuristic.
    sorted(GKDV.rglob("*.py")),
    ids=lambda p: p.relative_to(GKDV).as_posix(),
)
def test_confined_exemptions_stay_in_their_one_module(path):
    """A CONFINED exemption that spreads is no longer confined -- and nothing else notices.

    The allowlist is keyed on the module being imported, so once an entry exists the plain
    lint waves it through from ANY gkdv module. That is correct for a repo-wide mandated
    seam and wrong for a confined one, so the two kinds are enforced differently.
    """
    for module in _all_private_tracking_imports(path):
        expected = CONFINED_TO.get(module)
        assert expected is None or path.name == expected, (
            f"{path.name} imports {module}, which is a CONFINED exemption allowed only in "
            f"{expected}. Route it through that module instead of widening the exemption."
        )


def test_confinement_detector_fires_on_a_planted_violation(tmp_path):
    """META: the confinement check must actually detect. It scans a different symbol set
    from the main lint (allowlisted imports are invisible to that one), so it needs its own
    proof that it sees them at all."""
    assert CONFINED_TO, "meta-test is vacuous without at least one confined entry"
    module = next(iter(CONFINED_TO))
    planted = tmp_path / "_planted_confined.py"
    planted.write_text(f"from {module} import something\n", encoding="utf-8")
    assert _all_private_tracking_imports(planted) == [module]
    # ...and it must NOT flag a public tracking import.
    planted.write_text("from silly_kicks.tracking import get_individual_das\n", encoding="utf-8")
    assert _all_private_tracking_imports(planted) == []


@pytest.mark.parametrize(
    "path",
    sorted(GKDV.rglob("*.py")),  # rglob: see test_confined_exemptions_stay_in_their_one_module
    ids=lambda p: p.relative_to(GKDV).as_posix(),
)
def test_gkdv_imports_only_public_tracking_seams(path):
    hits = _imported_tracking_symbols(path)
    assert not hits, (
        f"{path.name}: imports PRIVATE tracking module(s) {hits}. Import the public seam "
        "(silly_kicks.tracking.<name>) or add an ALLOW_PRIVATE entry with a reason."
    )


@pytest.mark.parametrize(
    "path",
    # rglob, NOT glob: the flat glob left the `pitch_control/` and `preprocess/` subpackages
    # (17 modules) entirely unscanned, so the direction rule was enforced by heuristic rather
    # than by enumeration -- the same incomplete-by-heuristic failure mode as the retired
    # ADR-019 AST lint. The id disambiguates the three `__init__.py` files.
    sorted(TRACKING.rglob("*.py")),
    ids=lambda p: p.relative_to(TRACKING).as_posix(),
)
def test_tracking_never_imports_gkdv(path):
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        # Import nodes ONLY: ast.Global / ast.Nonlocal also carry `.names`, but as plain
        # strings rather than ast.alias, and blanket-walking them raises AttributeError.
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        mod = getattr(node, "module", None) or ""
        names = [a.name for a in node.names]
        assert "gkdv" not in mod and not any("gkdv" in n for n in names), (
            f"{path.name}: tracking/ must NEVER import gkdv/ -- the probe consumes ghost "
            "positions as DATA (a targets DataFrame) precisely to keep this direction closed."
        )


def test_detector_fires_on_a_planted_private_import(tmp_path):
    """META: the detector must actually detect. Without this the lint can silently pass."""
    planted = tmp_path / "_planted.py"
    planted.write_text("from silly_kicks.tracking._ghost_gk import GRID_X_MAX\n", encoding="utf-8")
    assert _imported_tracking_symbols(planted), "detector failed to flag a private import"


def test_allowlist_entries_are_actually_honored(tmp_path):
    """META: the allowlist and the detector must agree on key SHAPE.

    They did not: the detector matched on the module tail while the entries are full dotted
    paths, so every allowlisted import was still flagged. A key-shape mismatch in either
    direction makes the allowlist decorative -- the entries would either never apply (a
    permanently red gate) or, if someone "fixed" it by keying on tails, apply too broadly.
    """
    assert ALLOW_PRIVATE, "meta-test is vacuous without at least one entry"
    for module in ALLOW_PRIVATE:
        planted = tmp_path / "_allowed.py"
        planted.write_text(f"from {module} import something\n", encoding="utf-8")
        assert not _imported_tracking_symbols(planted), (
            f"{module} is allowlisted but still flagged -- ALLOW_PRIVATE keys must be full "
            "dotted module paths, matching how the detector compares them."
        )


def test_every_allowlisted_private_declares_its_confinement():
    """META: the "named, confined port" rule, enforced instead of described.

    The allowlist previously mixed confined ports with un-confined entries, and only the
    confined ones were checked -- so an entry could be allowed package-wide simply by being
    left out of ``CONFINED_TO``, which is the opposite of the intent. Requiring every entry to
    name its one module means a future addition cannot quietly become a package-wide waiver.
    """
    undeclared = sorted(set(ALLOW_PRIVATE) - set(CONFINED_TO))
    assert not undeclared, (
        f"{undeclared}: every ALLOW_PRIVATE entry must name the ONE gkdv module it is confined "
        "to in CONFINED_TO. A private with no natural single port is a private with public "
        "meaning -- promote it to the public surface instead of allowlisting it."
    )
    orphaned = sorted(set(CONFINED_TO) - set(ALLOW_PRIVATE))
    assert not orphaned, f"{orphaned}: confinement declared for a module that is not allowlisted"


def test_gkdv_package_is_non_empty():
    """META: pins the gate's surface -- an empty package would make the lint vacuous."""
    modules = sorted(p.relative_to(GKDV).as_posix() for p in GKDV.rglob("*.py"))
    assert len(modules) >= 5, f"expected the full gkdv module set, found {modules}"


#: Source tokens that mean "this test loaded the bundled weights".
_BUNDLED_WEIGHT_MARKERS = ('model="default"', "model='default'", "from_variant(")


def _pins_bundled_weights(text: str) -> bool:
    """True if *text* pins the bundled `default` weights rather than a fixture model."""
    return any(marker in text for marker in _BUNDLED_WEIGHT_MARKERS)


def test_no_gkdv_test_pins_the_bundled_default_weights():
    """The spec's retained parallelism rule (§9 item 4), enforced rather than trusted.

    gkdv numerics must come from synthetic/fixture models only. A test that loads the
    bundled `default` variant couples this suite to artifact-metadata integrity (PR-2's
    load() chirality enforcement is fail-closed) and will move when the weights move.
    """
    here = pathlib.Path(__file__).resolve()
    offenders = []
    for path in sorted(here.parent.glob("test_*.py")):
        # This scanner necessarily contains the markers it searches for; scanning itself
        # would make the gate permanently red. Its detector is covered by the meta-test below.
        if path.name == here.name:
            continue
        if _pins_bundled_weights(path.read_text(encoding="utf-8")):
            offenders.append(path.name)
    assert not offenders, (
        f"{offenders}: gkdv tests must use _fitted_model()[0], never the bundled weights "
        "(spec §9 item 4 parallelism rule)."
    )


def test_bundled_weight_detector_fires_on_planted_pins():
    """META: the weights scanner must actually detect.

    Its live scan set is small while gkdv's suite is being built out, so without this the
    gate could pass by scanning nothing meaningful.
    """
    for planted in ('m = model="default"', "m = model='default'", "M.from_variant('sc_extended')"):
        assert _pins_bundled_weights(planted), f"detector missed a planted pin: {planted!r}"
    assert not _pins_bundled_weights("m = _fitted_model()[0]"), "detector false-positives on the fixture model"
