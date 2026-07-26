"""C4 architecture-diagram readability guard.

Every element box in the C4 diagram (`docs/c4/architecture.dsl`) must carry a SUMMARY
description, not a changelog. Boxes have repeatedly ballooned into multi-ADR prose that
renders as an unreadable wall of text. This gate caps every person / software-system /
container description at 200 characters so the rendered diagram stays human-readable;
per-PR detail belongs in the ADRs + CHANGELOG, not the box.

If this fails: shorten the offending description in `architecture.dsl` to a summary
(<=200 chars) and regenerate `architecture.html` via the c4 skill.
"""

from __future__ import annotations

import re
from pathlib import Path

_DSL = Path(__file__).resolve().parents[1] / "docs" / "c4" / "architecture.dsl"
_MAX_DESCRIPTION_CHARS = 200

# `<identifier> = <kind> "<name>" "<description>" ...` — the description is the 2nd quoted field.
# Relationships (`src -> dst "desc" "tech"`) do not match this shape and are not capped.
_ELEMENT = re.compile(r'\w+ = (person|softwareSystem|container) "[^"]*" "([^"]*)"')


def _descriptions() -> list[tuple[str, str]]:
    text = _DSL.read_text(encoding="utf-8")
    return [(kind, desc) for kind, desc in _ELEMENT.findall(text)]


def test_c4_dsl_file_exists():
    assert _DSL.is_file(), f"C4 DSL not found at {_DSL}"


def test_every_c4_box_description_is_a_summary():
    descriptions = _descriptions()
    # Meta-assertion: the regex must actually be finding elements (guards against a silently
    # broken pattern that would let this gate pass vacuously).
    assert len(descriptions) >= 5, f"expected >=5 element descriptions, found {len(descriptions)}"

    offenders = [(kind, len(desc), desc[:60]) for kind, desc in descriptions if len(desc) > _MAX_DESCRIPTION_CHARS]
    assert not offenders, (
        f"C4 box descriptions exceeding {_MAX_DESCRIPTION_CHARS} chars (shorten to a summary; "
        f"detail belongs in ADRs/CHANGELOG): "
        + "; ".join(f"{kind} ({n} chars) '{head}...'" for kind, n, head in offenders)
    )


# --- Completeness: the model must not silently omit a shipped subpackage -------------------
#
# `silly_kicks.causal` went public at 4.47.0 and was NEVER modelled; the gap survived every
# release until it was found by hand-diffing `ls silly_kicks/*/` against the container list.
# Nothing pinned the diagram to the package tree, so nothing could have caught it -- the same
# incomplete-by-heuristic shape this repo has now deleted twice (the AST id-lint, the
# hand-maintained `_PUBLIC_MODULE_FILES`). Both times the fix was a DERIVED surface plus a
# meta-assertion, which is what this is.

_CONTAINER = re.compile(r'\w+ = container "([^"]*)"')

#: Subpackages deliberately NOT modelled as their own container, each with a stated reason.
#: An entry here is a decision on the record; an omission is a bug.
_UNMODELLED: dict[str, str] = {}


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1] / "silly_kicks"


def _shipped_subpackages() -> set[str]:
    """Every importable subpackage of `silly_kicks` -- derived, never enumerated by hand."""
    return {
        p.name
        for p in _package_root().iterdir()
        if p.is_dir() and not p.name.startswith((".", "_")) and (p / "__init__.py").is_file()
    }


def _modelled_names() -> set[str]:
    """Container names reduced to the subpackage they represent.

    A container name may carry extra scope (`silly_kicks.calibration + scripts/`), so the match is
    on the `silly_kicks.<pkg>` token rather than on string equality.
    """
    text = _DSL.read_text(encoding="utf-8")
    names = set()
    for raw in _CONTAINER.findall(text):
        for token in re.findall(r"silly_kicks\.(\w+)", raw):
            names.add(token)
    return names


def test_every_shipped_subpackage_has_a_c4_container():
    shipped = _shipped_subpackages()
    # Meta-assertion: a broken discovery would make this gate pass vacuously.
    assert len(shipped) >= 8, f"subpackage discovery looks broken, found {sorted(shipped)}"

    missing = sorted(shipped - _modelled_names() - set(_UNMODELLED))
    assert not missing, (
        "shipped subpackage(s) have no C4 container -- the diagram no longer describes the "
        "system. Add a container to docs/c4/architecture.dsl and regenerate architecture.html "
        f"via the c4 skill; if one is deliberately unmodelled, record it in _UNMODELLED with a "
        f"reason: {missing}"
    )


def test_unmodelled_entries_are_real_subpackages():
    """Self-burning-down: an exemption for a package that no longer exists is stale scaffolding."""
    shipped = _shipped_subpackages()
    stale = sorted(set(_UNMODELLED) - shipped)
    assert not stale, f"_UNMODELLED names packages that do not exist: {stale}"
