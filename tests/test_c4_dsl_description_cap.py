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
