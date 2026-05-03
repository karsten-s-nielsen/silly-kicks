"""Loose structural check on TODO.md to mirror lakehouse format.

PR-S20 restructured TODO.md into a lakehouse-style "On Deck" table with TF-N
deferred-feature entries (each carrying academic citations). These tests guard
the shape so a future contributor can't quietly drop the structure. Specific
TF-N IDs are NOT asserted -- shipped IDs leave the table per the
no-breadcrumbs-on-shipped-work convention.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TODO_PATH = REPO_ROOT / "TODO.md"


def test_todo_has_on_deck_section():
    text = TODO_PATH.read_text(encoding="utf-8")
    assert "## On Deck" in text


def test_todo_has_size_legend():
    text = TODO_PATH.read_text(encoding="utf-8")
    for kw in ["Monstah", "Wicked", "Dunkin'"]:
        assert kw in text, f"missing {kw} in size legend"


def test_todo_has_active_cycle_or_archive_marker():
    text = TODO_PATH.read_text(encoding="utf-8")
    assert "## Active Cycle" in text or "## Tech" in text


def test_todo_has_at_least_five_distinct_tf_entries():
    """Bench-shape gate -- at least 5 distinct TF-N IDs appear in TODO.md.

    Drops the previous IDs-by-name assertion so shipped TFs (TF-1, TF-6, TF-8,
    TF-9, TF-11, TF-12, TF-22) aren't required to remain as breadcrumbs.
    """
    text = TODO_PATH.read_text(encoding="utf-8")
    distinct = set(re.findall(r"\bTF-\d+\b", text))
    assert len(distinct) >= 5, f"expected at least 5 distinct TF-N entries; got {sorted(distinct)}"
