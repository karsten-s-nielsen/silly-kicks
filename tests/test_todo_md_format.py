"""Loose structural check on TODO.md to mirror lakehouse format.

PR-S20 restructured TODO.md into a lakehouse-style "On Deck" table with TF-1..TF-10
deferred-feature entries (each carrying academic citations). These tests guard the
shape so a future contributor can't quietly drop the structure.
"""

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


def test_todo_lists_tf1_through_tf6_at_minimum():
    text = TODO_PATH.read_text(encoding="utf-8")
    for tf in ["TF-1", "TF-2", "TF-3", "TF-4", "TF-5", "TF-6"]:
        assert tf in text, f"{tf} entry missing"
