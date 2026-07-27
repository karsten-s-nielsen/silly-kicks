"""Run-provenance guard for artifact-writing drivers (scripts/_provenance.py).

Exists because a corpus pass was launched from a tree with three modified drivers while
`git rev-parse HEAD` read clean — the artifacts would have recorded a commit that did not
describe the code that produced them. A verifiable-looking false SHA is worse than none.
"""

from __future__ import annotations

import pytest

import scripts._provenance as mod  # bare import: tests/scripts/ has NO __init__.py


def test_a_dirty_tree_is_REFUSED_by_default():
    prov = {"commit": "abc123def456", "dirty": True, "dirty_files": ["scripts/x.py"]}
    with pytest.raises(SystemExit) as excinfo:
        mod.require_clean_tree(prov, allow_dirty=False)
    msg = str(excinfo.value)
    assert "DIRTY" in msg
    assert "scripts/x.py" in msg, "the refusal must name what is dirty, not just complain"
    assert "abc123def456"[:12] in msg, "and which HEAD would have been falsely recorded"


def test_a_clean_tree_passes_through_unchanged():
    prov = {"commit": "abc123", "dirty": False, "dirty_files": []}
    assert mod.require_clean_tree(prov, allow_dirty=False) is prov


def test_allow_dirty_permits_the_run_but_KEEPS_the_dirty_flag():
    """The escape hatch must not launder the fact. A dev run may proceed; its artifact still says
    so, because otherwise the distinction survives only in someone's memory of the invocation."""
    prov = {"commit": "abc123", "dirty": True, "dirty_files": ["a.py"]}
    out = mod.require_clean_tree(prov, allow_dirty=True)
    assert out["dirty"] is True


def test_a_BUG_in_this_module_propagates_rather_than_reading_as_no_git(monkeypatch):
    """The narrowing that matters. `except Exception` would report a TypeError/AttributeError in
    this module as "git unavailable" and return dirty=True -- a real bug wearing a known failure's
    clothes, which is exactly the pattern ADR-043 removed from the DAS adapter. Only genuine
    git-invocation failures are degradable."""

    def _bug(*_a, **_k):
        raise TypeError("a real bug in this module, not a git failure")

    monkeypatch.setattr(mod, "_git", _bug)
    with pytest.raises(TypeError, match="a real bug"):
        mod.git_provenance()


def test_unknown_provenance_counts_as_DIRTY_not_clean(monkeypatch):
    """No git => untrustworthy, never 'clean'. Defaulting the other way is how an unverifiable
    run acquires a clean-looking record."""

    def _boom(*_a, **_k):
        raise OSError("git not found")  # a genuine, degradable git-invocation failure

    monkeypatch.setattr(mod, "_git", _boom)
    prov = mod.git_provenance()
    assert prov["commit"] == "unknown"
    assert prov["dirty"] is True
    with pytest.raises(SystemExit):
        mod.require_clean_tree(prov, allow_dirty=False)


def test_git_provenance_reports_the_real_tree():
    """Non-vacuity: the helper must actually shell out and return a plausible SHA, not a stub."""
    prov = mod.git_provenance()
    assert set(prov) == {"commit", "dirty", "dirty_files"}
    assert isinstance(prov["dirty"], bool)
    if prov["commit"] != "unknown":
        assert len(prov["commit"]) == 40, "expected a full git SHA"
    assert prov["dirty"] == bool(prov["dirty_files"]) or prov["commit"] == "unknown"
