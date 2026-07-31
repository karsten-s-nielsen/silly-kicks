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
    # An EXACT set, deliberately: this is the published shape, and a field appearing or vanishing
    # here is an API change for every artifact consumer. `tree_state` joined it additively in
    # 4.72.0 (ADR-052 Task 16b) -- `dirty` kept its exact meaning and value beside it.
    assert set(prov) == {"commit", "tree_state", "dirty", "dirty_files"}
    assert prov["tree_state"] in {"clean", "dirty", "unknown"}
    assert isinstance(prov["dirty"], bool)
    if prov["commit"] != "unknown":
        assert len(prov["commit"]) == 40, "expected a full git SHA"
    assert prov["dirty"] == bool(prov["dirty_files"]) or prov["commit"] == "unknown"


def test_dirty_files_does_not_CHOP_the_first_filename(monkeypatch):
    """Porcelain v1 puts the status in the first two COLUMNS, so an unstaged modification begins
    with a SPACE: `" M CHANGELOG.md"`. `.strip()` on the whole output removed that space from the
    FIRST line only, and the `line[3:]` slice then ate a character -- a refusal read `HANGELOG.md`.

    PATCHED AT `subprocess.run`, NOT at `_git`. The first version of this test replaced `_git`
    itself, so the `.rstrip()` that IS the fix never executed: planting `.strip()` back left the
    suite green (measured -- 12 passed). A test that cannot fail on the bug it was written for is
    the green-by-construction trap, and patching one level lower is what closes it.
    """
    import subprocess as _sp

    porcelain = " M CHANGELOG.md\n M scripts/_driver.py\n?? new_file.py\n"

    def _fake_run(argv, **_kw):
        out = "0123456789abcdef" * 2 + "\n" if argv[1] == "rev-parse" else porcelain
        return _sp.CompletedProcess(argv, 0, stdout=out, stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    got = mod.git_provenance()

    assert got["dirty_files"] == ["CHANGELOG.md", "scripts/_driver.py", "new_file.py"]
    assert got["dirty"] is True
    assert got["tree_state"] == "dirty"


def test_the_porcelain_test_actually_EXERCISES_the_rstrip(monkeypatch):
    """Non-vacuity for the test above, stated as its own property: the leading status column must
    survive into the parse. If `_git` ever goes back to `.strip()`, the first filename loses a
    character and the assertion above fires -- which is only true because `_git` really runs."""
    import inspect

    assert ".rstrip()" in inspect.getsource(mod._git), "the fix itself is gone"
    assert ".stdout.strip()" not in inspect.getsource(mod._git), "strip() eats the first status column"


# ---------------------------------------------------------------------------
# Task 16b: three-state provenance -- clean / dirty / unknown
# ---------------------------------------------------------------------------


def test_a_clean_tree_reports_state_clean(monkeypatch):
    monkeypatch.setattr(mod, "_git", lambda *a: "abc123" if a[0] == "rev-parse" else "")
    prov = mod.git_provenance()
    assert prov["tree_state"] == "clean"
    assert prov["dirty"] is False


def test_a_dirty_tree_reports_state_dirty(monkeypatch):
    monkeypatch.setattr(mod, "_git", lambda *a: "abc123" if a[0] == "rev-parse" else " M scripts/x.py")
    prov = mod.git_provenance()
    assert prov["tree_state"] == "dirty"
    assert prov["dirty"] is True


def test_git_UNAVAILABLE_reports_unknown_but_STILL_refuses(monkeypatch):
    """The honest RECORD and the fail-closed BEHAVIOUR are independent, and this pins both.

    `dirty: true` is a positive claim that uncommitted modifications exist. On a tarball checkout
    or a box without git that claim is false -- an artifact asserting something untrue about its own
    provenance is the defect this module exists to prevent, one level down. So `tree_state` stops
    asserting it, while `dirty` stays True and the refusal is byte-unchanged.
    """

    def _boom(*_a):
        raise OSError("git not found")

    monkeypatch.setattr(mod, "_git", _boom)
    prov = mod.git_provenance()

    assert prov["tree_state"] == "unknown"
    assert prov["dirty"] is True, "fail-closed: unknown provenance is never treated as clean"
    with pytest.raises(SystemExit, match="UNKNOWN provenance"):
        mod.require_clean_tree(prov, allow_dirty=False)


def test_the_unknown_refusal_does_not_claim_the_tree_was_MODIFIED(monkeypatch):
    """The two refusals say different things, which is the whole point of splitting them. The old
    single message listed `(git unavailable)` where the changed files go -- a dirty-tree sentence
    wearing an apology."""

    def _boom(*_a):
        raise OSError("git not found")

    monkeypatch.setattr(mod, "_git", _boom)
    with pytest.raises(SystemExit) as exc:
        mod.require_clean_tree(mod.git_provenance(), allow_dirty=False)

    assert "DIRTY tree" not in str(exc.value)
    assert "nothing is known about it" in str(exc.value)


def test_the_boolean_is_UNCHANGED_for_every_state(monkeypatch):
    """Hyrum. `run_tree_dirty` is read by `_partition.aggregate_manifests` (which ORs it across
    workers) and sits in every artifact already on disk. Widening it to the tri-state string would
    make `bool("clean")` TRUTHY and silently invert the aggregate -- so the two fields stay two.
    Pinned so nobody 'tidies' them into one."""
    for porcelain, expected in (("", False), (" M x.py", True)):
        monkeypatch.setattr(mod, "_git", lambda *a, _p=porcelain: "abc123" if a[0] == "rev-parse" else _p)
        assert mod.git_provenance()["dirty"] is expected
    assert bool("clean") is True, "the reason the boolean cannot simply become the string"
