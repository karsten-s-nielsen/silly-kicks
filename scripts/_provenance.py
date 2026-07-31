"""Run provenance for maintainer drivers that write registered artifacts.

WHY THIS EXISTS. `git rev-parse HEAD` returns the same SHA whether or not the working tree is
modified. A driver stamping that bare SHA onto an artifact therefore records a commit that does
NOT describe the code which produced the numbers -- verifiable-looking and false, which is strictly
worse than recording nothing. That happened: a corpus pass was launched from a tree with three
modified drivers while HEAD read clean.

So the rule is fail-closed: an artifact-writing run REFUSES a dirty tree unless the caller opts in
explicitly, and the dirtiness is recorded either way.
"""

from __future__ import annotations

import subprocess

#: The ONLY degradable failures. `except Exception` here would report a TypeError or AttributeError
#: in this module as "git unavailable" and quietly return dirty=True -- a bug wearing a known
#: failure's clothes, which is the pattern ADR-043 removed from the DAS adapter. A genuine
#: git-invocation failure (missing binary, not a repo, non-zero exit) is degradable; nothing else is.
_GIT_FAILURES = (subprocess.SubprocessError, OSError)


def _git(*args: str) -> str:
    """Run git and return stdout with TRAILING whitespace removed only.

    `rstrip`, not `strip`, and the difference is a measured bug. Porcelain v1 encodes the status in
    the first two COLUMNS, so an unstaged modification begins with a SPACE: `" M CHANGELOG.md"`.
    `.strip()` removes that leading space from the first line of the whole output, and the `line[3:]`
    slice below then chops the first character off the first filename -- so a refusal read
    `HANGELOG.md`, a path that does not exist. Only the FIRST entry was affected, which is exactly
    why it survived: the rest of the list looked fine.

    Scope, stated precisely rather than inflated: `dirty_files` is consumed ONLY by
    `require_clean_tree`'s message. No driver persists it -- they stamp `run_commit` and
    `run_tree_dirty` -- so no committed artifact carries a mangled path. The cost was a diagnostic
    that sends its reader looking for the wrong file at exactly the moment they are trying to find
    out what made the tree dirty. `rev-parse` output has no leading whitespace, so it is unaffected.
    """
    return subprocess.run(  # noqa: S603 -- argv is a fixed literal, never user input
        ["git", *args],  # noqa: S607 -- git from PATH is the house pattern
        capture_output=True,
        text=True,
        check=True,
    ).stdout.rstrip()


def git_provenance() -> dict:
    """``{"commit", "tree_state", "dirty", "dirty_files"}`` for the current tree.

    ``tree_state`` is ``"clean"``, ``"dirty"`` or ``"unknown"``. ``dirty`` is the ORIGINAL boolean,
    unchanged: ``True`` for BOTH ``dirty`` and ``unknown``, because unknown provenance is treated as
    untrustworthy and never as clean.

    TWO fields rather than one widened field, and that is a correctness decision. ``run_tree_dirty``
    is already published in every artifact on disk and is OR-ed across workers by
    `_partition.aggregate_manifests`; ``bool("clean")`` is **truthy**, so putting a tri-state string
    where the boolean lives would silently invert every aggregate.

    The distinction is not cosmetic. ``dirty: true`` asserts that uncommitted modifications EXIST;
    on a tarball checkout or a box without git that assertion is simply false, and an artifact
    making a false claim about its own provenance is the exact failure this module exists to
    prevent -- one level down.
    """
    try:
        commit = _git("rev-parse", "HEAD")
    except _GIT_FAILURES:
        return {"commit": "unknown", "tree_state": "unknown", "dirty": True, "dirty_files": []}
    try:
        porcelain = _git("status", "--porcelain")
    except _GIT_FAILURES:
        return {"commit": commit, "tree_state": "unknown", "dirty": True, "dirty_files": []}
    # Porcelain v1: two status chars + a space, then the path. UNTRACKED files ("??") count as
    # dirty on purpose -- a new, uncommitted module is exactly the kind of thing that changes what
    # runs while HEAD reads clean.
    files = [line[3:] for line in porcelain.splitlines() if line.strip()]
    return {
        "commit": commit,
        "tree_state": "dirty" if files else "clean",
        "dirty": bool(files),
        "dirty_files": files,
    }


def require_clean_tree(prov: dict, *, allow_dirty: bool) -> dict:
    """Return ``prov``, or refuse when the tree is dirty and the caller has not opted in.

    Passing ``allow_dirty=True`` is legitimate for dev smoke runs; the artifact still records
    ``dirty: true`` so the distinction survives into the output rather than living in someone's
    memory of how the run was invoked.
    """
    if prov["dirty"] and not allow_dirty:
        if prov.get("tree_state") == "unknown":
            raise SystemExit(
                "refusing to write a registered artifact with UNKNOWN provenance: git is "
                "unavailable, so the tree could not be inspected. Nothing here claims the tree is "
                "modified -- it claims nothing is known about it, which is equally unusable as a "
                "provenance record. Run from a git checkout, or pass --allow-dirty for a dev run."
            )
        # The `or "(git unavailable)"` fallback that used to sit here is GONE, and its removal is
        # the point: it existed only because this one message had to cover both states, listing an
        # empty file list as a parenthetical apology. The unknown case now has its own branch, so
        # `dirty_files` is non-empty by construction whenever this line runs.
        listed = ", ".join(prov["dirty_files"][:5])
        raise SystemExit(
            f"refusing to write a registered artifact from a DIRTY tree (HEAD={prov['commit'][:12]}): "
            f"{listed}. The recorded commit would not describe the code that ran. "
            "Commit first, or pass --allow-dirty for a dev run (the artifact will be marked dirty)."
        )
    return prov
