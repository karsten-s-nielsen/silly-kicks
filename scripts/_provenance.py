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
    return subprocess.run(  # noqa: S603 -- argv is a fixed literal, never user input
        ["git", *args],  # noqa: S607 -- git from PATH is the house pattern
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def git_provenance() -> dict:
    """``{"commit", "dirty", "dirty_files"}`` for the current tree.

    ``commit`` is ``"unknown"`` when git is unavailable (a tarball checkout, say), in which case
    ``dirty`` is ``True`` -- unknown provenance is treated as untrustworthy, never as clean.
    """
    try:
        commit = _git("rev-parse", "HEAD")
    except _GIT_FAILURES:
        return {"commit": "unknown", "dirty": True, "dirty_files": []}
    try:
        porcelain = _git("status", "--porcelain")
    except _GIT_FAILURES:
        return {"commit": commit, "dirty": True, "dirty_files": []}
    # Porcelain v1: two status chars + a space, then the path. UNTRACKED files ("??") count as
    # dirty on purpose -- a new, uncommitted module is exactly the kind of thing that changes what
    # runs while HEAD reads clean.
    files = [line[3:] for line in porcelain.splitlines() if line.strip()]
    return {"commit": commit, "dirty": bool(files), "dirty_files": files}


def require_clean_tree(prov: dict, *, allow_dirty: bool) -> dict:
    """Return ``prov``, or refuse when the tree is dirty and the caller has not opted in.

    Passing ``allow_dirty=True`` is legitimate for dev smoke runs; the artifact still records
    ``dirty: true`` so the distinction survives into the output rather than living in someone's
    memory of how the run was invoked.
    """
    if prov["dirty"] and not allow_dirty:
        listed = ", ".join(prov["dirty_files"][:5]) or "(git unavailable)"
        raise SystemExit(
            f"refusing to write a registered artifact from a DIRTY tree (HEAD={prov['commit'][:12]}): "
            f"{listed}. The recorded commit would not describe the code that ran. "
            "Commit first, or pass --allow-dirty for a dev run (the artifact will be marked dirty)."
        )
    return prov
