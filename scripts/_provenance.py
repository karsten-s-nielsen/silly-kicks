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

import pathlib
import platform as _platform
import subprocess
import uuid
from collections.abc import Mapping

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


def git_tree_hash() -> str:
    """The git TREE hash of the working-tree CONTENT -- a stable content identifier even when the tree is dirty vs
    HEAD or carries untracked files (review A-55: the no-flip verdict needs to be tied to the exact tree it ran on,
    e.g. 8a219312, not only to a commit + a dirty flag).

    Computed over a throwaway index (``GIT_INDEX_FILE``) seeded from HEAD then ``add -A`` (tracked edits + untracked
    files), so it matches the content a byte-identical ship of the tree would carry. ``"unknown"`` when git is absent.
    """
    import os
    import tempfile

    try:
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "GIT_INDEX_FILE": str(pathlib.Path(tmp) / "index")}
            subprocess.run(["git", "read-tree", "HEAD"], check=True, capture_output=True, env=env)  # noqa: S607
            subprocess.run(["git", "add", "-A"], check=True, capture_output=True, env=env)  # noqa: S607
            return subprocess.run(
                ["git", "write-tree"],  # noqa: S607 -- git from PATH is the house pattern
                check=True,
                capture_output=True,
                text=True,
                env=env,
            ).stdout.rstrip()
    except _GIT_FAILURES:
        return "unknown"


def git_provenance() -> dict:
    """``{"commit", "tree_state", "dirty", "dirty_files", "platform", "machine"}`` for this run.

    ``platform``/``machine`` answer WHERE, which the commit cannot. Added when ghost was re-fit on
    DGX Spark (aarch64) while `_feature_contract`'s tolerance note still asserted that every
    fingerprinted artifact was produced on x86 -- a claim that had silently become false. A contract
    mismatch on a cross-platform artifact must be diagnosable from the artifact itself. It belongs
    HERE, not in each trainer, for the same reason the commit does: this is the one seam every
    artifact driver already calls, so no driver can forget it.

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
    # Platform identity is resolved FIRST and merged into every return path, including the two
    # git-failure ones. It does not come from git, so dropping it on those paths would blind exactly
    # the runs least able to say where they ran (tarball checkouts, CI images, a box without git).
    host = {"platform": _platform.platform(), "machine": _platform.machine()}
    try:
        commit = _git("rev-parse", "HEAD")
    except _GIT_FAILURES:
        return {"commit": "unknown", "tree_state": "unknown", "dirty": True, "dirty_files": [], **host}
    try:
        porcelain = _git("status", "--porcelain")
    except _GIT_FAILURES:
        return {"commit": commit, "tree_state": "unknown", "dirty": True, "dirty_files": [], **host}
    # Porcelain v1: two status chars + a space, then the path. UNTRACKED files ("??") count as
    # dirty on purpose -- a new, uncommitted module is exactly the kind of thing that changes what
    # runs while HEAD reads clean.
    files = [line[3:] for line in porcelain.splitlines() if line.strip()]
    return {
        "commit": commit,
        "tree_state": "dirty" if files else "clean",
        "dirty": bool(files),
        "dirty_files": files,
        **host,
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


def objective_id(objective: object, inputs: Mapping[str, object], *, prov: Mapping[str, object] | None = None) -> str:
    """Stable identity for an Optuna store: ``"<module>.<qualname>@<commit>:<digest>"`` (D21, spec 12).

    ``ruthless-efficiency`` 0.7.0 requires every ``StoreConfig`` to carry an ``objective_id``; a store
    resumes only when the id matches, so the id must change whenever the objective's CODE or its
    declared INPUTS change. This helper builds it from the two facts already recorded on every
    artifact run:

    - ``<module>.<qualname>`` of the objective CLASS -- passing the class or an instance of it yields
      the same prefix, so a caller need not care which it holds.
    - ``<commit>`` from :func:`git_provenance` and ``<digest>`` from
      :func:`scripts._input_contract.declare_inputs`. The digest is mandatory: a bare mapping (one not
      produced by ``declare_inputs``) raises, so a store can never be keyed on an undeclared input set.

    When the tree is not ``"clean"`` -- ``"dirty"`` OR the ``"unknown"`` state ``git_provenance``
    reports on a box without git -- a ``"+dirty-<uuid4 hex>"`` nonce is appended, so the run never
    resumes across code it cannot faithfully describe (C4, the fail-closed reading of section 12).
    Open the store at :func:`store_path_for` ``(path, id)``: a dirty id gets a FRESH store file there
    (a fixed path would make ruthless REFUSE the reopen -- "written for a different objective").
    Crash recovery within one clean-tree run is unchanged.
    """
    digest = inputs.get("digest")
    if not digest:
        raise ValueError(
            "objective_id needs an `inputs` mapping carrying a `digest` produced by declare_inputs "
            f"(so the store key tracks the declared input set); got keys {sorted(inputs)}."
        )
    cls = objective if isinstance(objective, type) else type(objective)
    prov = prov if prov is not None else git_provenance()
    return f"{cls.__module__}.{cls.__qualname__}@{prov['commit']}:{digest}" + dirty_suffix(prov)


_DIRTY_MARK = "+dirty-"


def dirty_suffix(prov: Mapping[str, object]) -> str:
    """``""`` on a clean tree, else a fresh ``"+dirty-<uuid4 hex>"`` nonce (C4) -- the one D21 identity rule.

    Appended to a store identity, it makes a run on a tree it cannot describe never resume, and makes
    :func:`store_path_for` open it a fresh store. :func:`objective_id` uses it; so does an objective whose identity
    is otherwise not D21's (D2's shard-generation id, C27).
    """
    return "" if prov.get("tree_state") == "clean" else f"{_DIRTY_MARK}{uuid.uuid4().hex}"


def store_path_for(path: str | pathlib.Path, oid: str) -> str:
    """The sqlite store a run with objective id ``oid`` opens (C4 made true, not merely asserted).

    A clean id keeps ``path``: a crash-recovery rerun of the same clean run resumes it, and ruthless refuses a
    DIFFERENT clean id there. A dirty id (``objective_id``'s per-call ``+dirty-<nonce>``) gets a FRESH sibling
    file named by its nonce, so it never resumes AND never reopens a store an earlier call wrote -- which ruthless
    would refuse, so a dirty parallel trainer run's reduce really recomputes instead of raising. ``:memory:``
    passes through.

    Examples
    --------
    >>> store_path_for("out/study.db", "m.Obj@abc:123")
    'out/study.db'
    >>> store_path_for("out/study.db", "m.Obj@abc:123+dirty-0123456789abcdef").replace("\\\\", "/")
    'out/study.dirty-0123456789ab.db'
    """
    if str(path) == ":memory:" or _DIRTY_MARK not in oid:
        return str(path)
    nonce = oid.rsplit(_DIRTY_MARK, 1)[1][:12]
    p = pathlib.Path(path)
    return str(p.with_name(f"{p.stem}.dirty-{nonce}{p.suffix}"))
