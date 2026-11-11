"""``campaign prune``: list and delete the scratch run keys of superseded commits (HFU fit-units spec C3).

Each amend of a campaign commit gives every instance a new run key (``<campaign>-<commit7>-<job>-<tier>``); the
old ones can never resume and stay in the scratch bucket. Launch 7 stopped on the org's private storage limit
because the run keys of 13 amended-away commits were still there, and at the limit HF refuses the bucket delete
too. So the cleanup is a command with a listing, and it runs before the limit is near.

Deletion is irreversible. ``prune`` deletes only the run keys of the commits the operator names, and only keys
that are exactly a run key of a job of THIS campaign. It refuses HEAD's commit and a commit with a non-terminal
job. It never lists the bucket recursively from its root, and it never touches ``_campaign``, ``_staged``,
``_canary`` or another campaign's keys.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

#: One ``batch_bucket_files`` call deletes at most this many paths.
DELETE_CHUNK = 1000
STORAGE_LIMIT_MESSAGE = (
    "the org is at its storage limit: HF refuses the bucket delete; delete a private dataset or model first, "
    "then run prune again"
)


class PruneRefusedError(RuntimeError):
    """`prune` refused before it deleted anything."""


def _run_key_pattern(c: Any) -> re.Pattern[str]:
    pairs = sorted({(j.job, j.tier) for j in c.jobs})
    alts = "|".join(f"{re.escape(job)}-{re.escape(tier)}" for job, tier in pairs)
    return re.compile(rf"^{re.escape(c.name)}-(?P<c7>[0-9a-f]{{7}})-(?:{alts})(?:-r\d+)?$")


def _files(api: Any, bucket: str, run_key: str) -> list[tuple[str, int]]:
    """Every file of one run key: one recursive listing of ``<run_key>/`` (never the bucket root)."""
    base = run_key + "/"
    out = []
    for e in api.list_bucket_tree(bucket, prefix=base, recursive=True):
        if getattr(e, "type", None) == "file":
            if not e.path.startswith(base):
                raise PruneRefusedError(f"the listing of {base} returned {e.path!r}")
            out.append((e.path, int(getattr(e, "size", 0) or 0)))
    return out


def run_keys_by_commit(c: Any, api: Any) -> dict[str, list[dict]]:
    """``{commit7: [{"run_key", "files", "bytes"}]}`` for every top-level bucket folder that is exactly a run key
    of a job of this campaign (a ``-r<N>`` replica store included). Nothing else is listed."""
    pattern = _run_key_pattern(c)
    out: dict[str, list[dict]] = {}
    tops = sorted(
        e.path.rstrip("/")
        for e in api.list_bucket_tree(c.bucket, prefix="", recursive=False)
        if getattr(e, "type", None) != "file"
    )
    for name in tops:
        m = pattern.fullmatch(name)
        if m is None:
            continue
        files = _files(api, c.bucket, name)
        out.setdefault(m.group("c7"), []).append(
            {"run_key": name, "files": len(files), "bytes": sum(size for _p, size in files)}
        )
    return out


def manifest_commits() -> dict[str, int]:
    """``{commit7: number of shipped artifacts}`` from the ``training_commit`` of ``docs/artifacts/manifest.toml``."""
    from scripts.hf_jobs import manifest as mf

    out: dict[str, int] = {}
    for a in mf.load().artifacts.values():
        if a.training_commit:
            out[a.training_commit[:7]] = out.get(a.training_commit[:7], 0) + 1
    return out


def _live_commits(api: Any, campaign: str) -> set[str]:
    from scripts.hf_jobs.launch import TERMINAL

    live = set()
    for info in api.list_jobs(labels={"campaign": campaign}):
        stage = str(getattr(getattr(info, "status", None), "stage", "") or "")
        if stage not in TERMINAL:
            live.add(str((getattr(info, "labels", None) or {}).get("commit", ""))[:7])
    return live


def _is_403(exc: BaseException) -> bool:
    """HF refused the delete at the storage limit: the HTTP status 403, or the limit's own text (the phrase
    that `failures.classify` matches). The digits 403 anywhere else in an error text are not that."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status == 403 or "storage limit reached" in str(exc).lower()


def prune(
    c: Any,
    *,
    head: str,
    delete: bool,
    approved: Sequence[str],
    api: Any,
    manifest_commits: Mapping[str, int],
    say: Callable[[str], object] = print,
) -> dict:
    """List the run keys of every commit other than ``head`` (``delete=False``), or delete the run keys of the
    ``approved`` commits (``delete=True``; each commit is named by its 7 characters).

    Returns ``{"ok", "listing", "deleted"}`` (and ``"stopped"`` after a 403). ``ok`` is False when a file is left
    after a delete or when the run stopped. Every refusal is raised BEFORE the first delete."""
    head7 = head[:7]
    by_commit = run_keys_by_commit(c, api)
    listing = {c7: rows for c7, rows in sorted(by_commit.items()) if c7 != head7}
    result: dict[str, Any] = {"ok": True, "listing": listing, "deleted": {}}
    if not delete:
        for c7, rows in listing.items():
            for r in rows:
                say(f"[prune] {c7} {r['run_key']}: {r['files']} files, {r['bytes'] / 1e9:.2f} GB")
        return result

    commits = list(dict.fromkeys(approved))
    if not commits:
        raise PruneRefusedError("prune --delete needs --i-have-owner-approval <commit7> for each commit to delete")
    if head7 in commits:
        raise PruneRefusedError(f"{head7} is HEAD's commit: its run keys are the current campaign's")
    unknown = [c7 for c7 in commits if c7 not in listing]
    if unknown:
        raise PruneRefusedError(f"no run key of campaign {c.name!r} for commit(s) {unknown}")
    live = sorted(_live_commits(api, c.name) & set(commits))
    if live:
        raise PruneRefusedError(f"commit(s) {live} have a non-terminal job: cancel or wait, then prune")

    for c7 in commits:
        if c7 in manifest_commits:
            say(
                f"[prune] commit {c7} is the training_commit of {manifest_commits[c7]} shipped artifact(s); its run "
                "keys are run records, the artifacts are in the wheel and on the Hub"
            )
        done: list[dict] = []
        result["deleted"][c7] = done
        for r in listing[c7]:
            paths = [p for p, _s in _files(api, c.bucket, r["run_key"])]
            try:
                for i in range(0, len(paths), DELETE_CHUNK):
                    api.batch_bucket_files(c.bucket, delete=paths[i : i + DELETE_CHUNK])
            except Exception as exc:
                if not _is_403(exc):
                    raise
                say(f"[prune] {STORAGE_LIMIT_MESSAGE}")
                if not done:
                    del result["deleted"][c7]
                result.update(ok=False, stopped=STORAGE_LIMIT_MESSAGE)
                return result
            left = len(_files(api, c.bucket, r["run_key"]))
            done.append({**r, "files_left": left})
            say(f"[prune] {c7} {r['run_key']}: {r['files']} files deleted; {left} files left")
            if left:
                result["ok"] = False
    return result
