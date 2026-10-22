"""Artifact identity (ADR-128 spec 7 criterion 3, decision P3): what "byte-identical" means for a trained
artifact, within one commit and across commits.

Same commit (INV-1): EVERY file is byte-identical. Across commits (a rebaseline vs a shipped release):
every non-JSON payload file is byte-identical, and ``metadata.json`` / ``metrics.json`` are byte-identical
after removing the provenance keys (`PROVENANCE_KEYS`, recursively) -- they name the commit and the run,
so they change on every commit even when the weights do not. ``sklearn_version`` is NOT a provenance key:
a library version change is a real change.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROVENANCE_KEYS: tuple[str, ...] = (
    "training_commit",
    "training_platform",
    "run_commit",
    "run_tree_dirty",
    "run_tree_state",
    "reproducibility_note",
    "generation",
)
#: The JSON files compared after normalization; every other file is compared byte for byte, except
#: `SUMS` (below).
NORMALIZED: tuple[str, ...] = ("metadata.json", "metrics.json")
#: The checksum manifest lists the sha256 of ``metadata.json`` too, so across commits it is compared on
#: its payload lines only (the lines naming a `NORMALIZED` file are dropped); the payload files themselves
#: are still compared byte for byte.
SUMS = "SHA256SUMS"
#: The provenance keys a COMMIT-FREE stage output must not carry (S5.10): every `PROVENANCE_KEYS` entry except
#: ``generation``, which is a content-addressed token (not run-dependent) and stays in the output.
RUN_DEPENDENT: tuple[str, ...] = tuple(k for k in PROVENANCE_KEYS if k != "generation")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strip_keys(doc: Any, keys: Sequence[str]) -> Any:
    if isinstance(doc, Mapping):
        return {k: _strip_keys(v, keys) for k, v in doc.items() if k not in keys}
    if isinstance(doc, list):
        return [_strip_keys(v, keys) for v in doc]
    return doc


def commit_free(files: Mapping[str, bytes]) -> dict[str, bytes]:
    """A stage output's file set made COMMIT-FREE (S5.10): ``SHA256SUMS`` dropped and every `RUN_DEPENDENT`
    provenance key removed from the `NORMALIZED` JSON files; ``generation`` is kept. The JSON is re-encoded
    canonically (sorted keys, indent 2) so the output is deterministic (INV-1) whatever the writer's order.
    ``adopt.stamp_provenance`` re-adds the provenance keys and writes `SUMS`. Idempotent."""
    out: dict[str, bytes] = {}
    for name, data in files.items():
        if name == SUMS:
            continue
        if name in NORMALIZED:
            doc = _strip_keys(json.loads(data.decode("utf-8")), RUN_DEPENDENT)
            data = (json.dumps(doc, sort_keys=True, indent=2, allow_nan=True, default=str) + "\n").encode("utf-8")
        out[name] = data
    return out


def write_sums(directory: str | Path) -> None:
    """Write `SUMS` over every top-level file of ``directory`` (except `SUMS` itself): ``<sha256>  <name>``
    lines, name-sorted, LF. The ONE SHA256SUMS writer (trainers via ``adopt`` and `adopt.stamp_provenance`)."""
    d = Path(directory)
    lines = [f"{_sha256(p.read_bytes())}  {p.name}" for p in sorted(d.iterdir()) if p.is_file() and p.name != SUMS]
    (d / SUMS).write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_sums(directory: str | Path) -> bool:
    """True when `SUMS` lists EXACTLY the directory's other top-level files and every listed sha256 matches."""
    d = Path(directory)
    sums = d / SUMS
    if not sums.is_file():
        return False
    listed: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            listed[parts[-1].lstrip("*")] = parts[0]
    present = {p.name for p in d.iterdir() if p.is_file() and p.name != SUMS}
    return set(listed) == present and all(_sha256((d / name).read_bytes()) == sha for name, sha in listed.items())


def normalized_sums(data: bytes) -> bytes:
    """``SHA256SUMS`` without its `NORMALIZED` entries, lines sorted (the payload part of the manifest)."""
    keep = []
    for line in data.decode("utf-8").replace("\r\n", "\n").splitlines():
        parts = line.split()
        if parts and parts[-1].lstrip("*") not in NORMALIZED:
            keep.append(" ".join(parts))
    return "\n".join(sorted(keep)).encode("utf-8")


def _strip(doc: Any) -> Any:
    if isinstance(doc, Mapping):
        return {k: _strip(v) for k, v in doc.items() if k not in PROVENANCE_KEYS}
    if isinstance(doc, list):
        return [_strip(v) for v in doc]
    return doc


def normalized_json(data: bytes) -> bytes:
    """Canonical JSON of ``data`` without `PROVENANCE_KEYS` (sorted keys, compact separators; NaN kept)."""
    doc = json.loads(data.decode("utf-8"))
    return json.dumps(_strip(doc), sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8")


def artifact_identity(directory: str | Path, *, json_files: Sequence[str] = NORMALIZED) -> dict[str, dict[str, str]]:
    """``{"files": {name: sha256}, "normalized": {name: sha256}}`` over the top-level files of
    ``directory`` (an artifact is flat; a nested path is refused). ``normalized`` covers ``json_files``
    (default `NORMALIZED`; a driver passes its own report files, e.g. ``census.json``) and `SUMS`."""
    d = Path(directory)
    files: dict[str, str] = {}
    normalized: dict[str, str] = {}
    for p in sorted(d.iterdir()):
        if p.is_dir():
            raise ValueError(f"{d}: artifact directories are flat; {p.name}/ is nested")
        data = p.read_bytes()
        files[p.name] = _sha256(data)
        if p.name in json_files:
            normalized[p.name] = _sha256(normalized_json(data))
        elif p.name == SUMS:
            normalized[p.name] = _sha256(normalized_sums(data))
    return {"files": files, "normalized": normalized}


def compare(a: Mapping[str, Mapping[str, str]], b: Mapping[str, Mapping[str, str]], *, same_commit: bool) -> list[str]:
    """The differences between two identities (empty == identical under the rule). Same commit: every
    file's sha256. Across commits: every non-normalized file's sha256 plus the normalized JSON sha256s."""
    diffs: list[str] = []
    fa, fb = a["files"], b["files"]
    for name in sorted(set(fa) ^ set(fb)):
        diffs.append(f"{name}: present in only one artifact")
    for name in sorted(set(fa) & set(fb)):
        if not same_commit and name in a["normalized"] and name in b["normalized"]:
            if a["normalized"].get(name) != b["normalized"].get(name):
                diffs.append(f"{name}: differs after removing the provenance keys")
        elif fa[name] != fb[name]:
            diffs.append(f"{name}: bytes differ")
    return diffs
