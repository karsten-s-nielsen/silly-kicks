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
import re
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
    # The byte count of the saved artifact files (xShot / xCross `metrics.json`). Those files hold the commit and
    # the job platform string, so the count names the run, not the model: it changed by 1 byte between two
    # launches of one code state when the kernel version changed length (HFU fit-units spec 1.2, M7).
    "artifact_size_bytes",
)
#: The JSON files compared after normalization; every other file is compared byte for byte, except
#: `SUMS` (below).
NORMALIZED: tuple[str, ...] = ("metadata.json", "metrics.json")
#: The checksum manifest lists the sha256 of ``metadata.json`` too, so across commits it is compared on
#: its payload lines only (the lines naming a `NORMALIZED` file are dropped); the payload files themselves
#: are still compared byte for byte.
SUMS = "SHA256SUMS"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_sums(data: bytes, normalized: Sequence[str] = NORMALIZED) -> bytes:
    """``SHA256SUMS`` without the entries of the ``normalized`` files (default `NORMALIZED`), lines sorted (the
    payload part of the manifest)."""
    keep = []
    for line in data.decode("utf-8").replace("\r\n", "\n").splitlines():
        parts = line.split()
        if parts and parts[-1].lstrip("*") not in normalized:
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


#: The model card, compared across commits after each 40-hex commit is replaced by a fixed token.
CARD = "MODEL_CARD.md"
_COMMIT_RE = re.compile(rb"\b[0-9a-f]{40}\b")


def cross_commit_differences(a: str | Path, b: str | Path, *, commit_file: str | None = None) -> list[str]:
    """The differences between two artifact folders of DIFFERENT commits (empty == identical). THE rule of
    `campaign compare` (HFU fit-units spec 6 step 3):

    - ``metadata.json``, ``metrics.json`` and ``commit_file`` (the file that the artifact's manifest
      ``commit_field`` names, e.g. ``model.json`` for match_outcome) are compared as canonical JSON without
      `PROVENANCE_KEYS`;
    - `SUMS` is compared on the lines of the files that are not normalized by the line above;
    - `CARD` is compared after each 40-hex commit is replaced by a token (line endings normalized);
    - every other file is compared byte for byte;
    - a file in only one folder is a difference.
    """
    da, db = Path(a), Path(b)
    normalized = (*NORMALIZED, *((commit_file,) if commit_file else ()))
    names_a = {p.name for p in da.iterdir() if p.is_file()}
    names_b = {p.name for p in db.iterdir() if p.is_file()}
    diffs = [f"{name}: present in only one artifact" for name in sorted(names_a ^ names_b)]
    for name in sorted(names_a & names_b):
        xa, xb = (da / name).read_bytes(), (db / name).read_bytes()
        if name in normalized:
            if normalized_json(xa) != normalized_json(xb):
                diffs.append(f"{name}: differs after removing the provenance keys")
        elif name == SUMS:
            if normalized_sums(xa, normalized) != normalized_sums(xb, normalized):
                diffs.append(f"{name}: payload lines differ")
        elif name == CARD:
            if _card(xa) != _card(xb):
                diffs.append(f"{name}: differs after replacing the commit")
        elif xa != xb:
            diffs.append(f"{name}: bytes differ")
    return diffs


def _card(data: bytes) -> bytes:
    return _COMMIT_RE.sub(b"<commit>", data.replace(b"\r\n", b"\n"))
